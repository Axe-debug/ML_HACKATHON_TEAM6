"""Bayesian Hyperparameter Optimization for HistGradientBoosting on PathSnap."""
import os, sys, time, json
from pathlib import Path
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import HistGradientBoostingClassifier

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from load_dataset import load_training
from model_training import score_predictions, SEED

HPO_DIR = ROOT_DIR / 'results' / 'hpo'
HPO_DIR.mkdir(parents=True, exist_ok=True)

# Fixed group folds to prevent leakage
SEARCH_FOLDS = [0, 1, 2] # 3 folds for efficient Bayesian search

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.2, log=True),
        'max_iter': trial.suggest_int('max_iter', 80, 220, step=20),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 63),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 60),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.01, 10.0, log=True),
        'random_state': SEED
    }

    fold_f1s = []
    for fold in SEARCH_FOLDS:
        train, val = load_training(str(ROOT_DIR / 'processed'), validation_fold=fold)
        X_train = train['X'].reshape(-1, train['X'].shape[-1])
        y_train = train['y'].ravel()
        X_val = val['X'].reshape(-1, val['X'].shape[-1])

        clf = HistGradientBoostingClassifier(**params)
        clf.fit(X_train, y_train)

        prob = clf.predict_proba(X_val)[:, list(clf.classes_).index(1)]
        metrics, _ = score_predictions(val, prob)
        f1 = metrics['all_points']['macro_f1']
        fold_f1s.append(f1)

    return float(np.mean(fold_f1s))

def run_hpo(n_trials=20):
    print("="*70)
    print("BAYESIAN HYPERPARAMETER OPTIMIZATION: HistGradientBoosting")
    print(f"Number of Trials: {n_trials} | Search Strategy: TPE (Tree-structured Parzen Estimator)")
    print(f"Objective Function: Mean Cross-Validation Macro F1 across training folds {SEARCH_FOLDS}")
    print("="*70)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    total_time = time.time() - t0

    print("\n" + "="*70)
    print("BAYESIAN OPTIMIZATION COMPLETED")
    print(f"Total Search Time: {total_time:.1f}s ({total_time/60:.2f} mins)")
    print(f"Best Trial Value (CV Macro F1): {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("="*70)

    # Save search record
    trials_record = []
    for t in study.trials:
        trials_record.append({
            'trial_number': t.number,
            'value_macro_f1': t.value,
            'params': t.params,
            'state': str(t.state),
            'duration_seconds': t.duration.total_seconds() if t.duration else None
        })

    record_file = HPO_DIR / 'optuna_trials_record.json'
    with open(record_file, 'w') as f:
        json.dump({
            'model': 'hist_gradient_boosting',
            'search_strategy': 'Bayesian (TPE)',
            'objective': 'Group-Aware Cross-Validation Macro F1',
            'n_trials': n_trials,
            'total_time_seconds': total_time,
            'best_cv_macro_f1': study.best_value,
            'best_params': study.best_params,
            'trials': trials_record
        }, f, indent=2)
    print(f"Saved search record to: {record_file}")

    # Refit tuned model on full 5 folds and student test
    print("\nRefitting Tuned HistGradientBoosting across all 5 folds + Student-Test...")
    tuned_dir = ROOT_DIR / 'results' / 'tuned_hist_gradient_boosting'
    tuned_dir.mkdir(parents=True, exist_ok=True)

    best_params = study.best_params.copy()
    best_params['random_state'] = SEED

    records = []
    import joblib, platform, sklearn, hashlib

    (tuned_dir / 'configuration.json').write_text(json.dumps(best_params, indent=2))
    schema = json.loads((ROOT_DIR / 'processed' / 'schema.json').read_text())
    (tuned_dir / 'provenance.json').write_text(json.dumps({
        'model': 'hist_gradient_boosting_tuned',
        'seed': SEED,
        'python': platform.python_version(),
        'sklearn': sklearn.__version__,
        'best_params': best_params
    }, indent=2))

    splits = [(f'fold_{fold}', fold) for fold in range(5)] + [('student_test', None)]

    for label, fold in splits:
        train, val = load_training(str(ROOT_DIR / 'processed'), fold)
        X_train = train['X'].reshape(-1, train['X'].shape[-1])
        y_train = train['y'].ravel()
        X_val = val['X'].reshape(-1, val['X'].shape[-1])

        clf = HistGradientBoostingClassifier(**best_params)
        t_start = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_s = time.perf_counter() - t_start

        t_start = time.perf_counter()
        prob = clf.predict_proba(X_val)[:, list(clf.classes_).index(1)]
        inference_s = time.perf_counter() - t_start

        metrics, pred = score_predictions(val, prob)
        train_prob = clf.predict_proba(X_train)[:, list(clf.classes_).index(1)]
        train_metrics, _ = score_predictions(train, train_prob)

        rec = {
            'split': label,
            'fit_seconds': fit_s,
            'inference_seconds': inference_s,
            'inference_ms_per_observation': inference_s / len(pred) * 1000,
            'metrics': metrics,
            'fit_subset_metrics': train_metrics,
            'smoke': False
        }
        records.append(rec)
        np.savez_compressed(
            tuned_dir / f'{label}_predictions.npz',
            positive_probability=prob.reshape(val['y'].shape),
            predicted_arc_id=pred,
            reference_arc_id=val['reference_arc_id'],
            candidate_arc_ids=val['candidate_arc_ids'],
            trip_id=val['trip_id'],
            label_reliable=val['label_reliable']
        )
        if fold is None:
            joblib.dump(clf, tuned_dir / 'model.joblib')
        print(f"[Tuned HistGradientBoosting] {label}: Macro F1={metrics['all_points']['macro_f1']:.4f}, Top1={metrics['all_points']['top1_accuracy']:.4f}")

    cv = [r['metrics']['all_points']['macro_f1'] for r in records if r['split'].startswith('fold_')]
    cv_acc = [r['metrics']['all_points']['top1_accuracy'] for r in records if r['split'].startswith('fold_')]
    (tuned_dir / 'metrics.json').write_text(json.dumps(records, indent=2))
    (tuned_dir / 'summary.json').write_text(json.dumps({
        'model': 'hist_gradient_boosting_tuned',
        'cv_macro_f1_mean': float(np.mean(cv)),
        'cv_macro_f1_std': float(np.std(cv)),
        'cv_top1_acc_mean': float(np.mean(cv_acc)),
        'completed_folds': len(cv),
        'student_test_completed': True
    }, indent=2))

    print("\nTuned model refitted and saved successfully in:", tuned_dir)

if __name__ == '__main__':
    run_hpo(n_trials=20)
