"""Shared initial model comparisons. No tuning or HMM is performed here."""
from pathlib import Path
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
import argparse, json, time, platform, hashlib, sys
import numpy as np
from load_dataset import load_training

SEED = 212

def make_model(name):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    models = {
        'logistic': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=SEED)),
        'hist_gradient_boosting': lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=.1, max_leaf_nodes=31, l2_regularization=1., random_state=SEED),
        'random_forest': lambda: RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2, n_jobs=6, random_state=SEED),
        'extra_trees': lambda: ExtraTreesClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2, n_jobs=6, random_state=SEED),
        # Vectorized mini-batching (batch_size=2048) on 1M rows for stable multi-threaded BLAS/AVX operations.
        'mlp': lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=2048, max_iter=40, early_stopping=False, random_state=SEED)),
    }
    return models[name]()

def score_predictions(data, positive_probability):
    """Arc Macro F1 uses (trip, arc) labels; misses are included in all-point scores."""
    p = np.asarray(positive_probability, dtype=float).reshape(data['y'].shape)
    if not (np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))):
        raise ValueError('Invalid probabilities')
    selected = np.argmax(p, axis=1)
    pred = data['candidate_arc_ids'][np.arange(len(p)), selected]
    truth = data['reference_arc_id']
    correct = (pred == truth)
    hit = data['y'].any(axis=1)
    # Tiny explicit none-of-candidates mass prevents dropping candidate misses.
    eps = 1e-12
    weights = np.maximum(p, eps)
    denom = weights.sum(axis=1) + eps
    true_probability = np.where(hit, (weights * data['y']).sum(axis=1) / denom, eps / denom)
    
    def metrics(mask):
        if not mask.any():
            return None
        trip = data['trip_id'][mask]
        t = truth[mask]
        q = pred[mask]
        # Compute multiclass macro metrics without inventing globally shared arc IDs.
        key = lambda arc: np.char.add(np.char.add(trip.astype(str), ':'), arc.astype(str))
        tk = key(t)
        pk = key(q)
        classes = np.unique(np.r_[tk, pk])
        ti = np.searchsorted(classes, tk)
        pi = np.searchsorted(classes, pk)
        tp = np.bincount(ti[ti == pi], minlength=len(classes))
        tc = np.bincount(ti, minlength=len(classes))
        pc = np.bincount(pi, minlength=len(classes))
        precision = np.divide(tp, pc, out=np.zeros(len(classes)), where=pc > 0)
        recall = np.divide(tp, tc, out=np.zeros(len(classes)), where=tc > 0)
        f1 = np.divide(2 * tp, tc + pc, out=np.zeros(len(classes)), where=(tc + pc) > 0)
        return dict(
            observations=int(mask.sum()),
            top1_accuracy=float(correct[mask].mean()),
            macro_precision=float(precision.mean()),
            macro_recall=float(recall.mean()),
            macro_f1=float(f1.mean()),
            candidate_miss_rate=float((~hit[mask]).mean()),
            categorical_log_loss_with_miss_state=float(-np.log(true_probability[mask]).mean())
        )
        
    allpoints = np.ones(len(p), dtype=bool)
    binary_p = np.clip(p, eps, 1 - eps)
    y = data['y']
    result = dict(
        all_points=metrics(allpoints),
        reliable_points=metrics(data['label_reliable']),
        reliable_coverage=float(data['label_reliable'].mean()),
        candidate_binary_log_loss=float(-(y * np.log(binary_p) + (1 - y) * np.log(1 - binary_p)).mean()),
        route_continuity=None,
        route_continuity_reason='Needs raw road network and an agreed path/connectivity definition; do not invent a value.'
    )
    return result, pred

def run(model_name, root, out, folds, final=False, smoke=False):
    import sklearn, joblib, warnings
    out = Path(out) / model_name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} already contains results; choose a new --out')
    out.mkdir(parents=True, exist_ok=True)
    records = []
    config = make_model(model_name).get_params(deep=True)
    (out / 'configuration.json').write_text(json.dumps(config, indent=2, default=str))
    schema = json.loads((Path(root) / 'schema.json').read_text())
    provenance = dict(
        model=model_name,
        seed=SEED,
        python=platform.python_version(),
        numpy=np.__version__,
        sklearn=sklearn.__version__,
        schema=schema,
        smoke=smoke,
        argv=sys.argv,
        dataset_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(root).rglob('*.npz'))}
    )
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    splits = [(f'fold_{fold}', fold) for fold in folds] + ([('student_test', None)] if final else [])
    
    for label, fold in splits:
        train, validation = load_training(root, fold)
        if smoke:
            # A runtime check only; never report these as full experiment results.
            for data in [train, validation]:
                for key in data:
                    data[key] = data[key][:200]
        model = make_model(model_name)
        X = train['X'].reshape(-1, train['X'].shape[-1])
        y = train['y'].ravel()
        
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            start = time.perf_counter()
            model.fit(X, y)
            fit_s = time.perf_counter() - start
            
        loss_curve = None
        if hasattr(model, 'loss_curve_'):
            loss_curve = [float(v) for v in model.loss_curve_]
        elif hasattr(model, 'named_steps') and 'mlpclassifier' in model.named_steps:
            loss_curve = [float(v) for v in model.named_steps['mlpclassifier'].loss_curve_]
            
        start = time.perf_counter()
        prob = model.predict_proba(validation['X'].reshape(-1, validation['X'].shape[-1]))[:, list(model.classes_).index(1)]
        inference_s = time.perf_counter() - start
        
        metrics, pred = score_predictions(validation, prob)
        training_probability = model.predict_proba(X)[:, list(model.classes_).index(1)]
        training_metrics, _ = score_predictions(train, training_probability)
        
        record = dict(
            split=label,
            fit_seconds=fit_s,
            inference_seconds=inference_s,
            inference_ms_per_observation=inference_s / len(pred) * 1000,
            metrics=metrics,
            fit_subset_metrics=training_metrics,
            loss_curve=loss_curve,
            warnings=[str(w.message) for w in caught],
            smoke=smoke
        )
        records.append(record)
        (out / 'metrics.json').write_text(json.dumps(records, indent=2))
        np.savez_compressed(
            out / f'{label}_predictions.npz',
            positive_probability=prob.reshape(validation['y'].shape),
            predicted_arc_id=pred,
            reference_arc_id=validation['reference_arc_id'],
            candidate_arc_ids=validation['candidate_arc_ids'],
            trip_id=validation['trip_id'],
            label_reliable=validation['label_reliable']
        )
        if final and fold is None:
            joblib.dump(model, out / 'model.joblib')
        print(f"[{model_name}] {label} done: Macro F1={metrics['all_points']['macro_f1']:.4f}, Top1 Acc={metrics['all_points']['top1_accuracy']:.4f}, Fit={fit_s:.1f}s", flush=True)
        
    cv = [r['metrics']['all_points']['macro_f1'] for r in records if r['split'].startswith('fold_')]
    cv_acc = [r['metrics']['all_points']['top1_accuracy'] for r in records if r['split'].startswith('fold_')]
    (out / 'summary.json').write_text(json.dumps(dict(
        model=model_name,
        smoke=smoke,
        cv_macro_f1_mean=float(np.mean(cv)) if cv else None,
        cv_macro_f1_std=float(np.std(cv)) if cv else None,
        cv_top1_acc_mean=float(np.mean(cv_acc)) if cv_acc else None,
        completed_folds=len(cv),
        student_test_completed=any(r['split'] == 'student_test' for r in records)
    ), indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['logistic', 'hist_gradient_boosting', 'random_forest', 'extra_trees', 'mlp'])
    p.add_argument('--root', default='processed')
    p.add_argument('--out', default='results')
    p.add_argument('--folds', nargs='+', type=int, choices=range(5), default=list(range(5)))
    p.add_argument('--final', action='store_true', help='Also refit all training trips and evaluate student-test')
    p.add_argument('--smoke', action='store_true', help='First 200 eligible observations per split; diagnostic only')
    a = p.parse_args()
    run(a.model, a.root, a.out, a.folds, a.final, a.smoke)
