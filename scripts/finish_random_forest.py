"""Complete student_test final refit for random_forest and write summary.json."""
import sys, time, json, joblib
from pathlib import Path
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from model_training import make_model, score_predictions
from load_dataset import load_training

def main():
    rf_dir = ROOT_DIR / 'results' / 'run_01' / 'random_forest'
    metrics_file = rf_dir / 'metrics.json'
    with open(metrics_file) as f:
        records = json.load(f)
        
    print(f"Loaded {len(records)} existing fold records for random_forest.")
    
    # Check if student_test already in records
    if any(r['split'] == 'student_test' for r in records):
        print("student_test already in records!")
        return

    print("Refitting Random Forest on all 80 training trips (1.09M rows)...")
    train, validation = load_training(str(ROOT_DIR / 'processed'), None)
    model = make_model('random_forest')
    X = train['X'].reshape(-1, train['X'].shape[-1])
    y = train['y'].ravel()
    
    t0 = time.perf_counter()
    model.fit(X, y)
    fit_s = time.perf_counter() - t0
    print(f"Fit completed in {fit_s:.1f}s.")
    
    t0 = time.perf_counter()
    prob = model.predict_proba(validation['X'].reshape(-1, validation['X'].shape[-1]))[:, list(model.classes_).index(1)]
    inference_s = time.perf_counter() - t0
    
    metrics, pred = score_predictions(validation, prob)
    training_prob = model.predict_proba(X)[:, list(model.classes_).index(1)]
    training_metrics, _ = score_predictions(train, training_prob)
    
    record = dict(
        split='student_test',
        fit_seconds=fit_s,
        inference_seconds=inference_s,
        inference_ms_per_observation=inference_s / len(pred) * 1000,
        metrics=metrics,
        fit_subset_metrics=training_metrics,
        loss_curve=None,
        warnings=[],
        smoke=False
    )
    records.append(record)
    metrics_file.write_text(json.dumps(records, indent=2))
    
    np.savez_compressed(
        rf_dir / 'student_test_predictions.npz',
        positive_probability=prob.reshape(validation['y'].shape),
        predicted_arc_id=pred,
        reference_arc_id=validation['reference_arc_id'],
        candidate_arc_ids=validation['candidate_arc_ids'],
        trip_id=validation['trip_id'],
        label_reliable=validation['label_reliable']
    )
    joblib.dump(model, rf_dir / 'model.joblib')
    
    cv = [r['metrics']['all_points']['macro_f1'] for r in records if r['split'].startswith('fold_')]
    cv_acc = [r['metrics']['all_points']['top1_accuracy'] for r in records if r['split'].startswith('fold_')]
    (rf_dir / 'summary.json').write_text(json.dumps(dict(
        model='random_forest',
        smoke=False,
        cv_macro_f1_mean=float(np.mean(cv)) if cv else None,
        cv_macro_f1_std=float(np.std(cv)) if cv else None,
        cv_top1_acc_mean=float(np.mean(cv_acc)) if cv_acc else None,
        completed_folds=len(cv),
        student_test_completed=True
    ), indent=2))
    
    print("Random Forest student_test successfully evaluated and saved!")
    print(f"Student-test Macro F1: {metrics['all_points']['macro_f1']:.4f}, Top-1 Acc: {metrics['all_points']['top1_accuracy']:.4f}")

if __name__ == '__main__':
    main()
