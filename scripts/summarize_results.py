"""Print consolidated benchmark table across all 5 models."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results' / 'run_01'
MODELS = ['logistic', 'hist_gradient_boosting', 'mlp', 'extra_trees', 'random_forest']

print(f"{'Model':<24} | {'CV Macro F1':<16} | {'CV Top-1':<10} | {'Test Macro F1':<14} | {'Test Top-1':<10} | {'Log Loss':<10} | {'Inf Latency':<12}")
print("-" * 105)

for m in MODELS:
    summary = json.load(open(RESULTS / m / 'summary.json'))
    records = json.load(open(RESULTS / m / 'metrics.json'))
    test = [r for r in records if r['split'] == 'student_test'][0]
    ap = test['metrics']['all_points']
    
    cv_f1 = f"{summary['cv_macro_f1_mean']:.4f} ± {summary['cv_macro_f1_std']:.4f}"
    cv_acc = f"{summary['cv_top1_acc_mean']*100:.2f}%"
    test_f1 = f"{ap['macro_f1']:.4f}"
    test_acc = f"{ap['top1_accuracy']*100:.2f}%"
    ll = f"{ap['categorical_log_loss_with_miss_state']:.4f}"
    inf_ms = f"{test['inference_seconds']*1000:.1f} ms"
    
    print(f"{m:<24} | {cv_f1:<16} | {cv_acc:<10} | {test_f1:<14} | {test_acc:<10} | {ll:<10} | {inf_ms:<12}")
