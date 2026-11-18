"""Generate publication figures for Bayesian Hyperparameter Optimization."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
HPO_RECORD = ROOT / 'results' / 'hpo' / 'optuna_trials_record.json'
FIGURES = ROOT / 'reports' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

with open(HPO_RECORD) as f:
    hpo_data = json.load(f)

trials = hpo_data['trials']
trial_nums = [t['trial_number'] for t in trials]
trial_vals = [t['value_macro_f1'] for t in trials]

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 11})

# =============================================================================
# Plot 1: Optimization History
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
ax.plot(trial_nums, trial_vals, marker='o', color='#3498db', linewidth=1.8, label='Trial Objective Value (CV Macro F1)')

# Cumulative best
cum_best = np.maximum.accumulate(trial_vals)
ax.step(trial_nums, cum_best, where='post', color='#e74c3c', linewidth=2.5, linestyle='--', label='Best Objective Value Found')

best_idx = int(np.argmax(trial_vals))
ax.scatter([trial_nums[best_idx]], [trial_vals[best_idx]], color='#27ae60', s=140, zorder=5, edgecolor='black', label=f'Best: Trial {best_idx} ({trial_vals[best_idx]:.4f})')

ax.set_xlabel('Bayesian Optimization Trial Number')
ax.set_ylabel('Validation Objective (Macro F1)')
ax.set_title('Bayesian Optimization Convergence History (TPE Sampler, 20 Trials)')
ax.set_xticks(np.arange(0, 20, 2))
ax.set_ylim(0.59, 0.65)
ax.legend(frameon=True, facecolor='white', framealpha=0.95)
plt.tight_layout()
p1 = FIGURES / '11_bayesian_optimization_trials.png'
plt.savefig(p1)
plt.close()
print("Saved:", p1)

# =============================================================================
# Plot 2: Untuned vs Tuned Model Performance Comparison
# =============================================================================
untuned_records = json.load(open(ROOT / 'results' / 'run_01' / 'hist_gradient_boosting' / 'metrics.json'))
tuned_records = json.load(open(ROOT / 'results' / 'tuned_hist_gradient_boosting' / 'metrics.json'))

untuned_test = [r for r in untuned_records if r['split'] == 'student_test'][0]['metrics']['all_points']
tuned_test = [r for r in tuned_records if r['split'] == 'student_test'][0]['metrics']['all_points']

untuned_summary = json.load(open(ROOT / 'results' / 'run_01' / 'hist_gradient_boosting' / 'summary.json'))
tuned_summary = json.load(open(ROOT / 'results' / 'tuned_hist_gradient_boosting' / 'summary.json'))

metrics_names = ['CV Macro F1', 'Test Macro F1', 'Test Top-1 Acc (%)', 'Log Loss']
untuned_vals = [
    untuned_summary['cv_macro_f1_mean'],
    untuned_test['macro_f1'],
    untuned_test['top1_accuracy'] * 100,
    untuned_test['categorical_log_loss_with_miss_state']
]
tuned_vals = [
    tuned_summary['cv_macro_f1_mean'],
    tuned_test['macro_f1'],
    tuned_test['top1_accuracy'] * 100,
    tuned_test['categorical_log_loss_with_miss_state']
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

# F1 & Accuracy
x = np.arange(3)
width = 0.35
ax1.bar(x - width/2, [untuned_vals[0], untuned_vals[1], untuned_vals[2]/100], width, label='Untuned HistGradientBoosting', color='#7f8c8d', edgecolor='black')
ax1.bar(x + width/2, [tuned_vals[0], tuned_vals[1], tuned_vals[2]/100], width, label='Bayesian Tuned HistGradientBoosting', color='#2ecc71', edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(['CV Macro F1', 'Test Macro F1', 'Test Accuracy'], rotation=15)
ax1.set_ylim(0.5, 0.9)
ax1.set_ylabel('Score')
ax1.set_title('Generalization & Accuracy (Higher is Better)')
ax1.legend(frameon=True, facecolor='white', framealpha=0.95)

# Add value labels
for i in range(3):
    ax1.text(i - width/2, [untuned_vals[0], untuned_vals[1], untuned_vals[2]/100][i] + 0.015, f"{[untuned_vals[0], untuned_vals[1], untuned_vals[2]/100][i]:.3f}", ha='center', fontsize=9)
    ax1.text(i + width/2, [tuned_vals[0], tuned_vals[1], tuned_vals[2]/100][i] + 0.015, f"{[tuned_vals[0], tuned_vals[1], tuned_vals[2]/100][i]:.3f}", ha='center', fontsize=9, fontweight='bold')

# Log Loss
ax2.bar(['Untuned', 'Tuned'], [untuned_vals[3], tuned_vals[3]], color=['#e74c3c', '#27ae60'], width=0.45, edgecolor='black')
ax2.set_ylabel('Categorical Log Loss (Lower is Better)')
ax2.set_title('Probability Calibration on Student-Test')
ax2.set_ylim(0.5, 0.8)
for i, v in enumerate([untuned_vals[3], tuned_vals[3]]):
    ax2.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
p2 = FIGURES / '12_untuned_vs_tuned_comparison.png'
plt.savefig(p2)
plt.close()
print("Saved:", p2)
