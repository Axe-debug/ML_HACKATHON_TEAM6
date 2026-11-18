"""Generate comparative benchmark plots for all 5 PathSnap models."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results' / 'run_01'
FIGURES = ROOT / 'reports' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

MODELS = ['logistic', 'hist_gradient_boosting', 'mlp', 'extra_trees', 'random_forest']
DISPLAY_NAMES = ['Logistic Reg', 'Hist Gradient Boost', 'MLP (Neural Net)', 'Extra Trees', 'Random Forest']

data = {}
for m in MODELS:
    summary = json.load(open(RESULTS / m / 'summary.json'))
    records = json.load(open(RESULTS / m / 'metrics.json'))
    test = [r for r in records if r['split'] == 'student_test'][0]
    data[m] = {
        'summary': summary,
        'test': test,
        'records': records
    }

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 11})

# =============================================================================
# Plot 1: Macro F1-Score Comparison (CV vs Student-Test)
# =============================================================================
cv_f1 = [data[m]['summary']['cv_macro_f1_mean'] for m in MODELS]
cv_f1_err = [data[m]['summary']['cv_macro_f1_std'] for m in MODELS]
test_f1 = [data[m]['test']['metrics']['all_points']['macro_f1'] for m in MODELS]

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
x = np.arange(len(MODELS))
width = 0.35

rects1 = ax.bar(x - width/2, cv_f1, width, yerr=cv_f1_err, capsize=4,
                label='5-Fold CV Macro F1 (Train)', color='#2980b9', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x + width/2, test_f1, width,
                label='Held-out Student-Test Macro F1', color='#27ae60', edgecolor='black', alpha=0.9)

ax.set_ylabel('Macro F1-Score')
ax.set_title('Macro F1-Score Comparison Across 5 Model Families')
ax.set_xticks(x)
ax.set_xticklabels(DISPLAY_NAMES, rotation=15)
ax.set_ylim(0.5, 0.75)
ax.legend(frameon=True, facecolor='white', framealpha=0.95)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
p1 = FIGURES / '07_model_comparison_macro_f1.png'
plt.savefig(p1)
plt.close()
print("Saved:", p1)

# =============================================================================
# Plot 2: Top-1 Accuracy Comparison (CV vs Student-Test)
# =============================================================================
cv_acc = [data[m]['summary']['cv_top1_acc_mean'] * 100 for m in MODELS]
test_acc = [data[m]['test']['metrics']['all_points']['top1_accuracy'] * 100 for m in MODELS]

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
rects1 = ax.bar(x - width/2, cv_acc, width, label='5-Fold CV Top-1 Accuracy (Train)',
                color='#8e44ad', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x + width/2, test_acc, width, label='Held-out Student-Test Top-1 Accuracy',
                color='#e67e22', edgecolor='black', alpha=0.9)

ax.set_ylabel('Observation-Level Top-1 Accuracy (%)')
ax.set_title('Top-1 Accuracy Comparison Across 5 Model Families')
ax.set_xticks(x)
ax.set_xticklabels(DISPLAY_NAMES, rotation=15)
ax.set_ylim(65, 88)
ax.legend(frameon=True, facecolor='white', framealpha=0.95)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
p2 = FIGURES / '08_model_comparison_top1_accuracy.png'
plt.savefig(p2)
plt.close()
print("Saved:", p2)

# =============================================================================
# Plot 3: Log Loss vs Inference Latency
# =============================================================================
log_losses = [data[m]['test']['metrics']['all_points']['categorical_log_loss_with_miss_state'] for m in MODELS]
latencies = [data[m]['test']['inference_seconds'] * 1000 for m in MODELS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

ax1.bar(x, log_losses, color='#c0392b', edgecolor='black', alpha=0.85)
ax1.set_ylabel('Categorical Log Loss (Lower is Better)')
ax1.set_title('Observation Log Loss on Student-Test')
ax1.set_xticks(x)
ax1.set_xticklabels(DISPLAY_NAMES, rotation=25)
ax1.set_ylim(0.5, 0.85)
for i, v in enumerate(log_losses):
    ax1.text(i, v + 0.008, f'{v:.4f}', ha='center', fontsize=9)

ax2.bar(x, latencies, color='#16a085', edgecolor='black', alpha=0.85)
ax2.set_ylabel('Inference Time for 19,160 Obs (ms)')
ax2.set_title('Inference Latency on Student-Test')
ax2.set_xticks(x)
ax2.set_xticklabels(DISPLAY_NAMES, rotation=25)
for i, v in enumerate(latencies):
    ax2.text(i, v + 12, f'{v:.0f}ms', ha='center', fontsize=9)

plt.tight_layout()
p3 = FIGURES / '09_model_comparison_log_loss_and_latency.png'
plt.savefig(p3)
plt.close()
print("Saved:", p3)

# =============================================================================
# Plot 4: Authentic MLP Epoch-Wise Training Loss History
# =============================================================================
mlp_loss = data['mlp']['records'][0]['loss_curve'] # Fold 0 epoch curve
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
epochs = np.arange(1, len(mlp_loss) + 1)
ax.plot(epochs, mlp_loss, color='#2c3e50', linewidth=2.5, marker='o', markersize=4, label='MLP Training Loss (Binary Cross-Entropy)')
ax.set_xlabel('Training Epoch')
ax.set_ylabel('Loss Value')
ax.set_title('Authentic MLP Training History Across Epochs (Recorded from model.loss_curve_)')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(frameon=True, facecolor='white', framealpha=0.95)
plt.tight_layout()
p4 = FIGURES / '10_mlp_epoch_loss_curve.png'
plt.savefig(p4)
plt.close()
print("Saved:", p4)

print("All comparative plots generated successfully!")
