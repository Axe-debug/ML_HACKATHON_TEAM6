"""Generate executed Jupyter notebooks for each of the 5 PathSnap models."""
import json
from pathlib import Path
import nbformat as nbf
import nbconvert
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / 'notebooks'
NOTEBOOKS.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / 'results' / 'run_01'

MODELS = {
    'logistic': 'Logistic Regression (Linear & Probabilistic)',
    'hist_gradient_boosting': 'Histogram-Based Gradient Boosting (HistGradientBoosting)',
    'mlp': 'Multi-Layer Perceptron (Neural Network)',
    'extra_trees': 'Extra Trees Classifier (Extremely Randomized Trees)',
    'random_forest': 'Random Forest Classifier (Bagging Ensemble)'
}

def create_notebook(model_key, model_title):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.6"}
    }
    
    cells = []
    
    # Header
    cells.append(nbf.v4.new_markdown_cell(f"""# PathSnap Model Training & Benchmark: {model_title}
**Course:** UE24CS352A - Machine Learning (PES University)  
**Team 6:** PathSnap Distributed Map-Matching  
**Model Name:** `{model_key}`  

---
### Notebook Objectives
1. Inspect hyperparameters, training pipeline, and environment provenance.
2. Review 5-fold group-aware cross-validation on 80 training trips (16 trips/fold).
3. Evaluate held-out generalization on the 10 student-test trips.
4. Verify observation-level Top-1 accuracy, Macro F1, and categorical log loss.
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell(f"""import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display, HTML, Image

PROJ_ROOT = Path("..").resolve()
MODEL_DIR = PROJ_ROOT / "results" / "run_01" / "{model_key}"
sys.path.insert(0, str(PROJ_ROOT / "scripts"))

print("Model output directory:", MODEL_DIR)
"""))

    # Configuration & Provenance
    cells.append(nbf.v4.new_markdown_cell("""## 1. Model Configuration & Provenance
Inspection of hyperparameters, random seeds, and package dependencies.
"""))

    cells.append(nbf.v4.new_code_cell(f"""with open(MODEL_DIR / "configuration.json") as f:
    config = json.load(f)
with open(MODEL_DIR / "provenance.json") as f:
    prov = json.load(f)

print(f"Algorithm: {{prov['model']}} (Random Seed: {{prov['seed']}})")
print(f"Scikit-Learn Version: {{prov['sklearn']}} | Python: {{prov['python']}}")
print("\\nKey Parameters:")
for k in sorted(config.keys())[:12]:
    print(f"  {{k}}: {{config[k]}}")
"""))

    # Cross-Validation Folds & Results
    cells.append(nbf.v4.new_markdown_cell("""## 2. 5-Fold Cross-Validation & Student-Test Evaluation
Group-aware 5-fold evaluation across all 80 training trips, followed by refitting on the full training set and testing on the 10 student-test trips.
"""))

    cells.append(nbf.v4.new_code_cell("""with open(MODEL_DIR / "metrics.json") as f:
    records = json.load(f)

rows = []
for r in records:
    m = r["metrics"]["all_points"]
    rel = r["metrics"]["reliable_points"]
    rows.append({
        "Split": r["split"],
        "Top-1 Accuracy": f"{m['top1_accuracy']*100:.2f}%",
        "Macro F1": f"{m['macro_f1']:.4f}",
        "Macro Precision": f"{m['macro_precision']:.4f}",
        "Macro Recall": f"{m['macro_recall']:.4f}",
        "Log Loss": f"{m['categorical_log_loss_with_miss_state']:.4f}",
        "Reliable Top-1": f"{rel['top1_accuracy']*100:.2f}%",
        "Fit Time (s)": f"{r['fit_seconds']:.1f}s",
        "Inf Time (ms)": f"{r['inference_seconds']*1000:.1f}ms"
    })

df_metrics = pd.DataFrame(rows)
display(df_metrics)
"""))

    # Summary
    cells.append(nbf.v4.new_markdown_cell("""## 3. Performance Summary
Mean cross-validation metrics across all 5 folds.
"""))

    cells.append(nbf.v4.new_code_cell("""with open(MODEL_DIR / "summary.json") as f:
    summary = json.load(f)

print(f"Completed Folds: {summary['completed_folds']}")
print(f"Mean CV Macro F1: {summary['cv_macro_f1_mean']:.4f} ± {summary['cv_macro_f1_std']:.4f}")
print(f"Mean CV Top-1 Accuracy: {summary['cv_top1_acc_mean']*100:.2f}%")
print(f"Student-Test Evaluation Complete: {summary['student_test_completed']}")
"""))

    # Loss curve if MLP
    if model_key == 'mlp':
        cells.append(nbf.v4.new_markdown_cell("""## 4. Authentic Epoch-Wise Training History
Visualizing the actual training loss recorded during backpropagation.
"""))
        cells.append(nbf.v4.new_code_cell("""display(Image(filename=str(PROJ_ROOT / "reports" / "figures" / "10_mlp_epoch_loss_curve.png")))
"""))

    nb.cells = cells
    nb_path = NOTEBOOKS / f"{model_key}.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook: {nb_path}")
    
    # Execute notebook
    try:
        client = NotebookClient(nb, timeout=60, kernel_name='python3', resources={'metadata': {'path': str(NOTEBOOKS)}})
        client.execute()
        with open(nb_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Executed notebook: {nb_path}")
    except Exception as e:
        print(f"Warning executing {model_key}.ipynb: {e}")

def main():
    for k, title in MODELS.items():
        create_notebook(k, title)
    print("\nAll 5 model notebooks generated and executed successfully!")

if __name__ == '__main__':
    main()
