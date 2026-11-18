"""Master execution script to train and benchmark all 5 models for PathSnap."""
import sys, os, time, json
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from model_training import run, SEED

MODELS = ['logistic', 'hist_gradient_boosting', 'mlp', 'extra_trees', 'random_forest']

def main():
    out_dir = ROOT_DIR / 'results' / 'run_01'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PATHSNAP 5-MODEL BENCHMARK EXECUTION")
    print("Models to train:", MODELS)
    print("Output directory:", out_dir)
    print("="*70)
    
    start_total = time.time()
    results_summary = {}
    
    for model_name in MODELS:
        model_out = out_dir / model_name
        summary_file = model_out / 'summary.json'
        
        if summary_file.exists():
            print(f"\n>>> Model '{model_name}' already has completed results in {model_out}. Skipping rerun.")
            with open(summary_file) as f:
                results_summary[model_name] = json.load(f)
            continue
            
        print(f"\n>>> Starting Training & Benchmarking for: '{model_name}'")
        t0 = time.time()
        
        # Run 5 folds + final fit on student-test
        run(
            model_name=model_name,
            root=str(ROOT_DIR / 'processed'),
            out=str(out_dir),
            folds=[0, 1, 2, 3, 4],
            final=True,
            smoke=False
        )
        
        elapsed = time.time() - t0
        print(f">>> Completed '{model_name}' in {elapsed:.1f}s ({elapsed/60:.2f} mins)")
        
        with open(summary_file) as f:
            results_summary[model_name] = json.load(f)
            
    total_elapsed = time.time() - start_total
    print("\n" + "="*70)
    print(f"ALL 5 MODELS COMPLETED in {total_elapsed/60:.2f} mins!")
    print("Summary of Results:")
    print(json.dumps(results_summary, indent=2))
    print("="*70)

if __name__ == '__main__':
    main()
