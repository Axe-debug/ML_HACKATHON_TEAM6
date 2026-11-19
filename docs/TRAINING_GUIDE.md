# Shared model training contract

## Setup
Clone this repository, then use Python 3.12 or newer in a local environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
```

Data is already present under processed/. You do not need raw files, Folium, or the original PDFs
to run model comparisons. The preprocessing and map validation scripts require their separate
requirements and original raw files. Do not run those scripts as a prerequisite to training.
If NumPy versions differ, the NPZ arrays remain portable; record actual dependency versions.
Only load trusted model files (joblib uses pickle).

## Required notebook deliverable

Every model must have an executed notebook under notebooks/: logistic.ipynb,
hist_gradient_boosting.ipynb, random_forest.ipynb, extra_trees.ipynb, and mlp.ipynb.
Each owner creates their assigned notebooks; no invented experiment outputs are supplied.
Install requirements-notebooks.txt and launch jupyter lab. Set the notebook working directory
to the repository root and add scripts/ to sys.path. Import model_training.run and call
run(model_name, "processed", "results/run_01", folds=[0,1,2,3,4], final=True).
Retain actual outputs, warnings, metrics, appropriate plots and your own analysis. Keep large
predictions/weights outside notebooks in ignored artifacts. Check for secrets, private local
paths and unrelated outputs before review. The CLI below is an optional reproduction aid,
not a substitute for the required model notebook.

## Run your assigned model
Model names: logistic, hist_gradient_boosting, random_forest, extra_trees, mlp.
Replace random_forest below with your assignment:

```bash
# Runtime/interface check only: first 200 observations per split, never final evidence.
python scripts/model_training.py --model random_forest --folds 0 --smoke --out scratch/smoke
# Full comparison: 5 trip folds, then refit all fitting trips and evaluate student-test.
python scripts/model_training.py --model random_forest --final --out results/run_01
```

To run folds separately, choose fresh output directories per invocation; --folds 0 1 selects
specific folds. Do not claim five-fold results until all five complete. The script refuses to
silently overwrite an output folder. For reruns use run_02, etc. Smoke results are marked.
MLP warnings about nonconvergence must be retained and investigated; do not claim successful
convergence solely because fit returned. Epoch-loss plots are only valid for actual recorded
MLP loss histories; the starter does not save per-fold histories yet. Add this in the MLP branch.
The starter caps OpenMP/BLAS to two threads by default; tree estimators also use two workers.
These are initial reproducible settings, not claims of optimal hyperparameters.

The shared runner implements initial comparison only. Do not conduct independent Bayesian
searches before the five-model comparison; the owner will tune the selected model afterwards.
If adjusting initial settings due to memory/convergence, retain both configurations and record
why; discuss changes affecting the shared data contract with the owner.

## Dataset and fitting
Each NPZ contains observations x 8 candidates x 10 features. The fitting loader selects only
observations with reliable proxy labels and a present reference candidate, then retains all 8
rows. Flatten the first two axes for binary candidate classification; use class-1 probability
at inference. Logistic/MLP use fold-fitted StandardScaler pipelines; trees use raw features.
No global scaler is supplied. No reference metadata enters X.
The fitting set is 136,582 observations / 1,092,656 candidate rows across all training trips.
CV held-out trips retain all observations. The 19,160 student-test observations are evaluation
only, not part of model fitting. Do not use random candidate-row train/test splits.

## Shared metrics
The starter saves all-point proxy-label metrics and the reliable-label subset with coverage.
Candidate misses are retained as observation errors. Binary candidate log loss is reported
separately from observation-level categorical log loss.
- Top-1: choose the candidate with highest positive-class score, then compare its arc to proxy.
- Macro precision/recall/F1: classes are (trip ID, arc ID); use union of true and predicted classes
  within the evaluated split. Fold Macro F1 values are averaged equally and std is population std.
  This is our explicit convention, pending any organiser clarification.
- Candidate miss rate: reference arc absent from the eight generated candidates.
- Categorical log loss: floor candidate scores at 1e-12, append a fixed 1e-12 none-of-candidates
  weight, and normalise all 9 weights. Misses use that ninth state and are heavily penalised.
  This is an engineering scoring convention, not a learned missing-candidate probability.
- Inference timing: predict_proba on candidate features only. It excludes I/O, candidate generation,
  feature extraction and HMM. Full system latency must be measured later.
- Route continuity is explicitly N/A until agreed graph-based scoring is added. Do not fabricate
  it or interpret a smooth map as a measured continuity score. Owner/Madhu will add shared scoring
  and all five models can be rescored using saved candidate probabilities.
- Saved fit-subset metrics are on filtered fitting observations, while validation includes harder
  points. Their gap alone does not prove overfitting. Discuss coverage and compare like-for-like
  reliable subsets before drawing conclusions; add per-trip or per-class diagnostics as needed.

## Required handoff from EVERY model owner
Deliver executed model notebook(s). Inside your model-specific result directory retain:
- configuration.json and provenance.json (seeds, versions, data fingerprints, command).
- metrics.json (each completed fold and student test), summary.json, and captured warnings/logs.
- per-split predictions NPZ containing scores, chosen IDs, reference IDs, trip IDs and quality flags.
- model.joblib for the final all-training fit, including its preprocessing pipeline.
- analysis.md in your own words: suitability, loss/objective, important settings, observed strengths,
  weaknesses, evidence about fit, student-test results, timing and limitations. Use actual results.
- Any model-specific source changes, meaningful plots, and an exact reproduction command.

Runner execution does not create final report content or certify labels. All labels are derived
proxies. Investigate documented high-miss trips (00000054 and 00000033) with the owner before
claiming physical map-matching accuracy. Keep every experiment actually attempted documented.

## Repository hygiene
Commit source, small measured summaries/configs/analysis through your branch when authorised.
Large predictions and weights are ignored by default; retain them locally for final artifact handoff
and use an agreed artifact/release location later. Never commit .venv, caches, scratch runs, raw
archives, secrets or duplicate dataset ZIPs. No teammate needs a separate preprocessed dataset.

## Public training handoff
This repository uses a training-only export without GPS coordinates, absolute timestamps,
projection origins or original row IDs. Features, labels, candidate IDs, folds and fitting
masks match the full local outputs. Full originals remain local. The optional geographic
validator requires raw data and full regenerated artifacts, not the stripped public export.
Use scripts/export_training_data.py before publishing regenerated full data.
