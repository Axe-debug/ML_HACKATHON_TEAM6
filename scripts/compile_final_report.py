"""Compile all report sections into a single comprehensive Technical Report."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"

def main():
    sec1 = """# PathSnap: Distributed Map-Matching & Sequential Route Reconstruction
## Complete Technical Report & System Documentation

**Course:** UE24CS352A — Machine Learning  
**Institution:** PES University, Department of Computer Science & Engineering  
**Team 6:** Yaswanth, Madhu, Hrisheek, Joshwin  
**Winning System:** Tuned HistGradientBoostingClassifier + Graph-Constrained First-Order Hidden Markov Model (Viterbi Trellis Decoding)  
**Final Test Performance:** Top-1 Accuracy: **84.70%** (Gain: +3.51%) | Route Continuity: **96.98%** (Gain: +7.76%)  
**Distributed Architecture:** Two-Laptop LAN Client-Server (FastAPI Backend + Leaflet/Streamlit Evaluator UI)

---

# Section 1: Executive Summary & Problem Formulation

## 1.1 Problem Context & Objective
Modern intelligent transportation systems (ITS), navigation platforms, and autonomous fleets rely heavily on global positioning system (GPS) feeds. However, consumer GPS sensors in urban environments suffer from severe multipath reflections, satellite occlusion, and sensor jitter. Point-wise matching of GPS fixes to road network graphs frequently leads to erratic parallel road oscillation, topological disconnections, and ghost turns at junctions.

The **PathSnap** project requires designing, training, optimizing, and deploying an end-to-end distributed system that:
1. Conducts structured, mathematically rigorous Exploratory Data Analysis (EDA) on trajectory and road network datasets.
2. Formulates feature extraction that maps noisy GPS observations to $K=8$ candidate road segments.
3. Evaluates and benchmarks **5 distinct machine learning model families** using strictly group-aware 5-fold cross-validation.
4. Tunes the winning model using **Bayesian Hyperparameter Optimization** (Optuna).
5. Implements a **First-Order Hidden Markov Model (HMM)** with dynamic programming (**Viterbi Algorithm**) to enforce physical vehicle kinematics and road network graph topology.
6. Deploys the complete system across a **two-laptop distributed architecture** featuring a FastAPI backend on Laptop 1 and an interactive Evaluator UI on Laptop 2, guaranteeing zero data leakage of reserved evaluation ground truth.

## 1.2 System Pipeline Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │ Raw Input Data (.track, .nodes, .arcs, .route)│
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │  Local Equirectangular Metric Projection     │
                  │   (R = 6,371,008.8m, Max Distortion < 1.14%)  │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    KDTree Midpoint + Expanded Ball Search    │
                  │         K=8 Candidate Road Segments          │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    10-Dimensional Spatial-Kinematic Features │
                  │   (Dist, Heading Cos, Speed, Gaps, Ranks)    │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Tuned HistGradientBoosting Classifier     │
                  │  (Optuna TPE: lr=0.056, leaf=15, min_leaf=48) │
                  └──────────────────────┬───────────────────────┘
                                         │  Calibrated Emission Probs
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │  Graph-Constrained HMM & Viterbi Decoding    │
                  │ (Dijkstra Shortest Path, Direct Bonus, Gaps) │
                  └──────────────────────┬───────────────────────┘
                                         │  Optimal Sequence s*
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │   Distributed REST API & Leaflet Evaluator   │
                  │  (Laptop 1 FastAPI <===> Laptop 2 Client UI) │
                  └──────────────────────────────────────────────┘
```
"""

    sec2 = (REPORTS_DIR / "SECTION_2_EDA_AND_PREPROCESSING.md").read_text(encoding="utf-8")
    sec3 = (REPORTS_DIR / "SECTION_3_MODEL_EXPERIMENTATION.md").read_text(encoding="utf-8")
    sec4 = (REPORTS_DIR / "SECTION_4_BAYESIAN_OPTIMIZATION.md").read_text(encoding="utf-8")
    sec5 = (REPORTS_DIR / "SECTION_5_HMM_AND_VITERBI.md").read_text(encoding="utf-8")
    sec6 = (REPORTS_DIR / "SECTION_6_DISTRIBUTED_DEPLOYMENT.md").read_text(encoding="utf-8")

    sec7 = """---

# Section 7: Conclusion & Deliverables Summary

## 7.1 Key Technical Findings & Lessons Learned
1. **Kinematic-Topological Synergy:** Point-wise classifiers alone, regardless of model complexity, peak around 81% accuracy due to spatial ambiguity on parallel roads. Enforcing graph topology and shortest-path transition penalties via HMM Viterbi decoding is essential to bridge the gap from point classification to genuine vehicle route reconstruction, boosting Top-1 accuracy to **84.70%** and route continuity to **96.98%**.
2. **Local Equirectangular Metric Projection:** Applying a planar projection centered per trip avoids angular scale distortion (33.7% distortion in degrees) while maintaining sub-millisecond execution speeds and achieving <1.14% distortion against spherical Haversine distance.
3. **Robustness to Real-World Sensor Anomalies:** Transparently retaining sensor drift outlier trips (such as Trip `00000054` and Trip `00000033`) while utilizing confidence filtering masks prevents training corruption without distorting real-world evaluation benchmarks.
4. **Bayesian Optimization Dynamics:** Constraining tree depth (`max_leaf_nodes=15`) while increasing regularization prevented over-specialization on specific street geometries, yielding a +3.06% test F1 gain.

## 7.2 Complete Deliverables Checklist

| Deliverable Category | Specific Artifact | Status & Location |
| :--- | :--- | :---: |
| **EDA & Preprocessing** | `reports/SECTION_2_EDA_AND_PREPROCESSING.md` | ✅ Complete |
| **Interactive Map** | `reports/development_trip_00000001.html` (3.2 MB) | ✅ Complete |
| **Publication Figures** | 14 figures in `reports/figures/` (PNG, 300 DPI) | ✅ Complete |
| **5 Model Benchmarking**| `results/run_01/{model}/` (metrics, summary, joblib) | ✅ Complete |
| **Jupyter Notebooks** | 6 executed notebooks in `notebooks/` | ✅ Complete |
| **Bayesian HPO** | `results/hpo/optuna_trials_record.json`, tuned model | ✅ Complete |
| **HMM Viterbi Decoder** | `scripts/hmm_decoder.py`, `results/hmm_evaluation/` | ✅ Complete |
| **FastAPI Backend** | `api/main.py` (Laptop 1 server, port 8000) | ✅ Complete |
| **Evaluator Web UI** | `ui/index.html` (Leaflet) & `ui/app.py` (Streamlit) | ✅ Complete |
| **Unit Test Suite** | `tests/test_preprocessing.py` (All 7 unit tests pass)| ✅ Complete |
"""

    full_report = f"{sec1}\n\n{sec2}\n\n{sec3}\n\n{sec4}\n\n{sec5}\n\n{sec6}\n\n{sec7}"
    out_file = REPORTS_DIR / "FINAL_TECHNICAL_REPORT.md"
    out_file.write_text(full_report, encoding="utf-8")
    print(f"Successfully compiled FINAL_TECHNICAL_REPORT.md ({len(full_report):,} bytes).")

if __name__ == "__main__":
    main()
