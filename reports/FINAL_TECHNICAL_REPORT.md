# PathSnap: Distributed Map-Matching & Route Reconstruction
## Complete Technical Report & Evaluation Dossier

**UE24CS352A: Machine Learning — Department of Computer Science & Engineering, PES University**  
**Team 6:** Yaswanth, Madhu, Hrisheek, Joshwin  
**Winning System:** Tuned HistGradientBoostingClassifier + Graph-Constrained First-Order Hidden Markov Model (Viterbi Trellis Decoding)  
**Final Test Performance:** Top-1 Accuracy: **84.70%** (Gain: +3.51%) | Route Continuity: **96.98%** (Gain: +7.76%)  
**Distributed Architecture:** Two-Laptop LAN Client-Server (FastAPI Backend on Laptop 1 + Leaflet/Streamlit Evaluator UI on Laptop 2)

---

# Section 1: Executive Summary & Problem Formulation

## 1.1 Problem Context & Motivation
Modern intelligent transportation systems (ITS), navigation platforms, and autonomous fleets rely heavily on global positioning system (GPS) feeds. However, consumer GPS sensors in urban environments suffer from severe multipath reflections, satellite occlusion, and sensor jitter. Point-wise matching of GPS fixes to road network graphs frequently leads to erratic parallel road oscillation, topological disconnections, and ghost turns at junctions.

The **PathSnap** project requires designing, training, optimizing, and deploying an end-to-end distributed system that:
1. Conducts structured, mathematically rigorous Exploratory Data Analysis (EDA) on trajectory and road network datasets.
2. Formulates feature extraction that maps noisy GPS observations to $K=8$ candidate road segments.
3. Evaluates and benchmarks **5 distinct machine learning model families** using strictly group-aware 5-fold cross-validation.
4. Tunes the winning model using **Bayesian Hyperparameter Optimization** (Optuna).
5. Implements a **First-Order Hidden Markov Model (HMM)** with dynamic programming (**Viterbi Algorithm**) to enforce physical vehicle kinematics and road network graph topology.
6. Deploys the complete system across a **two-laptop distributed architecture** featuring a FastAPI backend on Laptop 1 and an interactive Evaluator UI on Laptop 2, guaranteeing zero data leakage of reserved evaluation ground truth.

## 1.2 End-to-End System Architecture

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

---

# Section 2: Dataset Understanding and Exploratory Data Analysis

## 2.1 Question 1: How Did You Preprocess the Data?

The raw trajectory and road network data were preprocessed using a mathematically grounded, reproducible pipeline implemented in `scripts/preprocess.py` and validated in `scripts/validate_dataset.py`.

### 1. Geographical Projection & Scale Invariance
Longitude and latitude are angular coordinates. At latitude $\phi \approx 48.5^\circ\text{N}$, $1^\circ$ of longitude corresponds to $\approx 73.68\text{ km}$, while $1^\circ$ of latitude corresponds to $\approx 111.19\text{ km}$ ($33.7\%$ distortion under Euclidean degree distance).  
To preserve computational efficiency without external GIS dependencies, each trip is projected into a local **equirectangular planar metric coordinate system** centred at the median road-node coordinates $(\lambda_0, \phi_0)$ of that trip:
$$x = R \cdot (\lambda - \lambda_0) \cdot \frac{\pi}{180} \cdot \cos\left(\phi_0 \cdot \frac{\pi}{180}\right) \quad [\text{metres}]$$
$$y = R \cdot (\phi - \phi_0) \cdot \frac{\pi}{180} \quad [\text{metres}]$$
where $R = 6,371,008.8\text{ m}$. In `scripts/validate_dataset.py`, we benchmarked the projected planar segment lengths against spherical geodesic Haversine distances across all road networks. The maximum distortion observed across the entire dataset was **$1.14\%$** for segments $>1\text{ m}$, confirming that local projection error is negligible compared to consumer GPS noise ($5\text{--}15\text{ m}$).

### 2. Candidate Road Segment Generation ($K=8$)
For each GPS point, we generate a fixed set of $K=8$ candidate road segments using exact point-to-segment orthogonal projection:
* A spatial `cKDTree` index is built on the midpoints of all directed road segments.
* To prevent missing long segments whose midpoints might lie outside a narrow query radius, the search sphere dynamically expands by the maximum segment half-length: $r_{\text{search}} = d_{\text{seed}} + \max(\text{segment\_half\_length})$.
* Exact point-to-segment orthogonal distances and projection fractions $[0, 1]$ are computed, with ties broken deterministically by `arc_id`.

### 3. Reference Traversal Alignment & Proxy Label Derivation
The supplied `.route` file represents a traversal sequence through the network graph rather than a 1:1 GPS observation label. We align GPS points to the route sequence using a **monotonic dynamic programming formulation**:
$$\min_{\pi} \sum_{t=1}^{T} d(p_t, \text{segment}(\pi_t))^2 \quad \text{subject to } \pi_1 \le \pi_2 \le \dots \le \pi_T$$
Derived targets are termed **proxy labels**. A high-confidence fitting mask selects observations with residual $\le 50\text{ m}$ and margin $\ge 3\text{ m}$ ($136,582$ rows, $65.53\%$) for training. Crucially, **all observations remain in student-test and evaluator-test benchmark evaluation sets**.

---

## 2.2 Question 2: Patterns Observed & Feature Extraction Rationale

### Observed Patterns during EDA:
1. **Nominally 1 Hz Sampling with Outlier Gaps:** Exactly **$97.2\%$** of observations have time intervals $\Delta t = 1.0\text{ s}$. However, occasional multi-second dropouts exist (e.g. Trip `00000021` features a gap of 2,161s), where linear interpolation would fabricate non-existent vehicle motion.
2. **Urban Noise vs Highway Smoothness:** GPS noise standard deviation is approximately $5\text{--}15\text{ m}$ in street grids, causing raw points to oscillate across parallel lanes.
3. **Directed Graph Disconnectedness:** Road networks are directed graphs with multiple weakly connected components (median 20 components in train, 11 in test).
4. **Sensor Drift Outliers:** Two specific training trips account for 89.8% of candidate misses: Trip `00000054` (13,162 misses, 95th pct error 209 m) and Trip `00000033` (3,867 misses, 95th pct error 142 m). These are transparently retained.

### 10-Dimensional Feature Extraction Rationale:

| Index | Feature Name | Description | Technical Rationale |
| :---: | :--- | :--- | :--- |
| **0** | `distance_m` | Orthogonal distance from GPS fix to candidate segment. | Primary emission feature; probability decays with orthogonal distance. |
| **1** | `heading_cos` | Cosine similarity between GPS velocity and segment vector. | Penalizes road segments travelling opposite to vehicle direction (one-ways, dual carriageways). |
| **2** | `heading_available` | Binary indicator (1 if $\Delta t \in (0, 30\text{s}]$ and step $\ge 2\text{m}$, else 0). | Distinguishes stopped/static vehicles from perpendicular motion. |
| **3** | `segment_length_m` | Physical length of candidate road segment. | Controls for candidate capture bias on long highways vs dense urban links. |
| **4** | `projection_fraction` | Normalized position $[0, 1]$ along candidate segment. | Identifies whether projection falls near midpoints or at junction turns. |
| **5** | `speed_mps` | Instantaneous GPS speed ($\Delta \text{distance} / \Delta t$). | Captures vehicle dynamics and stop-and-go behavior. |
| **6** | `previous_gap_s` | Time elapsed since preceding observation ($\Delta t$). | Informs tree when GPS temporal continuity has degraded. |
| **7** | `gps_step_m` | Euclidean distance travelled since preceding GPS fix. | Measures raw displacement between consecutive measurements. |
| **8** | `candidate_distance_rank`| Integer rank $[0, 7]$ of candidate ordered by distance. | Relative ordering feature; nearest candidate is rank 0. |
| **9** | `distance_minus_nearest_m`| Candidate distance minus distance to rank-0 candidate. | Margin metric highlighting ambiguous parallel streets (margin $\approx 0$). |

---

## 2.3 EDA Checklist & Partition Inventory

| Exploration Category | Training Split (80 Trips) | Student-Test Split (10 Trips) | Evaluator-Test Split (10 Trips) |
| :--- | :---: | :---: | :---: |
| **Trip Count** | 80 | 10 | 10 |
| **Total GPS Observations** | 208,417 | 19,160 | 19,674 |
| **Observations per Trip (Min / Med / Max)** | 361 / 1,529 / 19,313 | 262 / 1,420 / 3,958 | 458 / 1,245 / 6,854 |
| **Missing / Invalid Values (NaN/Inf)** | 0 | 0 | 0 |
| **Coordinate Order** | `(longitude, latitude)` | `(longitude, latitude)` | `(longitude, latitude)` |
| **Trips with Gaps $>30\text{s}$** | 28 trips (133 events) | 2 trips (2 events) | 3 trips (4 events) |
| **Road Nodes (Total / Median per trip)** | 1,933,485 / 9,608 | 90,689 / 6,596 | 94,120 / 7,120 |
| **Road Arcs (Total / Median per trip)** | 3,876,904 / 19,526 | 185,041 / 13,145 | 192,450 / 14,200 |
| **Weakly Connected Components (Median)** | 20.0 | 11.0 | 12.0 |
| **Isolated Nodes** | 0 | 0 | 0 |
| **Zero-Length Arcs (`from == to`)** | 100 (in 29 trips) | 6 (in 3 trips) | 8 (in 3 trips) |

---

# Section 3: Model Experimentation and Comparison

## 3.1 Overall Model Comparison

All models were evaluated using group-aware 5-fold cross-validation on 80 training trips, refitted on all 80 training trips ($1,092,656$ rows), and evaluated on the held-out student-testing partition ($19,160$ observations, $153,280$ candidate rows).

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear / Probabilistic | `StandardScaler()`, $C=1.0$, `l2`, `lbfgs` | $0.6428 \pm 0.049$ | 80.76% | N/A* | 0.6284 | 0.7729 | **10.0 ms** |
| **HistGradientBoosting**| Gradient Boosting | `max_iter=150`, $\eta=0.1$, `max_leaf=31`, $\lambda=1.0$ | $0.6471 \pm 0.044$ | 80.78% | N/A* | 0.6113 | 0.6710 | 375.6 ms |
| **MLP (Neural Net)** | Neural Networks | `StandardScaler()`, `(64, 32)`, `batch=2048`, `adam` | $0.6506 \pm 0.051$ | 80.50% | N/A* | 0.6271 | 0.6486 | 230.2 ms |
| **Extra Trees** | Randomized Ensembles | `n_estimators=100`, `max_depth=20`, `min_leaf=2` | $\mathbf{0.6574 \pm 0.044}$ | $\mathbf{81.15\%}$ | N/A* | $\mathbf{0.6332}$ | $\mathbf{0.6329}$ | 579.1 ms |
| **Random Forest** | Bagging Ensembles | `n_estimators=100`, `max_depth=20`, `min_leaf=2` | $0.6513 \pm 0.046$ | 81.06% | N/A* | 0.6215 | 0.7401 | 396.1 ms |

*\*Note on Route Continuity:* Classifier-only candidate scoring evaluates observations independently without sequential graph transition constraints; route continuity is formally measured in the HMM / Viterbi decoding phase (Section 5).

---

## 3.2 Compulsory Questions for EVERY Model

### Model 1: Logistic Regression
* **A. Model Understanding:**
  1. *Suitability:* Fast linear baseline establishing the floor of linear separability for monotonic distance and heading features.
  2. *Strengths & Limitations:* Extremely fast (10 ms); convex loss guarantees global convergence. However, incapable of capturing non-linear interactions (e.g. speed-dependent heading reliability).
* **B. Training & Configuration:**
  3. *Hyperparameters:* $C=1.0, \text{penalty}=\text{'l2'}, \text{solver}=\text{'lbfgs'}, \text{max\_iter}=1000$.
  4. *Loss Function:* Binary cross-entropy with $L_2$ weight regularization.
* **C. Experimental Results:**
  5. *Student-Test Performance:* Top-1 Acc: **80.76%**, Macro F1: **0.6284**, Log Loss: **0.7729**.
  6. *Overfitting vs Underfitting:* Mild underfitting. Linear capacity bounds accuracy despite minimal generalization gap.

### Model 2: Histogram-Based Gradient Boosting (HistGradientBoosting)
* **A. Model Understanding:**
  1. *Suitability:* Fast histogram binning (256 bins) scales to millions of continuous spatial candidate rows while sequentially correcting residual classification errors.
  2. *Strengths & Limitations:* Naturally captures non-linear conditional thresholds; outputs calibrated probabilities. Sequential tree growth is less parallelizable than random forests.
* **B. Training & Configuration:**
  3. *Hyperparameters:* `max_iter=150`, `learning_rate=0.1`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=20`.
  4. *Loss Function:* Negative log-likelihood (binary cross-entropy).
* **C. Experimental Results:**
  5. *Student-Test Performance:* Top-1 Acc: **80.78%**, Macro F1: **0.6113**, Log Loss: **0.6710**.
  6. *Overfitting vs Underfitting:* Balanced. Regularized leaves prevent memorizing noisy GPS multipath shifts.

### Model 3: Multi-Layer Perceptron (MLP Neural Net)
* **A. Model Understanding:**
  1. *Suitability:* Deep continuous functional approximation with non-linear ReLU activations.
  2. *Strengths & Limitations:* Capable of universal functional approximation. Highly sensitive to feature scaling; slower training convergence on tabular spatial data.
* **B. Training & Configuration:**
  3. *Hyperparameters:* Hidden layers `(64, 32)`, ReLU activation, Adam optimizer, `batch_size=2048`, `max_iter=50`, early stopping patience 10.
  4. *Loss Function:* Binary cross-entropy optimized via backpropagation.
* **C. Experimental Results:**
  5. *Student-Test Performance:* Top-1 Acc: **80.50%**, Macro F1: **0.6271**, Log Loss: **0.6486**.
  6. *Overfitting vs Underfitting:* Balanced. Validation loss plateaued smoothly around epoch 35 without diverging.

### Model 4: Extra Trees (Extremely Randomized Trees)
* **A. Model Understanding:**
  1. *Suitability:* Extreme randomization of cut-point thresholds reduces ensemble variance across noisy GPS features.
  2. *Strengths & Limitations:* High raw accuracy; robust to noisy spatial outliers. Memory intensive ($>100\text{ MB}$ weights); highest inference latency ($579.1\text{ ms}$).
* **B. Training & Configuration:**
  3. *Hyperparameters:* `n_estimators=100`, `max_depth=20`, `min_samples_leaf=2`, `max_features='sqrt'`.
  4. *Loss Function:* Gini impurity minimization across random split thresholds.
* **C. Experimental Results:**
  5. *Student-Test Performance:* Top-1 Acc: **81.15%**, Macro F1: **0.6332**, Log Loss: **0.6329**.
  6. *Overfitting vs Underfitting:* Slight overfitting on fitting rows, controlled by `min_samples_leaf=2` and `max_depth=20`.

### Model 5: Random Forest
* **A. Model Understanding:**
  1. *Suitability:* Bootstrap aggregating of unpruned decision trees provides robust variance reduction.
  2. *Strengths & Limitations:* Highly stable against localized measurement noise. Large memory footprint ($156\text{ MB}$); uncalibrated probability outputs.
* **B. Training & Configuration:**
  3. *Hyperparameters:* `n_estimators=100`, `max_depth=20`, `min_samples_leaf=2`.
  4. *Loss Function:* Gini impurity across optimal feature splits.
* **C. Experimental Results:**
  5. *Student-Test Performance:* Top-1 Acc: **81.06%**, Macro F1: **0.6215**, Log Loss: **0.7401**.
  6. *Overfitting vs Underfitting:* Controlled by depth limit; test performance tracks validation folds closely.

---

# Section 4: Bayesian Hyperparameter Optimisation (HistGradientBoosting)

Per project guidelines, **HistGradientBoostingClassifier** was selected for Bayesian Hyperparameter Optimization. Using Optuna's Tree-structured Parzen Estimator (TPE) across 20 trials with 5-fold group-aware CV:

1. **Hyperparameters Included:** `learning_rate` (log-uniform $[0.01, 0.2]$), `max_iter` (int $[80, 200]$), `max_leaf_nodes` (int $[15, 63]$), `min_samples_leaf` (int $[10, 80]$), `l2_regularization` (log-uniform $[10^{-3}, 10.0]$).
2. **Objective Function:** Mean 5-fold cross-validation Macro F1-score across 80 training trips.
3. **Number of Trials:** 20 complete trials (recorded in `results/hpo/optuna_trials_record.json`).
4. **Best Values Obtained (Trial 15):** `learning_rate=0.05628, max_iter=140, max_leaf_nodes=15, min_samples_leaf=48, l2_regularization=0.07677`.
5. **Did Tuned Model Outperform Untuned?** Yes, across all evaluation metrics.

### Untuned vs. Tuned Comparison Table

| Metric | Untuned HistGradientBoosting | Tuned HistGradientBoosting (Trial 15) | Relative Gain |
| :--- | :---: | :---: | :---: |
| **5-Fold CV Macro F1** | 0.6471 | **0.6610** | **+2.15%** |
| **Student-Test Top-1 Accuracy** | 80.78% | **81.18%** | **+0.40%** |
| **Student-Test Macro F1** | 0.6113 | **0.6300** | **+3.06%** |
| **Categorical Log Loss** | 0.6710 | **0.6621** | **Improved Calibration** |
| **Inference Time (19,160 obs)** | 375.6 ms | **354.2 ms** | **+5.7% Faster** |

---

# Section 5: HMM and Viterbi Route Reconstruction

## 5.1 Implementation Details
1. **Hidden States:** The $K=8$ candidate road segments at each observation $t$.
2. **Emission Probabilities:** Normalized posterior probabilities from tuned HistGradientBoosting:
   $$\ln b_k(y_t) = \ln \tilde{p}_{t, k}, \quad \tilde{p}_{t, k} = \frac{\max(\hat{P}_{t, k}, 10^{-12})}{\sum_{j=1}^{8} \max(\hat{P}_{t, j}, 10^{-12})}$$
3. **Transition Probabilities:** Shortest-path distance $d_{\text{net}}(u, v)$ along directed road network graph $G$:
   $$\ln a_{u, v} = \begin{cases}
   0.0 & \text{if } \Delta t > 30\text{ s (Sensor dropout reset)} \\
   \beta_{\text{direct}} - \frac{d_{\text{gps}}}{\beta} & \text{if } u = v \lor \text{to}(u) == \text{from}(v) \text{ (Adjacency)} \\
   -\frac{|d_{\text{net}}(u, v) - d_{\text{gps}}|}{\beta} & \text{if } d_{\text{net}}(u, v) \le 250\text{ m} \\
   \text{Penalty}_{\text{disconnect}} & \text{otherwise}
   \end{cases}$$
   Parameters: $\beta = 15.0\text{ m}, \beta_{\text{direct}} = +2.0, \text{Penalty}_{\text{disconnect}} = -12.0$.
4. **Handling Missing Candidates / Large GPS Gaps:** When $\Delta t > 30\text{s}$, transition constraints are relaxed ($\ln a_{u, v} = 0$), allowing the Viterbi trellis to re-anchor based purely on emission confidence.
5. **Handling Disconnected Road Segments:** If no directed path exists in $G$, $d_{\text{net}}(u, v) = \infty$, incurring a disconnection penalty of $-12.0$.

## 5.2 Experimental Results: Classifier-Only vs. Classifier + HMM Viterbi

### Student-Test Partition (10 Trips, 19,160 Observations)
* **Top-1 Accuracy:** Classifier-Only: **81.18%** $\to$ HMM+Viterbi: **84.70%** (**+3.51% absolute gain**).
* **Mean Route Continuity:** Classifier-Only: **89.23%** $\to$ HMM+Viterbi: **96.98%** (**+7.76% absolute gain**).
* **Universal Improvement:** All 10 student-test trips improved in both Top-1 accuracy and route continuity.

### Evaluator-Test Partition (10 Trips, 19,674 Observations)
* **Top-1 Accuracy:** Classifier-Only: **74.13%** $\to$ HMM+Viterbi: **76.71%** (**+2.57% absolute gain**).
* **Mean Route Continuity:** Classifier-Only: **89.04%** $\to$ HMM+Viterbi: **95.62%** (**+6.58% absolute gain**).

---

# Section 6: Geographical Visualisation Using Folium

In strict compliance with Page 6 of the guidelines, interactive geographical maps were generated using Folium featuring all four mandatory layers:
1. **Original GPS Observations:** Blue circle markers (`radius=2.5`) with hover tooltips displaying observation index and timestamp.
2. **Relevant Road-Network Geometry:** Sub-graph arcs within the trajectory bounding box rendered as semitransparent slate lines (`weight=1.5, opacity=0.45`).
3. **Reference Route (Ground Truth):** Bold emerald green polyline (`weight=6, opacity=0.8`) displaying the true traversed vehicle route.
4. **Predicted Route (HMM / Viterbi Decoded):** High-contrast orange dashed polyline (`weight=3.5, opacity=0.9`) displaying the continuous reconstructed route.

### Map Design Features:
* **Distinguishable Layers & Colors:** LayerControl toggles each layer independently.
* **Clear Fixed Legend:** Screen-pinned floating legend displaying color swatches, layer descriptions, and Trip accuracy.
* **Interactive Artifact:** Saved at `reports/folium_trip_00000049_four_layers.html` and `reports/development_trip_00000001.html`.

---

# Section 7: API, UI and Deployment

## 7.1 Distributed Architecture: Two-Laptop Setup
* **Laptop 1 (Inference Server):**
  * Runs FastAPI backend (`api/main.py`) listening on `0.0.0.0:8000`.
  * Houses trained model weights (`model.joblib`), HMM sequential decoder, and strictly guarded server-side ground-truth route data.
  * Emits live standardized request logs directly to the command line.
* **Laptop 2 (Evaluator Client Interface):**
  * Runs zero-install web interface (`ui/index.html`) using Esri World Street Map tiles.
  * Connects to Laptop 1 IP (`http://<laptop1_ip>:8000`), presents trip dropdown selector, and renders live scorecards and route trajectories.

## 7.2 Execution Instructions
* **Laptop 1 (Start Server):**
  ```powershell
  cd "team_6"
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
  ```
* **Laptop 2 (Launch UI):**
  Open `ui/index.html` in Chrome or Edge, enter Laptop 1's IP address (e.g. `http://10.0.15.194:8000`), select any trip (`00000015` to `00000094`), and click **"Run Map-Matching & Score"**.

---

# Section 8: Academic Integrity & Deliverables Verification

All code, models, and results presented in this report represent authentic, independently executed work by Team 6. No external code copying or data leakage occurred. All 10 evaluator trips remain server-side and were evaluated live.
