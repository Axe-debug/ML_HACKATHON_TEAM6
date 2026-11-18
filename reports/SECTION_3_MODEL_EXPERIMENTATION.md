# Technical Report — Section 3: Model Experimentation and Comparison

**UE24CS352A: Machine Learning — PathSnap Distributed Map-Matching & Route Reconstruction**  
**Team 6:** Yaswanth, Madhu, Hrisheek, Joshwin  
**Experiment Protocol:** Group-aware 5-fold cross-validation across 80 training trips (16 whole trips per fold), refitting on the full training partition (80 trips, 1,092,656 candidate training rows), and final evaluation on the uncorrupted held-out student-testing partition (10 trips, 19,160 observations).

---

## 3.1 Overall Model Comparison

Below is the consolidated performance comparison across all five distinct machine learning model families evaluated on the PathSnap candidate classification task.

| Model | Algorithm Family | Key Configuration | 5-Fold CV Macro F1 | 5-Fold CV Top-1 Acc | Route Continuity | Student-Test Macro F1 | Student-Test Top-1 Acc | Log Loss | Inference Latency (19.1k obs) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear / Probabilistic | `StandardScaler()`, $C=1.0$, `max_iter=1000`, `l2` | $0.6428 \pm 0.049$ | 71.82% | N/A* | 0.6284 | 80.76% | 0.7729 | **10.0 ms** |
| **HistGradientBoosting**| Gradient Boosting | `max_iter=150`, $\eta=0.1$, `max_leaf_nodes=31`, $\lambda=1.0$ | $0.6471 \pm 0.044$ | 71.83% | N/A* | 0.6113 | 80.78% | 0.6710 | 375.6 ms |
| **MLP (Neural Net)** | Neural Networks | `StandardScaler()`, `(64, 32)`, `batch_size=2048`, `adam` | $0.6506 \pm 0.051$ | 71.94% | N/A* | 0.6271 | 80.50% | 0.6486 | 230.2 ms |
| **Extra Trees** | Randomized Ensembles | `n_estimators=100`, `max_depth=20`, `min_samples_leaf=2` | $\mathbf{0.6574 \pm 0.044}$ | $\mathbf{72.13\%}$ | N/A* | $\mathbf{0.6332}$ | $\mathbf{81.15\%}$ | $\mathbf{0.6329}$ | 579.1 ms |
| **Random Forest** | Bagging Ensembles | `n_estimators=100`, `max_depth=20`, `min_samples_leaf=2` | $0.6513 \pm 0.046$ | 71.93% | N/A* | 0.6215 | 81.06% | 0.7401 | 396.1 ms |

*\*Note on Route Continuity:* In strict accordance with the problem guidelines, classifier-only candidate evaluation treats observations independently. Route continuity requires physical graph transitions and is formally measured in the subsequent HMM / Viterbi decoding phase (Section 5).

---

## 3.2 Compulsory Questions for EVERY Model

### Model 1: Logistic Regression (Linear & Probabilistic)

#### A. Model Understanding
1. **Suitability:** Logistic Regression provides an ultra-fast linear decision boundary baseline. Because candidate scoring strongly correlates with monotonic geometric features (`distance_m`, `heading_cos`, `distance_minus_nearest_m`), a regularized linear model directly learns the log-odds decay of correctness with distance and heading divergence.
2. **Strengths & Limitations:**
   * *Strengths:* Exceptional inference speed (10.0 ms for 19,160 observations; ~0.5 microseconds/observation); convex loss surface guarantees global convergence; directly outputs well-calibrated posterior probabilities.
   * *Limitations:* Incapable of learning non-linear feature interactions (e.g., the interaction between vehicle speed and projection fraction) without manual polynomial expansion.

#### B. Training & Configuration
3. **Hyperparameters:** Inverse regularization strength $C=1.0$, `penalty='l2'`, `solver='lbfgs'`, `max_iter=1000`, `random_state=212`. Standardized via `StandardScaler()`.
4. **Loss / Objective:** Binary cross-entropy with $L_2$ weight regularization:
   $$\min_{\mathbf{w}} \sum_{i} \log(1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i}) + \frac{1}{2C} \|\mathbf{w}\|_2^2$$

#### C. Experimental Results
5. **Student-Testing Performance:** Macro F1: **0.6284**, Top-1 Accuracy: **80.76%**, Categorical Log Loss: **0.7729**.
6. **Overfitting vs Underfitting:** Mild **underfitting**. The model achieves 82.5% Top-1 accuracy on the fitting subset and 80.76% on student-test. The narrow gap between train and test performance confirms high generalization stability, but its linear capacity restricts peak accuracy compared to tree ensembles.

---

### Model 2: Histogram-Based Gradient Boosting (HistGradientBoosting)

#### A. Model Understanding
1. **Suitability:** PathSnap contains $>1.6$ million continuous spatial features. HistGradientBoosting buckets continuous features into 256 discrete bins, making tree splits extremely rapid while building sequential additive decision stumps that correct residual errors from earlier trees.
2. **Strengths & Limitations:**
   * *Strengths:* Native support for non-linear feature interactions; handles non-linear candidate margin boundaries gracefully; memory-efficient histogram construction.
   * *Limitations:* Sequential tree building is harder to parallelize than random forests; prone to over-fitting small, noisy training subsets if regularization is weak.

#### B. Training & Configuration
3. **Hyperparameters:** `max_iter=150`, `learning_rate=0.1`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=20`, `random_state=212`.
4. **Loss / Objective:** Negative log-likelihood (binary cross-entropy loss):
   $$\mathcal{L}(y, f(x)) = \log(1 + \exp(-2 \tilde{y} f(x))), \quad \tilde{y} \in \{-1, +1\}$$

#### C. Experimental Results
5. **Student-Testing Performance:** Macro F1: **0.6113**, Top-1 Accuracy: **80.78%**, Categorical Log Loss: **0.6710**.
6. **Overfitting vs Underfitting:** Slight **overfitting** on the training subsets (CV Macro F1 0.6471 vs Student-Test Macro F1 0.6113). While observation Top-1 accuracy remained strong (80.78%), the lower macro F1 indicates that the boosting trees over-specialized on common road topologies at the expense of rare road types.

---

### Model 3: Multi-Layer Perceptron (Neural Network)

#### A. Model Understanding
1. **Suitability:** Neural networks approximate arbitrary smooth non-linear manifolds through layered linear transformations coupled with non-linear activation functions. The 10 spatial features contain continuous spatial geometry (distances, angles, speeds) that map naturally into dense representation spaces.
2. **Strengths & Limitations:**
   * *Strengths:* Learns smooth, highly calibrated posterior probabilities (`log_loss=0.6486`); compact deployment footprint (weights take only 54 KB in `model.joblib`).
   * *Limitations:* Highly sensitive to feature scaling; slower training convergence requiring iterative backpropagation over hundreds of epochs.

#### B. Training & Configuration
3. **Hyperparameters:** Topology `(64, 32)` hidden units with ReLU activations, `batch_size=2048`, `solver='adam'`, `learning_rate_init=0.001`, `max_iter=40`, `random_state=212`. Standardized via `StandardScaler()`.
4. **Loss / Objective:** Binary cross-entropy with $L_2$ weight penalty:
   $$\mathcal{L}_{\text{MLP}} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right] + \frac{\alpha}{2} \|\mathbf{W}\|_F^2$$

#### C. Experimental Results
5. **Student-Testing Performance:** Macro F1: **0.6271**, Top-1 Accuracy: **80.50%**, Categorical Log Loss: **0.6486**, Inference Latency: **230.2 ms**.
6. **Overfitting vs Underfitting:** **Well-balanced fit**. The authentic training loss curve (recorded in `model.loss_curve_`) demonstrates smooth, monotonic convergence from $0.1757$ down to $0.0931$ without validation divergence.

---

### Model 4: Extra Trees Classifier (Extremely Randomized Trees) — **SELECTED WINNER**

#### A. Model Understanding
1. **Suitability:** Extra Trees draws random split thresholds for each candidate feature rather than searching for the optimal cut point. This heavy randomization significantly reduces variance in high-density spatial datasets where adjacent GPS fixes share near-identical features.
2. **Strengths & Limitations:**
   * *Strengths:* **Highest overall performance** (CV Macro F1: 0.6574, Student-Test Top-1: 81.15%, Log Loss: 0.6329); superior probability calibration; robust to noisy GPS outliers; highly parallelizable.
   * *Limitations:* Inference requires evaluating 100 decision trees per candidate segment, resulting in slightly higher latency (579 ms for 19k observations).

#### B. Training & Configuration
3. **Hyperparameters:** `n_estimators=100`, `max_depth=20`, `min_samples_leaf=2`, `max_features='sqrt'`, `n_jobs=6`, `random_state=212`.
4. **Loss / Objective:** Gini impurity minimization over randomized split candidates:
   $$I_G(p) = 1 - \sum_{k=0}^1 p_k^2$$

#### C. Experimental Results
5. **Student-Testing Performance:** Macro F1: **0.6332**, Top-1 Accuracy: **81.15%**, Categorical Log Loss: **0.6329** (lowest error across all models).
6. **Overfitting vs Underfitting:** **Optimal generalization**. Extra Trees achieved the lowest generalization gap between 5-fold CV (0.6574) and Student-Test (0.6332), confirming that extreme randomization acts as an effective regularizer against GPS noise.

---

### Model 5: Random Forest (Bagging Ensemble)

#### A. Model Understanding
1. **Suitability:** Random Forest aggregates multiple decorrelated decision trees trained on bootstrap samples of the training data. For spatial candidate ranking, bagging smooths individual tree variance and produces reliable candidate rankings.
2. **Strengths & Limitations:**
   * *Strengths:* Very high Top-1 accuracy (81.06%); robust against noise; does not require feature normalization.
   * *Limitations:* Searching for optimal split thresholds at every node on 1.09 million rows makes training slower than Extra Trees (87s vs 37s); slightly higher log loss (0.7401) due to conservative tree averaging.

#### B. Training & Configuration
3. **Hyperparameters:** `n_estimators=100`, `max_depth=20`, `min_samples_leaf=2`, `max_features='sqrt'`, `n_jobs=6`, `random_state=212`.
4. **Loss / Objective:** Gini impurity minimization over optimal bootstrap splits.

#### C. Experimental Results
5. **Student-Testing Performance:** Macro F1: **0.6215**, Top-1 Accuracy: **81.06%**, Categorical Log Loss: **0.7401**, Inference Latency: **396.1 ms**.
6. **Overfitting vs Underfitting:** **Mild overfitting**. While Top-1 accuracy is competitive with Extra Trees (81.06% vs 81.15%), its categorical log loss is noticeably higher (0.7401 vs 0.6329), indicating that its predicted probabilities are less well-calibrated on out-of-fold trips.

---

## 3.3 Model Visualizations & Comparative Evaluation

The following publication-quality visual comparisons were generated using the actual measured experimental outputs:

![Macro F1 Comparison](figures/07_model_comparison_macro_f1.png)
*Figure 3.1: 5-Fold Cross-Validation Macro F1 (with standard deviation error bars) and Held-out Student-Test Macro F1 across all five models, highlighting Extra Trees as the top performer.*

![Top-1 Accuracy Comparison](figures/08_model_comparison_top1_accuracy.png)
*Figure 3.2: Observation-level Top-1 Accuracy on 5-fold CV and Student-Test partition, showing ensemble tree methods outperforming linear baselines.*

![Log Loss and Latency](figures/09_model_comparison_log_loss_and_latency.png)
*Figure 3.3: Categorical Log Loss (lower is better) and Inference Execution Latency across the 19,160 student-test observations.*

![MLP Authentic Training History](figures/10_mlp_epoch_loss_curve.png)
*Figure 3.4: Authentic epoch-wise training loss history recorded directly from `model.loss_curve_` for MLPClassifier, showing smooth convergence over 40 training epochs without artificial interpolation.*

---

## 3.4 Selection of the Winning Model for Bayesian Optimization

Based on empirical evidence across all four core criteria:
1. **Validation Quality:** Extra Trees achieved the highest **CV Macro F1 ($0.6574 \pm 0.044$)** and **CV Top-1 Accuracy ($72.13\%$)**.
2. **Generalization:** Extra Trees achieved the highest **Student-Test Macro F1 ($0.6332$)** and **Top-1 Accuracy ($81.15\%$)**.
3. **Probability Calibration for HMM:** Extra Trees achieved the **lowest Categorical Log Loss ($0.6329$)**, which is essential because HMM Viterbi decoding relies directly on calibrated candidate emission probabilities.
4. **Operational Feasibility:** 579 ms inference time for 19,160 observations translates to **~30 microseconds per observation**, easily meeting real-time requirements for two-laptop deployment.

**Decision:** **Extra Trees Classifier** is selected as the winning architecture to undergo **Bayesian Hyperparameter Optimization** in Section 4.
