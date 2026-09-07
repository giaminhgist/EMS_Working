# 01 — Classical ML Baselines

Eight classical methods are evaluated on the hand-crafted feature set. They
mirror the traditional baselines of the EMS benchmark (EDB_SVM/EDB_QDA/EDB_BYS
from Zhang et al. 2022 and ESR_SVM/ESR_RF from Huang et al. 2020) plus standard
additions (linear SVM, L1 logistic regression, KNN).

## Input representation

All methods below use the **`agg`** subject-level representation unless noted:

- `agg` — for each of the 45 per-stimulus features, the **mean** and **std**
  across the 100 stimuli → **90** dims, plus `n_valid_stim` → **91** dims.
- `lr_l1` uses **`concat`** — the raw concatenation of all 100 per-stimulus
  vectors → **4,500** dims (NaN → 0), mirroring the EDB-style
  "statistics-per-stimulus concatenated" subject features; L1 regularization
  performs implicit stimulus/feature selection.

## Shared pipeline

Every sklearn model uses the same preprocessing (fitted on training subjects
only):

```
X (n_subj, D)  →  SimpleImputer(median)  →  StandardScaler()  →  estimator
                                  ↓                        ↓
                missing per-feature values        zero mean / unit variance
                filled by train-set medians       per feature
```

Final decision: `pred = prob >= 0.5` (SZ positive).

## Methods

### 1. `svm_rbf` — SVM with RBF kernel

| Property | Value |
|---|---|
| Estimator | `sklearn.svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)` |
| Parameters | C=1, gamma=1/(D·Var(X)) |
| Decision | Platt-scaled probability, threshold 0.5 |
| Dims | 91 → 1 |

### 2. `svm_linear` — SVM with linear kernel
Same as `svm_rbf` with `kernel='linear'`.

### 3. `rf` — Random Forest

| Property | Value |
|---|---|
| Estimator | `RandomForestClassifier(n_estimators=500)` |
| Decision | mean class probability over 500 trees, threshold 0.5 |
| Dims | 91 → 1 |

### 4. `qda` — Quadratic Discriminant Analysis

| Property | Value |
|---|---|
| Pipeline | impute → scale → `PCA(20)` → `QuadraticDiscriminantAnalysis(reg_param=0.5)` |
| Decision | posterior P(SZ\|x), threshold 0.5 |
| Note | QDA needs more samples than features per class (~60 train samples/class); PCA to 20 components + shrinkage (reg_param=0.5) makes the covariance estimate well-posed |
| Dims | 91 → 20 → 2 |

### 5. `gnb` — Gaussian Naive Bayes

| Property | Value |
|---|---|
| Estimator | `GaussianNB()` |
| Decision | posterior P(SZ\|x), threshold 0.5 |
| Note | assumes feature independence (wrong, but the paper's EDB_BYS analog) |
| Dims | 91 → 2 |

### 6. `lr` — Logistic Regression (L2)

| Property | Value |
|---|---|
| Estimator | `LogisticRegression(C=1.0, penalty='l2', max_iter=2000)` |
| Decision | sigmoid score, threshold 0.5 |
| Dims | 91 → 1 |

### 7. `lr_l1` — Logistic Regression (L1) on concatenated features

| Property | Value |
|---|---|
| Estimator | `LogisticRegression(C=1.0, penalty='l1', solver='liblinear')` |
| Input | `concat` 4,500-dim per-stimulus concatenation (NaN→0) |
| Decision | sigmoid score, threshold 0.5 |
| Note | L1 penalty gives implicit per-stimulus feature selection on the EDB-style representation |
| Dims | 4,500 → 1 |

### 8. `knn` — k-Nearest Neighbors

| Property | Value |
|---|---|
| Estimator | `KNeighborsClassifier(n_neighbors=5)` |
| Decision | fraction of SZ among the 5 nearest (standardized) training subjects, threshold 0.5 |
| Dims | 91 → 1 |

## Complexity

| Method | Training cost (160 subjects) | Inference cost |
|---|---|---|
| svm_rbf | O(n²D) ~ 1 s | ~1 ms |
| svm_linear / lr / lr_l1 | O(nD²) ~ 1 s | ~1 ms |
| rf | 500 trees ~ 2 s | ~10 ms |
| qda | O(nD²) ~ 0.5 s | ~1 ms |
| gnb | O(nD) | ~1 ms |
| knn | lazy, O(1) | O(nD) per query |
