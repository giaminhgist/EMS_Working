"""Model factory: classical ML baselines and the basic FNN.

Classical methods mirror the traditional baselines of the EMS benchmark
(EDB_* / ESR_* in the paper) plus standard additions. All sklearn models
receive standardized features (StandardScaler + median imputation fitted on
the training folds only).

FNN: a basic feed-forward network (details in docs/baseline/02_fnn.md).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import SEED

# --------------------------------------------------------------------------- #
# Classical ML
# --------------------------------------------------------------------------- #

def make_ml_pipeline(method):
    """Return a sklearn Pipeline (impute -> scale -> estimator) per method."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    if method == "svm_rbf":
        est = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
                  random_state=SEED)
    elif method == "svm_linear":
        est = SVC(C=1.0, kernel="linear", probability=True, random_state=SEED)
    elif method == "rf":
        est = RandomForestClassifier(n_estimators=500, random_state=SEED,
                                     n_jobs=-1)
    elif method == "qda":
        # QDA requires n_samples > n_features per class; with 91 features and
        # ~60 train samples per class we first reduce to 20 PCA components.
        from sklearn.decomposition import PCA
        return Pipeline([("imputer", imputer), ("scaler", scaler),
                         ("pca", PCA(n_components=20, random_state=SEED)),
                         ("est", QuadraticDiscriminantAnalysis(reg_param=0.5))])
    elif method == "gnb":
        est = GaussianNB()
    elif method == "lr":
        est = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    elif method == "lr_l1":
        est = LogisticRegression(C=1.0, penalty="l1", solver="liblinear",
                                 max_iter=2000, random_state=SEED)
    elif method == "knn":
        est = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"unknown method {method}")
    return Pipeline([("imputer", imputer), ("scaler", scaler), ("est", est)])


def fit_predict_ml(method, X_train, y_train, X_eval):
    """Fit on train, return probabilities on eval (shape (n_eval,))."""
    pipe = make_ml_pipeline(method)
    pipe.fit(np.asarray(X_train, dtype=np.float64), np.asarray(y_train))
    if hasattr(pipe["est"], "predict_proba"):
        return pipe.predict_proba(np.asarray(X_eval, dtype=np.float64))[:, 1]
    return pipe.decision_function(np.asarray(X_eval, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Basic FNN
# --------------------------------------------------------------------------- #

class BasicFNN(nn.Module):
    """Basic feed-forward network: Linear -> BN -> ReLU -> Dropout blocks.

    Input  : (batch, in_dim)  — subject-level hand-crafted feature vector
    Hidden : 128 -> 64 -> 32 with BatchNorm1d / ReLU / Dropout(0.3)
    Output : (batch, 1) sigmoid — P(SZ)
    """

    def __init__(self, in_dim, hidden=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers, d_in = [], in_dim
        for d_out in hidden:
            layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out),
                       nn.ReLU(), nn.Dropout(dropout)]
            d_in = d_out
        layers += [nn.Linear(d_in, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def train_fnn(X_train, y_train, X_val, y_val, in_dim, epochs=150, lr=1e-3,
              batch_size=16, weight_decay=1e-4, seed=SEED, device=None):
    """Train the basic FNN with early stopping on the val set (by AUC).

    Returns (model, val_predictions).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BasicFNN(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()
    X_train = torch.tensor(np.asarray(X_train, dtype=np.float32), device=device)
    y_train = torch.tensor(np.asarray(y_train, dtype=np.float32), device=device).view(-1, 1)
    n = len(X_train)
    best_auc, best_state, patience, wait = -1.0, None, 20, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(torch.tensor(np.asarray(X_val, dtype=np.float32),
                                    device=device)).cpu().numpy().ravel()
        auc = _auc(y_val, pv)
        if auc > best_auc:
            best_auc, best_state, wait = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = model(torch.tensor(np.asarray(X_val, dtype=np.float32),
                                device=device)).cpu().numpy().ravel()
    return model, pv


def predict_fnn(model, X, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(np.asarray(X, dtype=np.float32),
                                  device=device)).cpu().numpy().ravel()


def _auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.5
