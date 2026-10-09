import numpy as np, torch, torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

LOGCOLS = [0, 1, 2, 3, 4, 8, 9, 10, 11, 13]


def prep_fit(X):
    Z = X.values.copy(); Z[:, LOGCOLS] = np.log1p(Z[:, LOGCOLS])
    sc = StandardScaler().fit(Z); return sc


def prep(sc, X):
    Z = X.values.copy(); Z[:, LOGCOLS] = np.log1p(Z[:, LOGCOLS])
    return sc.transform(Z).astype(np.float32)


class Mahalanobis:
    name = "Mahalanobis (baseline)"
    def fit(self, Z):
        self.mu = Z.mean(0); self.P = np.linalg.pinv(np.cov(Z.T)); return self
    def score(self, Z):
        D = Z - self.mu; return np.einsum("ij,jk,ik->i", D, self.P, D)


class IForest:
    name = "Isolation Forest"
    def fit(self, Z):
        self.m = IsolationForest(n_estimators=300, max_samples=512, random_state=0, n_jobs=-1).fit(Z); return self
    def score(self, Z): return -self.m.score_samples(Z)


class Autoencoder:
    name = "Autoencoder (PyTorch)"
    def __init__(self, d, epochs=25, seed=0):
        torch.manual_seed(seed); self.epochs = epochs
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU(),
                                 nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, d))
    def fit(self, Z):
        X = torch.from_numpy(Z); opt = torch.optim.Adam(self.net.parameters(), 2e-3)
        for _ in range(self.epochs):
            for i in torch.randperm(len(X)).split(512):
                opt.zero_grad(); l = ((self.net(X[i]) - X[i]) ** 2).mean(); l.backward(); opt.step()
        return self
    @torch.no_grad()
    def score(self, Z):
        X = torch.from_numpy(Z); return ((self.net(X) - X) ** 2).mean(1).numpy()


class Ensemble:
    """Average of percentile ranks vs. a benign reference distribution."""
    name = "Ensemble (IF + AE)"
    def __init__(self, members): self.m = members
    def fit_ref(self, Zval): self.ref = [np.sort(m.score(Zval)) for m in self.m]; return self
    def score(self, Z):
        return np.mean([np.searchsorted(r, m.score(Z)) / len(r) for m, r in zip(self.m, self.ref)], axis=0)
