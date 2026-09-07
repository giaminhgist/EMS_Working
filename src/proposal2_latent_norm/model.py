"""Proposal 2 — Learned Stimulus-Conditioned Normative Modeling.

Encoder f_theta maps each stimulus feature vector to a latent z; a HC latent
normative bank (per-stimulus mu^z, sigma^z) is recomputed EVERY EPOCH from the
train-fold HC encodings (model.refresh_bank hook, called by the trainer);
a comparator produces per-stimulus deviation d_{i,s}; a set-level pooling
aggregates the S deviations into the subject embedding; an MLP head classifies.
See docs/research/design.md.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # EMS-Minh/src
from data.common import labels_of  # noqa: E402
from data.tabular import subject_matrices, feature_norm_stats  # noqa: E402

EPS = 1e-6


class LatentNormDataset:
    """Subjects with per-stimulus standardized feature tensors + masks.

    Input normalization uses feature stats fitted on train subjects only
    (unsupervised, leakage-safe). The encoder consumes (B, S, D) per batch.
    """

    def __init__(self, subject_ids, train_ids_for_stats):
        self.subjects = list(subject_ids)
        self.labels = labels_of(self.subjects)
        mu, sigma = feature_norm_stats(train_ids_for_stats)
        mats = subject_matrices(self.subjects)
        self.X, self.mask = [], []
        for s in self.subjects:
            X, mask = mats[s]
            X = (np.nan_to_num(X, nan=0.0) - mu) / sigma
            X = X * mask[:, None].astype(np.float32)
            self.X.append(torch.from_numpy(X.astype(np.float32)))
            self.mask.append(torch.from_numpy(mask.astype(np.float32)))

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return (self.X[idx], self.mask[idx]), self.labels[idx], self.subjects[idx]

    def collate(self, batch):
        return (torch.stack([b[0] for b in batch], 0),
                torch.stack([b[1] for b in batch], 0))

    def hc_indices(self):
        return [i for i, s in enumerate(self.subjects) if s < 200]


class LatentNormModel(nn.Module):
    """Encoder + HC latent bank + comparator + set pooling + head.

    comparator: 'mlp' (g_phi([z, mu, z-mu, z*mu])) | 'sub' (z-mu) | 'zsub'
    pool:       'attention' | 'mean' | 'deepset' (mean||max)
    lambda_norm: weight of the HC latent-concentration regularization
                 (per-batch, graph-attached).
    """

    def __init__(self, d_in=45, latent_dim=128, dev_dim=64, comparator="mlp",
                 pool="attention", lambda_norm=0.0, dropout=0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.comparator = comparator
        self.pool = pool
        self.lambda_norm = lambda_norm
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, latent_dim))
        if comparator == "mlp":
            self.comp = nn.Sequential(
                nn.Linear(latent_dim * 4, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, dev_dim))
        else:  # 'sub' | 'zsub'
            self.proj = nn.Linear(latent_dim, dev_dim)
        if pool == "attention":
            self.scorer = nn.Sequential(nn.Linear(dev_dim, 32), nn.ReLU(),
                                        nn.Linear(32, 1))
        self.head = nn.Sequential(
            nn.Linear(dev_dim * (2 if pool == "deepset" else 1), 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))
        # HC latent normative bank (per stimulus) — refreshed every epoch
        self.register_buffer("bank_mu", torch.zeros(100, latent_dim))
        self.register_buffer("bank_sigma", torch.ones(100, latent_dim))

    # ------------------------------------------------------------------ #
    # bank refresh hook (trainer calls it once per epoch, before training)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def refresh_bank(self, train_ds):
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        hc = train_ds.hc_indices()
        D = torch.stack([train_ds[i][0][0] for i in hc], 0).to(device)
        mask = torch.stack([train_ds[i][0][1] for i in hc], 0).to(device)
        z, _ = self._encode(D, mask)
        valid = mask[..., None]
        mu = (z * valid).sum(0) / valid.sum(0).clamp(min=1)          # (S, L)
        var = ((z - mu[None]) ** 2 * valid).sum(0) / valid.sum(0).clamp(min=1)
        self.bank_mu.copy_(mu)
        self.bank_sigma.copy_(torch.sqrt(var.clamp(min=EPS)) + EPS)
        self.train(was_training)

    # ------------------------------------------------------------------ #
    def _encode(self, D, mask):
        B, S, _ = D.shape
        z = self.encoder(D.view(B * S, -1)).view(B, S, self.latent_dim)
        return z, mask

    def _deviate(self, z, mask, y=None):
        B, S, L = z.shape
        mu = self.bank_mu[None].expand(B, S, L)
        sig = self.bank_sigma[None].expand(B, S, L)
        if self.comparator == "mlp":
            feat = torch.cat([z, mu, z - mu, z * mu], dim=-1)
            d = self.comp(feat)
        elif self.comparator == "sub":
            d = self.proj(z - mu)
        else:  # zsub
            d = self.proj((z - mu) / (sig + EPS))
        d = d * mask[..., None]
        # HC latent-concentration regularization (per batch, graph-attached)
        norm_loss = None
        if self.training and self.lambda_norm > 0 and y is not None:
            hc = (y.view(-1) == 0)
            if hc.any():
                sq = ((z - mu) ** 2).sum(-1) * mask          # (B, S)
                norm_loss = sq[hc].mean() * self.lambda_norm
        return d, norm_loss

    def _pool(self, d, mask):
        valid = mask[..., None]
        if self.pool == "mean":
            return (d * valid).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        if self.pool == "deepset":
            mean = (d * valid).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            maxv = (d * valid + (1 - valid) * (-1e9)).max(1).values
            return torch.cat([mean, maxv], dim=1)
        score = self.scorer(d) + (mask - 1)[..., None] * 1e9
        a = torch.softmax(score, dim=1)
        return (a * d * valid).sum(1)

    def forward(self, x, y=None):
        D, mask = x
        z, mask = self._encode(D, mask)
        d, norm_loss = self._deviate(z, mask, y)
        h = self._pool(d, mask)
        prob = torch.sigmoid(self.head(h))
        return prob, {"emb": h, "norm_loss": norm_loss}
