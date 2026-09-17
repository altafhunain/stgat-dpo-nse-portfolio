"""
models/gnn/stgat_dpo.py
========================
Spatio-Temporal Graph Attention Network
with Differentiable Portfolio Optimization (STGAT-DPO)

Architecture
------------
  [N stocks × T days × F features]
        ↓
  TemporalEncoder     — independent LSTM per stock over T-day window
        ↓  [N × H_lstm]
  DynamicGraph        — HMM-conditioned correlation adjacency [N × N]
        ↓
  GraphAttentionLayer — 2-head GAT message passing  [N × H_gat]
        ↓
  ReturnHead          — MLP → predicted excess return μ̂ [N]
  CovarianceHead      — dot-product → PSD covariance Σ̂ [N × N]
        ↓
  PortfolioLayer      — differentiable Markowitz allocation → w [N]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# HMM REGIME DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class HMMRegimeDetector:
    """
    3-state Gaussian HMM fitted on Nifty 50 daily log-returns.
    States are re-labelled: 0=bear, 1=sideways, 2=bull
    by mean return ordering.
    """
    def __init__(self, n_states: int = 3, seed: int = 42):
        self.n_states  = n_states
        self.seed      = seed
        self.model     = None
        self.state_map = None        # raw state → semantic label

    def fit(self, nifty_returns: np.ndarray):
        """
        Fit HMM on 1-D array of daily log returns.
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            raise ImportError("hmmlearn required: pip install hmmlearn")

        r = nifty_returns.astype(np.float64)
        T = len(r)
        vol = np.array([r[max(0,i-21):i+1].std()*(252**0.5) for i in range(T)])
        vov = np.array([vol[max(0,i-63):i+1].std() for i in range(T)])
        mom = np.array([r[max(0,i-21):i+1].sum() for i in range(T)])
        X   = np.column_stack([r, vol, vov, mom])
        self.feat_mean = X.mean(axis=0)
        self.feat_std  = X.std(axis=0) + 1e-8
        X   = (X - self.feat_mean) / self.feat_std

        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=500,
            tol=1e-4,
            random_state=self.seed,
        )
        self.model.fit(X)

        # Map raw states → semantic labels by mean return
        means = self.model.means_[:, 0]
        order = np.argsort(means)   # ascending: bear < sideways < bull
        # state_map[raw_state] = semantic_label (0,1,2)
        self.state_map = {order[i]: i for i in range(self.n_states)}
        return self

    def predict(self, nifty_returns: np.ndarray) -> np.ndarray:
        """
        Viterbi-decode regime states for a return sequence.
        Returns int array of semantic labels:
          0=bear, 1=sideways, 2=bull, 3=momentum_bull (4-state only)
        Momentum-bull: bull days with high cross-sectional return dispersion
        (clear momentum leaders/laggards) get lambda=1.5 (less conservative).
        """
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        r=nifty_returns.astype(np.float64); T=len(r)
        vol=np.array([r[max(0,i-21):i+1].std()*(252**0.5) for i in range(T)])
        vov=np.array([vol[max(0,i-63):i+1].std() for i in range(T)])
        mom=np.array([r[max(0,i-21):i+1].sum() for i in range(T)])
        X=np.column_stack([r,vol,vov,mom]); X=(X-X.mean(axis=0))/(X.std(axis=0)+1e-8)
        raw = self.model.predict(X)
        states = np.array([self.state_map[s] for s in raw], dtype=np.int32)

        if self.n_states == 4:
            # Split bull days (state==2) into:
            #   2 = mean-reverting bull  (vov below median on bull days)
            #   3 = momentum bull        (vov above median on bull days)
            # vov (vol-of-vol) is a proxy for cross-sectional momentum dispersion:
            # high vov = trending/directional market = momentum regime
            bull_mask = (states == 2)
            if bull_mask.sum() > 10:
                vov_norm = (vov - vov.mean()) / (vov.std() + 1e-8)
                threshold = np.median(vov_norm[bull_mask])
                momentum_mask = bull_mask & (vov_norm > threshold)
                states[momentum_mask] = 3

        return states

    def predict_proba(self, nifty_returns: np.ndarray) -> np.ndarray:
        """
        Posterior state probabilities for each timestep. [T, 3]
        """
        r=nifty_returns.astype(np.float64); T=len(r)
        vol=np.array([r[max(0,i-21):i+1].std()*(252**0.5) for i in range(T)])
        vov=np.array([vol[max(0,i-63):i+1].std() for i in range(T)])
        mom=np.array([r[max(0,i-21):i+1].sum() for i in range(T)])
        X=np.column_stack([r,vol,vov,mom]); X=(X-X.mean(axis=0))/(X.std(axis=0)+1e-8)
        _, posteriors = self.model.score_samples(X)
        # Reorder columns by semantic label
        reordered = np.zeros_like(posteriors)
        for raw, sem in self.state_map.items():
            reordered[:, sem] = posteriors[:, raw]
        return reordered.astype(np.float32)

    def predict_causal(self, nifty_returns: np.ndarray, min_window: int = 30) -> np.ndarray:
        """
        Leakage-free alternative to predict(): re-decodes an expanding
        window ending at each day (using frozen training-set normalisation
        stats from fit(), never test-period statistics) and keeps only the
        terminal state, so no state label depends on future observations.
        Returns int array of semantic labels (0=bear, 1=sideways, 2=bull).
        Does not support the 4-state momentum-bull split.
        """
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        if not hasattr(self, "feat_mean"):
            raise RuntimeError("This model was fit before predict_causal support was added; refit.")
        r = nifty_returns.astype(np.float64)
        T = len(r)
        vol = np.array([r[max(0,i-21):i+1].std()*(252**0.5) for i in range(T)])
        vov = np.array([vol[max(0,i-63):i+1].std() for i in range(T)])
        mom = np.array([r[max(0,i-21):i+1].sum() for i in range(T)])
        X_raw = np.column_stack([r, vol, vov, mom])
        X_norm = (X_raw - self.feat_mean) / self.feat_std

        states = np.zeros(T, dtype=np.int32)
        for i in range(T):
            if i < min_window:
                states[i] = states[i-1] if i > 0 else 0
                continue
            raw_path = self.model.predict(X_norm[:i+1])
            states[i] = self.state_map[raw_path[-1]]
        return states


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_MAP = {
    "IT":           ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","MPHASIS.NS"],
    "Banking":      ["HDFCBANK.NS","ICICIBANK.NS","AXISBANK.NS","INDUSINDBK.NS",
                     "BAJFINANCE.NS","SBIN.NS"],
    "FMCG":         ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS",
                     "DABUR.NS","GODREJCP.NS","TATACONSUM.NS","UBL.NS"],
    "Auto":         ["MARUTI.NS","BAJAJFINSV.NS","M&M.NS"],
    "Pharma":       ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS"],
    "Energy":       ["RELIANCE.NS","NTPC.NS","COALINDIA.NS","BPCL.NS",
                     "ONGC.NS","TATAPOWER.NS"],
    "Infrastructure":["LT.NS","ADANIPORTS.NS","ULTRACEMCO.NS"],
    "Consumer":     ["ASIANPAINT.NS","TITAN.NS","BERGEPAINT.NS","HAVELLS.NS","VOLTAS.NS"],
    "Telecom":      ["BHARTIARTL.NS"],
    "Metals":       ["HINDALCO.NS","JSWSTEEL.NS"],
    "Insurance":    ["HDFCLIFE.NS","SBILIFE.NS","ICICIPRULI.NS"],
    "Power":        ["POWERGRID.NS"],
    "Chemicals":    ["PIDILITIND.NS"],
    "Cement":       ["GRASIM.NS","SHREECEM.NS","AMBUJACEM.NS"],
}

# Regime-conditioned correlation thresholds for edge creation
REGIME_THRESHOLDS = {2: 0.25, 1: 0.35, 0: 0.45, 3: 0.20}  # bull / sideways / bear / momentum_bull

# Regime-conditioned risk aversion λ
REGIME_LAMBDA = {2: 2.5, 1: 4.0, 0: 7.0, 3: 1.5}


def build_sector_adj(tickers: list[str]) -> np.ndarray:
    """
    Static sector adjacency matrix. 1 if same sector, 0 otherwise.
    Shape: [N, N]
    """
    N   = len(tickers)
    adj = np.zeros((N, N), dtype=np.float32)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    for sector, members in SECTOR_MAP.items():
        indices = [ticker_to_idx[t] for t in members if t in ticker_to_idx]
        for i in indices:
            for j in indices:
                if i != j:
                    adj[i, j] = 1.0
    return adj


def build_dynamic_adj(returns_window: np.ndarray,
                       regime: int,
                       sector_adj: np.ndarray,
                       sector_weight: float = 0.3,
                       eligible_mask: np.ndarray = None) -> np.ndarray:
    """
    Build adjacency matrix for a given window of returns.

    Parameters
    ----------
    returns_window : [T_corr, N]  daily returns for correlation window
    regime         : 0=bear, 1=sideways, 2=bull
    sector_adj     : [N, N] static sector adjacency
    sector_weight  : weight of sector edges in final adjacency

    Returns
    -------
    adj : [N, N] float32, values in [0, 1]
    """
    threshold = REGIME_THRESHOLDS[regime]

    # Dynamic correlation edges
    corr = np.corrcoef(returns_window.T)   # [N, N]
    corr = np.nan_to_num(corr, nan=0.0)

    dynamic_adj = (np.abs(corr) > threshold).astype(np.float32)
    np.fill_diagonal(dynamic_adj, 0.0)    # no self-loops

    # Use correlation magnitude as edge weight (capped)
    dynamic_adj = dynamic_adj * np.abs(corr).clip(0, 1)
    if eligible_mask is not None:
        ineligible = ~eligible_mask
        dynamic_adj[ineligible, :] = 0.0
        dynamic_adj[:, ineligible] = 0.0

    # Combine with static sector adjacency
    adj = (1 - sector_weight) * dynamic_adj + sector_weight * sector_adj
    adj = adj.clip(0, 1)
    np.fill_diagonal(adj, 0.0)

    return adj.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL ENCODER — LSTM PER STOCK
# ─────────────────────────────────────────────────────────────────────────────
class TemporalEncoder(nn.Module):
    """
    Shared-weight LSTM applied independently to each stock's time series.
    Input:  [B, N, T, F]
    Output: [B, N, H]
    """
    def __init__(self, n_features: int, hidden_dim: int = 64,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, F]
        B, N, T, F = x.shape
        x_flat = x.reshape(B * N, T, F)             # [B*N, T, F]
        out, _ = self.lstm(x_flat)                   # [B*N, T, H]
        h_last  = out[:, -1, :]                      # [B*N, H] — last timestep
        h_last  = self.norm(h_last)
        return h_last.reshape(B, N, self.hidden_dim) # [B, N, H]


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ATTENTION LAYER
# ─────────────────────────────────────────────────────────────────────────────
class GraphAttentionLayer(nn.Module):
    """
    Multi-head Graph Attention (Veličković et al., 2018).
    Dense implementation — suitable for N=51 (O(N²) trivially fast).

    Input:  h [B, N, in_dim],  adj [B, N, N] or [N, N]
    Output: [B, N, out_dim]
    """
    def __init__(self, in_dim: int, out_dim: int,
                 n_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads
        self.out_dim  = out_dim

        self.W       = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src   = nn.Parameter(torch.Tensor(1, n_heads, self.head_dim))
        self.a_dst   = nn.Parameter(torch.Tensor(1, n_heads, self.head_dim))
        self.leaky   = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(out_dim)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        h   : [B, N, in_dim]
        adj : [B, N, N]  or  [N, N]
        """
        B, N, _ = h.shape

        # Project
        Wh = self.W(h)                           # [B, N, out_dim]
        Wh = Wh.view(B, N, self.n_heads,
                     self.head_dim)              # [B, N, H, D]

        # Attention scores: e_ij = LeakyReLU(a_src·h_i + a_dst·h_j)
        # a_src: [1, H, D], Wh: [B, N, H, D]
        e_src = (Wh * self.a_src).sum(-1)        # [B, N, H]
        e_dst = (Wh * self.a_dst).sum(-1)        # [B, N, H]

        # Broadcast: e[b,i,j,h] = e_src[b,i,h] + e_dst[b,j,h]
        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)   # [B, N, N, H]
        e = self.leaky(e)

        # Mask non-edges
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).unsqueeze(-1)      # [1, N, N, 1]
        else:
            adj = adj.unsqueeze(-1)                   # [B, N, N, 1]

        e = e.masked_fill(adj == 0, float("-inf"))
        alpha = F.softmax(e, dim=2)                   # [B, N, N, H]
        # Replace NaN from all-zero rows (isolated nodes) with 0
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        # Aggregate: h'[b,i,h,d] = sum_j alpha[b,i,j,h] * Wh[b,j,h,d]
        # alpha: [B, N, N, H], Wh: [B, N, H, D]
        # out[b,i,h,d] = sum_j alpha[b,i,j,h] * Wh[b,j,h,d]
        out = torch.einsum("bijh,bjhd->bihd",
                            alpha, Wh)               # [B, N, H, D]
        out = out.reshape(B, N, self.out_dim)        # [B, N, out_dim]
        out = F.elu(out)
        out = self.norm(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO LAYER — DIFFERENTIABLE MARKOWITZ
# ─────────────────────────────────────────────────────────────────────────────
class PortfolioLayer(nn.Module):
    """
    Differentiable portfolio optimization.

    Given predicted returns μ̂ and covariance Σ̂, computes
    Markowitz weights subject to long-only and weight cap constraints.

    Strategy: analytical unconstrained solution w* ∝ Σ̂⁻¹μ̂,
    followed by differentiable projection onto the feasible set.

    This is fully differentiable — gradients flow back through
    Cholesky decomposition into the GNN parameters.
    """
    def __init__(self, n_stocks: int, max_weight: float = 0.15):
        super().__init__()
        self.n         = n_stocks
        self.max_w     = max_weight
        self.log_lambda = nn.Parameter(torch.tensor(0.0))  # learnable risk aversion

    def forward(self, mu: torch.Tensor,
                sigma: torch.Tensor,
                regime_lambda: Optional[float] = None) -> torch.Tensor:
        """
        mu    : [B, N]  predicted excess returns
        sigma : [B, N, N]  predicted covariance matrix
        regime_lambda : override λ if provided (from HMM state)

        Returns w : [B, N]  portfolio weights
        """
        B, N = mu.shape

        # Risk aversion: use regime override if given, else learned
        if regime_lambda is not None:
            lam = regime_lambda
        else:
            lam = torch.exp(self.log_lambda).clamp(0.5, 20.0)

        # Regularise covariance for numerical stability
        eps = 1e-4
        sigma_reg = sigma + eps * torch.eye(N, device=sigma.device).unsqueeze(0)

        # Cholesky decomposition — differentiable in PyTorch
        try:
            L     = torch.linalg.cholesky(sigma_reg)       # [B, N, N]
            # Solve Σw = μ/λ  via  Σ = L L'
            mu_lam = mu / lam                               # [B, N]
            # Forward substitution: solve L y = μ/λ
            y     = torch.linalg.solve_triangular(
                        L, mu_lam.unsqueeze(-1), upper=False)   # [B, N, 1]
            # Backward substitution: solve L' w = y
            w_raw = torch.linalg.solve_triangular(
                        L.transpose(-2, -1), y, upper=True)     # [B, N, 1]
            w_raw = w_raw.squeeze(-1)                           # [B, N]
        except RuntimeError:
            # Fallback: diagonal approximation if Cholesky fails
            diag_var = torch.diagonal(sigma_reg, dim1=-2, dim2=-1)  # [B, N]
            w_raw    = mu / (lam * diag_var.clamp(min=1e-6))

        # ── Constrained projection ────────────────────────────────────────
        # Step 1: Non-negativity (long-only) — differentiable via softplus
        w_pos = F.softplus(w_raw, beta=5)     # smooth approximation of relu

        # Step 2: Max-weight cap — differentiable via sigmoid rescaling
        w_cap = self.max_w * torch.sigmoid(w_pos / self.max_w * 5 - 2.5)

        # Step 3: Normalize to sum = 1
        w_sum = w_cap.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        w     = w_cap / w_sum

        # Final safety clip (non-differentiable but prevents extreme values)
        # Only applied during inference
        if not self.training:
            w = w.clamp(0, self.max_w)
            w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return w


# ─────────────────────────────────────────────────────────────────────────────
# STGAT-DPO — FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────
class STGATDPOModel(nn.Module):
    """
    Spatio-Temporal Graph Attention Network with
    Differentiable Portfolio Optimization.

    Forward pass:
        x    : [B, N, T, F]   node feature sequences
        adj  : [B, N, N]      dynamic adjacency matrices
        regime_lambda : float  risk aversion from HMM state (optional)

    Returns:
        w    : [B, N]   portfolio weights
        mu   : [B, N]   predicted excess returns
        cov  : [B, N, N] predicted covariance
    """
    def __init__(self,
                 n_stocks:    int   = 51,
                 n_features:  int   = 8,
                 lookback:    int   = 20,
                 lstm_hidden: int   = 64,
                 lstm_layers: int   = 2,
                 gat_hidden:  int   = 64,
                 gat_heads:   int   = 2,
                 dropout:     float = 0.2,
                 max_weight:  float = 0.15):
        super().__init__()

        self.n_stocks   = n_stocks
        self.n_features = n_features
        self.lookback   = lookback

        # ── Temporal encoder ──────────────────────────────────────────────
        self.temporal_encoder = TemporalEncoder(
            n_features=n_features,
            hidden_dim=lstm_hidden,
            n_layers=lstm_layers,
            dropout=dropout,
        )

        # ── Graph attention layers (2 stacked) ───────────────────────────
        self.gat1 = GraphAttentionLayer(
            in_dim=lstm_hidden,
            out_dim=gat_hidden,
            n_heads=gat_heads,
            dropout=dropout,
        )
        self.gat2 = GraphAttentionLayer(
            in_dim=gat_hidden,
            out_dim=gat_hidden,
            n_heads=gat_heads,
            dropout=dropout,
        )

        # Residual projection if dimensions don't match
        self.res_proj = (nn.Linear(lstm_hidden, gat_hidden, bias=False)
                         if lstm_hidden != gat_hidden else nn.Identity())

        # -- Return head --------------------------------------------------
        self.return_head = nn.Sequential(
            nn.Linear(gat_hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # ── Covariance head ───────────────────────────────────────────────
        # Projects embeddings to covariance factor space
        # Σ̂ = F F' / d + diag(σ²)   — guaranteed PSD
        self.cov_proj = nn.Linear(gat_hidden, 16, bias=False)
        self.diag_vol = nn.Sequential(
            nn.Linear(gat_hidden, 1),
            nn.Softplus(),           # ensure positive diagonal
        )

        # ── Portfolio layer ───────────────────────────────────────────────
        self.portfolio = PortfolioLayer(
            n_stocks=n_stocks,
            max_weight=max_weight,
        )

    def encode(self, x: torch.Tensor,
               adj: torch.Tensor) -> torch.Tensor:
        """
        x   : [B, N, T, F]
        adj : [B, N, N]
        Returns node embeddings [B, N, gat_hidden]
        """
        # Temporal encoding
        h = self.temporal_encoder(x)              # [B, N, lstm_hidden]

        # Residual connection across GAT layers
        res = self.res_proj(h)                    # [B, N, gat_hidden]

        # GAT layer 1
        h = self.gat1(h, adj)                     # [B, N, gat_hidden]
        h = h + res                               # residual add

        # GAT layer 2
        res2 = h
        h = self.gat2(h, adj)                     # [B, N, gat_hidden]
        h = h + res2                              # residual add

        return h

    def predict_returns(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, N, gat_hidden] -> mu_hat: [B, N]"""
        return self.return_head(h).squeeze(-1)

    def predict_covariance(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, N, gat_hidden] → Σ̂: [B, N, N]
        Uses low-rank + diagonal structure: Σ̂ = F F' / k + diag(σ²)
        """
        F_mat = self.cov_proj(h)                  # [B, N, k=16]
        d     = F_mat.shape[-1]
        # Low-rank component
        cov_lr  = torch.bmm(F_mat, F_mat.transpose(1, 2)) / d  # [B, N, N]
        # Diagonal component (stock-specific volatility)
        diag_v  = self.diag_vol(h).squeeze(-1)    # [B, N]
        cov_diag = torch.diag_embed(diag_v)       # [B, N, N]
        # Combined — guaranteed PSD
        return cov_lr + cov_diag

    def forward(self,
                x: torch.Tensor,
                adj: torch.Tensor,
                regime_lambda: Optional[float] = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (weights [B,N], mu [B,N], cov [B,N,N])
        """
        h   = self.encode(x, adj)
        mu  = self.predict_returns(h)
        cov = self.predict_covariance(h)
        w   = self.portfolio(mu, cov, regime_lambda)
        return w, mu, cov

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def sharpe_loss(portfolio_returns: torch.Tensor,
                rf_daily: float = 0.065 / 252) -> torch.Tensor:
    """
    Negative annualised Sharpe ratio loss.
    portfolio_returns: [B] realized daily portfolio returns
    """
    excess = portfolio_returns - rf_daily
    mean   = excess.mean()
    std    = excess.std()
    if torch.isnan(std) or std < 1e-6:
        return -mean * (252 ** 0.5)
    sharpe = mean / std * (252 ** 0.5)
    return -sharpe


def information_ratio_loss(port_rets: torch.Tensor,
                            bench_rets: torch.Tensor) -> torch.Tensor:
    """
    Negative Information Ratio loss (primary objective — beat Nifty 50).
    Falls back to negative mean return if std is degenerate.
    """
    active = port_rets - bench_rets
    mean   = active.mean()
    std    = active.std()
    if torch.isnan(std) or std < 1e-6:
        # Fallback: negative mean excess return (still meaningful gradient)
        return -mean * (252 ** 0.5)
    ir = mean / std * (252 ** 0.5)
    return -ir


def pinball_loss(quantile_preds: torch.Tensor,
                 ret_actual: torch.Tensor,
                 quantile_levels: list = [0.05, 0.25, 0.50, 0.75, 0.95]
                 ) -> torch.Tensor:
    """
    Quantile (pinball) loss for the quantile-head predictions.
    quantile_preds : [B, N, Q]  predicted quantiles
    ret_actual     : [B, N]     realised returns (broadcast over Q)
    """
    target = ret_actual.unsqueeze(-1)                    # [B, N, 1]
    errors = target - quantile_preds                     # [B, N, Q]
    losses = []
    for i, q in enumerate(quantile_levels):
        e = errors[..., i]
        losses.append(torch.max((q - 1) * e, q * e))
    return torch.stack(losses, dim=-1).mean()
def combined_loss(port_rets: torch.Tensor,
                  bench_rets: torch.Tensor,
                  mu_pred: torch.Tensor,
                  ret_actual: torch.Tensor,
                  alpha: float = 0.6,
                  beta: float  = 0.2,
                  gamma: float = 0.2,
                  quantile_preds: torch.Tensor = None,
                  quantile_levels: list = None) -> torch.Tensor:
    """
    Combined loss:
      alpha x (-IR) + beta x MSE(mu_hat, actual_returns)
      + gamma x pinball_loss(quantiles, actual_returns)   [if quantile_preds given]

    Backward compatible: if quantile_preds is None, behaves exactly as
    the original two-term loss (gamma term is simply omitted).
    """
    ir_loss  = information_ratio_loss(port_rets, bench_rets)
    mse_loss = F.mse_loss(mu_pred, ret_actual)
    total = alpha * ir_loss + beta * mse_loss
    if quantile_preds is not None:
        levels = quantile_levels or [0.05, 0.25, 0.50, 0.75, 0.95]
        total = total + gamma * pinball_loss(quantile_preds, ret_actual, levels)
    return total
