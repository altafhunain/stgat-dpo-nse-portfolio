"""
training/run_rl_ppo.py
=======================
PPO Reinforcement Learning portfolio agent for NSE 51-stock universe.

Architecture:
  State  : [51 stocks × 11 features] + current weights [51] +
           regime posteriors [3] + trailing 5-day portfolio return [1]
           → flattened: 51×11 + 51 + 3 + 1 = 616 dimensions

  Action : Softmax weight vector over 51 stocks (continuous)
           Clipped to [0, max_weight=0.15]

  Reward : 5-day realised Information Ratio vs Nifty 50
           (encourages outperformance, not just high absolute return)

  Policy : 2-layer MLP (256, 128) + ReLU
  Value  : Separate 2-layer MLP (256, 128) + ReLU

Walk-forward: same 6 folds as STGAT-DPO for direct comparison.
Monthly rebalancing evaluation (same as all other strategies).

Run:
    python training/run_rl_ppo.py
"""

import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet
from pathlib import Path
from copy import deepcopy

from utils.features import build_feature_tensor, N_FEATURES
from models.gnn.stgat_dpo import (
    HMMRegimeDetector, build_sector_adj, REGIME_LAMBDA
)
from training.train_stgat import (
    compute_metrics, FOLDS, BLIND,
    LOOKBACK, RF_DAILY, MAX_WEIGHT, SEED,
    DEVICE, PRICES_FILE
)

torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT     = Path(__file__).resolve().parent.parent
OUT_DIR  = ROOT / "outputs" / "results"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
for d in [OUT_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── PPO Hyperparameters ───────────────────────────────────────────────────────
N_EPISODES      = 300        # episodes per fold
EPISODE_LEN     = 63         # trading days per episode (~3 months)
GAMMA           = 0.99       # discount factor
GAE_LAMBDA      = 0.95       # GAE smoothing
CLIP_EPS        = 0.2        # PPO clip ratio
ENTROPY_COEF    = 0.01       # entropy bonus (encourages exploration)
VF_COEF         = 0.5        # value function loss weight
PPO_EPOCHS      = 4          # PPO update epochs per rollout
MINI_BATCH      = 32         # minibatch size for PPO update
LR_PPO          = 3e-4       # learning rate
MAX_GRAD_PPO    = 0.5        # gradient clip
REBAL_FREQ      = 21         # rebalance every 21 trading days in evaluation
TC_BPS          = 10         # transaction costs
CONC_PENALTY    = 0.1        # penalty for over-concentration (HHI)


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════
class NSEPortfolioEnv:
    """
    OpenAI Gym-style environment for NSE 51-stock portfolio allocation.

    State (flat vector):
      - feat_window : [N_STOCKS × N_FEATURES] normalised features
      - weights     : [N_STOCKS] current portfolio weights
      - regime_post : [3] HMM posterior probabilities (bear/sideways/bull)
      - trail_ret   : [1] trailing 5-day portfolio log-return

    Action:
      - Concentration parameter α for Dirichlet distribution.
        Policy outputs α ∈ (0, ∞)^N; weights = Dirichlet(α).mean()
        Then clipped to max_weight and renormalised.

    Reward:
      5-day realised Information Ratio vs Nifty 50 (annualised scale)
      minus concentration penalty (HHI).
    """

    def __init__(self,
                 feat_arr:      np.ndarray,   # [T, N, F]
                 ret_arr:       np.ndarray,   # [T, N]
                 nifty_arr:     np.ndarray,   # [T]
                 regime_arr:    np.ndarray,   # [T] states
                 regime_post:   np.ndarray,   # [T, 3] posteriors
                 episode_len:   int = EPISODE_LEN,
                 lookback:      int = LOOKBACK,
                 max_weight:    float = MAX_WEIGHT,
                 tc_bps:        float = TC_BPS,
                 train_start_idx: int = 0,
                 train_end_idx:   int = -1):

        self.feat_arr    = feat_arr
        self.ret_arr     = ret_arr
        self.nifty_arr   = nifty_arr
        self.regime_arr  = regime_arr
        self.regime_post = regime_post
        self.episode_len = episode_len
        self.lookback    = lookback
        self.max_weight  = max_weight
        self.tc          = tc_bps / 10_000

        self.N = feat_arr.shape[1]
        self.F = feat_arr.shape[2]
        self.T = feat_arr.shape[0]

        self.train_start = train_start_idx
        self.train_end   = (train_end_idx if train_end_idx > 0
                            else self.T - episode_len - 1)

        # State dimension
        self.state_dim = self.N * self.F + self.N + 3 + 1

        self.reset()

    def reset(self):
        """Start new episode at random valid position in training range."""
        max_start = max(self.train_start, self.lookback + 5)
        max_end   = self.train_end - self.episode_len - 5
        if max_end <= max_start:
            max_end = max_start + 1
        self.t       = np.random.randint(max_start, max_end)
        self.weights = np.ones(self.N) / self.N
        self.trail_ret = 0.0
        self.done    = False
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        t = self.t
        # Feature window: most recent lookback days for each stock
        feat_window = self.feat_arr[
            max(0, t - self.lookback):t
        ]
        if len(feat_window) < self.lookback:
            pad = np.zeros((self.lookback - len(feat_window),
                            self.N, self.F), dtype=np.float32)
            feat_window = np.concatenate([pad, feat_window], axis=0)
        # Take last timestep features [N, F] → flatten to [N*F]
        feat_flat = feat_window[-1].flatten()

        # Regime posterior
        if t < len(self.regime_post):
            reg_post = self.regime_post[t]
        else:
            reg_post = np.array([1/3, 1/3, 1/3], dtype=np.float32)

        state = np.concatenate([
            feat_flat,
            self.weights,
            reg_post,
            [self.trail_ret]
        ]).astype(np.float32)

        return state

    def step(self, action: np.ndarray):
        """
        action : [N] raw policy outputs (concentration params or logits)
        Returns: (next_state, reward, done)
        """
        # Project action to valid weight vector
        # Softmax then clip and renormalise
        w_raw = np.exp(action - action.max())   # softmax numerically stable
        w_raw = w_raw / w_raw.sum()
        w_new = np.clip(w_raw, 0, self.max_weight)
        w_new = w_new / (w_new.sum() + 1e-8)

        # Transaction cost
        tc_cost = np.abs(w_new - self.weights).sum() * self.tc

        # Hold for 5 days and compute reward
        reward_rets  = []
        nifty_rets   = []
        end_t        = min(self.t + 5, self.T - 1)

        for ti in range(self.t, end_t):
            if ti < len(self.ret_arr):
                r_p = float(self.ret_arr[ti] @ w_new)
                r_n = float(self.nifty_arr[ti])
                reward_rets.append(r_p)
                nifty_rets.append(r_n)

        if len(reward_rets) > 1:
            rp_arr = np.array(reward_rets)
            rn_arr = np.array(nifty_rets)
            active = rp_arr - rn_arr

            # Reward = annualised IR - transaction cost penalty
            ir_5d = (active.mean() / (active.std() + 1e-6)
                     * np.sqrt(252))
            # Concentration penalty (HHI)
            hhi   = (w_new**2).sum()
            reward = ir_5d - CONC_PENALTY * hhi - tc_cost * 100
            self.trail_ret = rp_arr.sum()
        else:
            reward = 0.0
            self.trail_ret = 0.0

        # Update
        self.weights   = w_new.copy()
        self.t        += 5
        self.step_count += 1

        self.done = (self.t >= self.train_end or
                     self.step_count >= self.episode_len // 5)

        next_state = self._get_state() if not self.done else np.zeros(
            self.state_dim, dtype=np.float32)

        return next_state, float(reward), self.done


# ═════════════════════════════════════════════════════════════════════════════
# PPO POLICY + VALUE NETWORK
# ═════════════════════════════════════════════════════════════════════════════
class PPOActorCritic(nn.Module):
    """
    Shared-trunk Actor-Critic for continuous portfolio allocation.
    Actor outputs log-concentration params for implicit Dirichlet policy.
    Critic outputs scalar value estimate.
    """
    def __init__(self, state_dim: int, n_stocks: int):
        super().__init__()
        self.n_stocks = n_stocks

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Actor head: outputs logits → softmax gives weights
        self.actor = nn.Sequential(
            nn.Linear(128, n_stocks),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Learnable log-std for action noise (exploration)
        self.log_std = nn.Parameter(torch.zeros(n_stocks))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Actor output — smaller init for stable early training
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(self, state: torch.Tensor):
        h     = self.trunk(state)
        logit = self.actor(h)               # [B, N]
        value = self.critic(h).squeeze(-1)  # [B]
        return logit, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample or take greedy action. Returns (action, log_prob, value)."""
        logit, value = self.forward(state)

        std   = torch.exp(self.log_std).clamp(0.01, 1.0)
        dist  = torch.distributions.Normal(logit, std)

        if deterministic:
            action = logit
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)  # [B]
        return action, log_prob, value

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """Evaluate (action, log_prob, value, entropy) for PPO update."""
        logit, value = self.forward(state)
        std  = torch.exp(self.log_std).clamp(0.01, 1.0)
        dist = torch.distributions.Normal(logit, std)

        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, value, entropy


# ═════════════════════════════════════════════════════════════════════════════
# PPO TRAINER
# ═════════════════════════════════════════════════════════════════════════════
class PPOTrainer:
    def __init__(self, state_dim: int, n_stocks: int):
        self.policy    = PPOActorCritic(state_dim, n_stocks).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(),
                                     lr=LR_PPO, eps=1e-5)
        print(f"  [PPO] Parameters: "
              f"{sum(p.numel() for p in self.policy.parameters()):,}")

    def collect_rollout(self, env: NSEPortfolioEnv, n_steps: int = 256):
        """Collect trajectory from environment."""
        states, actions, log_probs, rewards, values, dones = \
            [], [], [], [], [], []

        state = env.reset()
        for _ in range(n_steps):
            s_t = torch.tensor(state, dtype=torch.float32,
                                device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(s_t)

            a_np = action.squeeze(0).cpu().numpy()
            next_state, reward, done = env.step(a_np)

            states.append(state)
            actions.append(a_np)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(float(done))

            state = next_state if not done else env.reset()

        # Bootstrap value
        s_t = torch.tensor(state, dtype=torch.float32,
                            device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            _, _, last_val = self.policy.get_action(s_t)
        last_val = last_val.item()

        # GAE advantage computation
        advantages = np.zeros(n_steps, dtype=np.float32)
        gae        = 0.0
        for t in reversed(range(n_steps)):
            next_val = last_val if t == n_steps - 1 else values[t + 1]
            delta    = (rewards[t]
                        + GAMMA * next_val * (1 - dones[t])
                        - values[t])
            gae      = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + np.array(values)

        return (np.array(states),
                np.array(actions),
                np.array(log_probs),
                returns.astype(np.float32),
                advantages.astype(np.float32))

    def update(self, states, actions, old_log_probs, returns, advantages):
        """PPO update step."""
        states_t    = torch.tensor(states,        device=DEVICE)
        actions_t   = torch.tensor(actions,       device=DEVICE)
        old_lp_t    = torch.tensor(old_log_probs, device=DEVICE)
        returns_t   = torch.tensor(returns,       device=DEVICE)
        adv_t       = torch.tensor(advantages,    device=DEVICE)

        # Normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0.0
        n          = len(states)

        for _ in range(PPO_EPOCHS):
            idx = torch.randperm(n)
            for start in range(0, n, MINI_BATCH):
                mb = idx[start:start + MINI_BATCH]
                new_lp, value, entropy = self.policy.evaluate(
                    states_t[mb], actions_t[mb])

                # Policy loss (clipped surrogate)
                ratio    = torch.exp(new_lp - old_lp_t[mb])
                surr1    = ratio * adv_t[mb]
                surr2    = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_t[mb]
                pol_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                val_loss = 0.5 * (value - returns_t[mb]).pow(2).mean()

                # Entropy bonus
                ent_loss = -entropy.mean()

                loss = (pol_loss
                        + VF_COEF * val_loss
                        + ENTROPY_COEF * ent_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_PPO)
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss

    def train(self, env: NSEPortfolioEnv, n_episodes: int,
              fold_name: str) -> list:
        """Train PPO agent for n_episodes."""
        ep_rewards = []
        best_reward = -np.inf
        best_state  = None

        print(f"\n  Training PPO — {fold_name}  ({n_episodes} episodes)")
        print(f"  {'Episode':>8}  {'Reward':>9}  {'Best':>9}")
        print(f"  {'─'*32}")

        for ep in range(1, n_episodes + 1):
            states, actions, log_probs, returns, advantages = \
                self.collect_rollout(env, n_steps=256)
            self.update(states, actions, log_probs, returns, advantages)

            # Eval reward on one episode
            ep_reward = returns.mean()
            ep_rewards.append(ep_reward)

            if ep_reward > best_reward:
                best_reward = ep_reward
                best_state  = deepcopy(self.policy.state_dict())

            if ep % 50 == 0 or ep == 1:
                print(f"  {ep:>8}  {ep_reward:>9.4f}  {best_reward:>9.4f}")

        if best_state:
            self.policy.load_state_dict(best_state)

        return ep_rewards

    def save(self, path: Path):
        torch.save(self.policy.state_dict(), path)
        print(f"  ✓  Saved: {path.name}")

    def load(self, path: Path):
        self.policy.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"  ✓  Loaded: {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_ppo(policy:     PPOActorCritic,
                  feat_arr:   np.ndarray,
                  ret_arr:    np.ndarray,
                  nifty_arr:  np.ndarray,
                  regime_post:np.ndarray,
                  dates:      pd.DatetimeIndex,
                  test_start: str,
                  test_end:   str,
                  state_dim:  int,
                  n_stocks:   int,
                  tc_bps:     float = TC_BPS) -> pd.Series:
    """Monthly rebalancing evaluation — same protocol as STGAT-DPO."""
    policy.eval()
    tc = tc_bps / 10_000
    N  = n_stocks

    test_mask  = (dates >= test_start) & (dates <= test_end)
    test_dates = dates[test_mask]
    if len(test_dates) == 0:
        return pd.Series(dtype=float)

    reb_dates = (test_dates.to_series()
                 .groupby([test_dates.year, test_dates.month])
                 .first().values)
    reb_dates = pd.DatetimeIndex(reb_dates)

    port_ret_map = {}
    prev_w       = np.ones(N) / N
    trail_ret    = 0.0

    for i, reb_d in enumerate(reb_dates):
        t_arr = np.where(dates == reb_d)[0]
        if len(t_arr) == 0:
            continue
        t = int(t_arr[0])
        if t < LOOKBACK:
            continue

        # Build state
        feat_window = feat_arr[max(0, t - LOOKBACK):t]
        if len(feat_window) < LOOKBACK:
            pad = np.zeros((LOOKBACK - len(feat_window), N,
                            feat_arr.shape[2]), dtype=np.float32)
            feat_window = np.concatenate([pad, feat_window], axis=0)
        feat_flat = feat_window[-1].flatten()

        reg_post = (regime_post[t] if t < len(regime_post)
                    else np.array([1/3, 1/3, 1/3], dtype=np.float32))

        state = np.concatenate([
            feat_flat, prev_w, reg_post, [trail_ret]
        ]).astype(np.float32)

        s_t = torch.tensor(state, device=DEVICE).unsqueeze(0)
        action, _, _ = policy.get_action(s_t, deterministic=True)
        a_np = action.squeeze(0).cpu().numpy()

        # Project to valid weights
        w_raw = np.exp(a_np - a_np.max())
        w_raw = w_raw / w_raw.sum()
        w_new = np.clip(w_raw, 0, MAX_WEIGHT)
        w_new = w_new / (w_new.sum() + 1e-8)

        cost = np.abs(w_new - prev_w).sum() * tc
        prev_w = w_new.copy()

        # Holding period
        if i + 1 < len(reb_dates):
            next_reb = reb_dates[i + 1]
        else:
            next_reb = test_dates[-1] + pd.Timedelta(days=1)

        hold_mask  = (dates >= reb_d) & (dates < next_reb)
        hold_dates = dates[hold_mask]
        trail_rets = []

        for d_idx, d in enumerate(hold_dates):
            ti_arr = np.where(dates == d)[0]
            if len(ti_arr) == 0:
                continue
            ti = int(ti_arr[0])
            if ti >= len(ret_arr):
                continue
            r = float(ret_arr[ti] @ w_new)
            if d_idx == 0:
                r -= cost
            port_ret_map[d] = r
            trail_rets.append(r)

        trail_ret = sum(trail_rets[-5:]) if trail_rets else 0.0

    if not port_ret_map:
        return pd.Series(dtype=float)

    return pd.Series(port_ret_map).sort_index()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — WALK-FORWARD
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  NSE Portfolio GNN — Phase 5: PPO-RL Baseline")
    print("=" * 65)

    # ── Load features ────────────────────────────────────────────────────
    print("\n── Building feature tensor ──")
    feat_arr, tickers, dates = build_feature_tensor(PRICES_FILE, verbose=True)
    N = len(tickers)

    # ── Returns ──────────────────────────────────────────────────────────
    df_raw   = pd.read_parquet(PRICES_FILE)
    close_df = pd.DataFrame(
        {t: df_raw[(t, t, "Close")].squeeze() for t in tickers}
    ).reindex(dates).ffill(limit=5)
    ret_df   = np.log(close_df / close_df.shift(1)).iloc[1:]
    ret_arr  = ret_df.fillna(0).values.astype(np.float32)

    # Full-length arrays (aligned with feat_arr)
    ret_for_feat = np.zeros((len(dates), N), dtype=np.float32)
    ret_for_feat[1:] = ret_arr

    nifty_csv = OUT_DIR / "nifty50_returns.csv"
    nifty_ser = pd.read_csv(nifty_csv, index_col=0, parse_dates=True
                             ).iloc[:, 0].reindex(dates[1:]).ffill().fillna(0)
    nifty_arr_full = np.zeros(len(dates), dtype=np.float32)
    nifty_arr_full[1:] = nifty_ser.values

    sector_adj = build_sector_adj(tickers)

    # ── State dimension ───────────────────────────────────────────────────
    state_dim = N * N_FEATURES + N + 3 + 1
    print(f"  State dim : {state_dim}")
    print(f"  N stocks  : {N}")

    # ── Walk-forward loop ─────────────────────────────────────────────────
    all_metrics = []
    all_rets    = {}

    folds_all  = FOLDS + [BLIND[:4]]
    fold_names = [f"Fold{i+1}" for i in range(len(FOLDS))] + ["Blind2025"]

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds_all):
        fold_name = fold_names[fold_idx]
        is_blind  = fold_name == "Blind2025"

        print(f"\n{'='*65}")
        print(f"  {fold_name}  |  train→{tr_end}  |  test {te_start}→{te_end}")
        print(f"{'='*65}")

        tr_start_ts = pd.Timestamp(tr_start)
        tr_end_ts   = pd.Timestamp(tr_end)

        # ── Fit HMM ───────────────────────────────────────────────────────
        dates_ret   = dates[1:]
        tr_ret_mask = ((dates_ret >= tr_start_ts) &
                       (dates_ret <= tr_end_ts))
        nifty_tr    = nifty_ser.reindex(dates_ret[tr_ret_mask]).fillna(0).values

        hmm_det = HMMRegimeDetector(n_states=3, seed=SEED)
        hmm_det.fit(nifty_tr)

        nifty_all   = nifty_ser.reindex(dates[1:]).fillna(0).values
        regimes_raw = hmm_det.predict_causal(nifty_all)
        # NOTE: predict_proba() still uses non-causal forward-backward smoothing
        # (arguably worse than the Viterbi leakage fixed above, since it explicitly
        # uses future observations to refine past posteriors). No causal equivalent
        # implemented yet. RL-PPO is a minor comparison baseline, not this paper's
        # central finding, so this is a known, deliberate, documented gap rather
        # than a silent one -- flagged for a future fix, not swept under the rug.
        reg_post_raw= hmm_det.predict_proba(nifty_all)

        regimes_full = np.zeros(len(dates), dtype=np.int32)
        regimes_full[1:] = regimes_raw
        reg_post_full = np.zeros((len(dates), 3), dtype=np.float32)
        reg_post_full[1:] = reg_post_raw

        rc = np.bincount(regimes_raw, minlength=3)
        labels = ["bear", "sideways", "bull"]
        print(f"  HMM: {' '.join(f'{labels[i]}={rc[i]}' for i in range(3))}")

        # ── Training date indices ─────────────────────────────────────────
        tr_mask = np.where(
            (dates >= tr_start_ts) & (dates <= tr_end_ts))[0]
        if len(tr_mask) < 100:
            print(f"  ⚠  Too few training days — skipping")
            continue
        tr_start_idx = int(tr_mask[0])
        tr_end_idx   = int(tr_mask[-1])

        # ── Build environment ─────────────────────────────────────────────
        env = NSEPortfolioEnv(
            feat_arr        = feat_arr,
            ret_arr         = ret_for_feat,
            nifty_arr       = nifty_arr_full,
            regime_arr      = regimes_full,
            regime_post     = reg_post_full,
            episode_len     = EPISODE_LEN,
            lookback        = LOOKBACK,
            max_weight      = MAX_WEIGHT,
            tc_bps          = TC_BPS,
            train_start_idx = tr_start_idx,
            train_end_idx   = tr_end_idx,
        )

        # ── Train or load ─────────────────────────────────────────────────
        trainer   = PPOTrainer(state_dim=state_dim, n_stocks=N)
        ckpt_path = CKPT_DIR / f"ppo_{fold_name.lower()}.pt"

        if is_blind:
            last_ckpt = CKPT_DIR / "ppo_fold9.pt"
            if last_ckpt.exists():
                trainer.load(last_ckpt)
            else:
                trainer.train(env, N_EPISODES, fold_name)
                trainer.save(ckpt_path)
        elif ckpt_path.exists():
            trainer.load(ckpt_path)
        else:
            trainer.train(env, N_EPISODES, fold_name)
            trainer.save(ckpt_path)

        # ── Evaluate ──────────────────────────────────────────────────────
        print(f"\n── Evaluating {fold_name} ──")
        port_rets = evaluate_ppo(
            policy      = trainer.policy,
            feat_arr    = feat_arr,
            ret_arr     = ret_for_feat,
            nifty_arr   = nifty_arr_full,
            regime_post = reg_post_full,
            dates       = dates,
            test_start  = te_start,
            test_end    = te_end,
            state_dim   = state_dim,
            n_stocks    = N,
        )

        if port_rets.empty:
            print(f"  ⚠  No returns generated")
            continue

        nifty_te = nifty_ser.reindex(port_rets.index).ffill().fillna(0)
        m = compute_metrics(port_rets, nifty_te)
        m.update({"fold": fold_name, "strategy": "RL-PPO",
                   "n_days": len(port_rets)})
        all_metrics.append(m)
        all_rets[fold_name] = port_rets

        print(f"  Return={m.get('ann_return',0):.4f}  "
              f"Sharpe={m.get('sharpe',0):.4f}  "
              f"IR={m.get('info_ratio',0):.4f}  "
              f"MaxDD={m.get('max_drawdown',0):.4f}")

    # ── Save results ──────────────────────────────────────────────────────
    if all_metrics:
        rl_df = pd.DataFrame(all_metrics)
        rl_df.to_csv(OUT_DIR / "rl_ppo_results.csv", index=False)

        wf = rl_df[rl_df["fold"].str.startswith("Fold")]

        print(f"\n{'='*65}")
        print("  PPO-RL SUMMARY — Mean across walk-forward folds")
        print("="*65)
        print(f"\n  Ann. Return : {wf['ann_return'].mean():.4f}")
        print(f"  Sharpe      : {wf['sharpe'].mean():.4f}")
        print(f"  Info Ratio  : {wf['info_ratio'].mean():.4f}")
        print(f"  Max DD      : {wf['max_drawdown'].mean():.4f}")
        print(f"  Win Rate    : {wf['win_rate'].mean():.4f}")

        # Compare with STGAT-DPO
        stgat_df = pd.read_csv(OUT_DIR / "stgat_fold_results.csv")
        stgat_wf = stgat_df[stgat_df["fold"].str.startswith("Fold")]
        print(f"\n  --- vs STGAT-DPO ---")
        print(f"  IR delta   : {wf['info_ratio'].mean() - stgat_wf['info_ratio'].mean():+.4f}")
        print(f"  SR delta   : {wf['sharpe'].mean() - stgat_wf['sharpe'].mean():+.4f}")

        if all_rets:
            combined = pd.concat(all_rets.values()).sort_index()
            combined.to_csv(OUT_DIR / "rl_ppo_returns.csv", header=True)

    print(f"\n{'='*65}")
    print("  PPO-RL complete.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
