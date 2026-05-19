# pettingzoo_test.py — Deep RL Validation of the DERL Mechanism

## Overview

`pettingzoo_test.py` is a deep reinforcement learning validation of the Forager's Dilemma governance framework. It replicates the same five experimental conditions from the paper — using **Independent PPO (IPPO)** through a **PettingZoo `ParallelEnv`** interface — rather than the tabular Q-learning used in `main_v3_final.py`. The goal is to show that the DERL mechanism produces ethical emergence under a neural function approximator, not just under tabular methods.

---

## Architecture

### 1. Environment: `Env` (verbatim copy)

The `Env` class is copied unchanged from `main_v3_final.py`. It runs a 5×5 grid world with 4 agents and 8 resources over 50-step episodes. Agents choose from 10 discrete actions (move, gather, mine, signal-truth, signal-lie, punish, verify). Reward shaping depends on which governance flags are active (`use_hardcoded`, `use_emergent`, `use_intrinsic`, `coop_bonus`).

### 2. Observation Encoding: `decode_obs()`

The environment's internal `_sid()` method encodes each agent's local state as a 7-bit integer (values 0–127). `decode_obs()` unpacks these integers into **7-dimensional binary float vectors** for the neural network:

| Bit | Meaning |
|-----|---------|
| 0 | Resource within 1 step |
| 1 | Any peer within observation radius |
| 2 | Bad-acting peer nearby (mined or lied last step) |
| 3 | Any broadcast on the board |
| 4 | Resources depleted below 50% |
| 5 | Own reputation ≥ 2 (visible to peers) |
| 6 | Agent is in a cartel (AC condition only) |

### 3. PettingZoo Wrapper: `ForagerParallelEnv`

Wraps `Env` in the `ParallelEnv` interface. Each agent receives a `Box(0, 1, shape=(7,))` observation and acts in `Discrete(10)`. All agents step simultaneously each timestep — the wrapper passes a joint action dict to `Env.step()` and returns per-agent observation, reward, termination, and truncation dicts. This is compatible with any PettingZoo-aware RL library.

### 4. Policy Network: `ActorCritic`

One `ActorCritic` MLP per agent (4 total), mirroring the independent Q-tables in the tabular baseline:

```
Input (7) → Linear(64) → Tanh → Linear(64) → Tanh → trunk
trunk → Linear(10) → logits  [actor head]
trunk → Linear(1)  → value   [critic head]
```

Weights are initialized with orthogonal initialization. The actor head uses `std=0.01` for near-uniform initial action probabilities, encouraging exploration from the start.

### 5. Training: `train_ppo()` (IPPO)

Each agent trains independently using **Proximal Policy Optimization (PPO)** with **Generalized Advantage Estimation (GAE)**. Key hyperparameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `gamma` | 0.7 | Matches tabular Q-learning discount |
| `gae_lambda` | 0.95 | Standard GAE |
| `clip_coef` | 0.2 | Standard PPO clipping |
| `lr` | 2.5e-4 | Adam, standard for PPO |
| `ent_coef` | 0.01 | Encourages exploration of governance actions |
| `episodes_per_rollout` | 5 | ~250 environment steps per update |

Training loop: collect one episode into per-agent buffers → every 5 episodes, compute GAE advantages and run 4 epochs of minibatch PPO updates for each agent independently → clear buffers. The function returns a metric history dict with the same 13 keys as the tabular `train()` in `main_v3_final.py`, enabling direct comparison.

---

## The Five Conditions

`run_all()` trains all five conditions from the paper with 5 random seeds (42–46). Each uses the same `Env` class — only the constructor flags differ:

| Condition | Code | What changes |
|-----------|------|-------------|
| **SRB** — Scalar Reward Baseline | `use_emergent=False` | No social mechanisms. Agents learn from task rewards only (gather/mine). Benchmark for unregulated behavior. |
| **ES** — Exogenous Sanctions | `use_hardcoded=True` | Penalties for mining and lying are hardcoded into the reward function by a designer. Tests whether external rules alone are sufficient. |
| **DPF** — Decentralised Peer Feedback | `use_emergent=True` | Agents can punish bad actors and verify false broadcasts, earning rewards for doing so. No intrinsic shaping — only emergent social pressure. |
| **DERL** — **(Ours)** | `use_emergent=True, use_intrinsic=True` | DPF + intrinsic shaping: truth-telling rewarded, lying penalized at the reward level. Tests whether the full DERL mechanism produces stable epistemic norms. |
| **AC** — Adversarial Coalition | `cartel=[0,1], cartel_share=0.3` | Agents 0 and 1 pool 30% of their task rewards, creating incentive to shield each other from punishment. Tests robustness of the DPF mechanism to collusion. |

---

## What the Script Tests

The core claim of the paper is that **emergent social mechanisms (punishment + verification)** combined with **intrinsic epistemic shaping** (DERL) produce self-sustaining ethical behavior without designer-specified rules. This script validates that claim under a neural approximator:

- **DERL vs. SRB**: DERL should converge to higher truth rate and gather rate, lower lie rate and mine rate — agents learn ethical norms from social feedback, not just task rewards.
- **DERL vs. ES**: Both should achieve ethical behavior, but ES relies on hardcoded penalties; DERL achieves the same through emergent social pressure, validating mechanism-level generality.
- **DPF vs. DERL**: DERL should be more stable across seeds (lower variance) because intrinsic shaping prevents oscillation between truth-telling and lying equilibria.
- **DPF vs. AC**: The AC condition tests whether two colluding agents can undermine the punishment mechanism. DPF should degrade under AC; the plots quantify by how much.
- **Cumulative reward**: DERL should surpass SRB by around episode 15,000 — the crossover point where cooperative norms become load-bearing for collective reward.

---

## Outputs

Per-seed checkpoints are saved to `checkpoints/` as `.npz` files (e.g., `DERL_seed42.npz`). If a checkpoint exists it is loaded rather than recomputed — runs are crash-safe and resumable.

Nine figures are written to `plots/v4_ppo/` (PDF + PNG):

| File | Content |
|------|---------|
| `fig01_epistemic` | Truth rate over episodes |
| `fig02_ethical` | Gather rate over episodes |
| `fig03_hallucination` | Lie rate over episodes |
| `fig04_moral_drift` | Mine rate over episodes |
| `fig05_social` | Punishment rate, verification rate, mean reputation (DPF vs DERL) |
| `fig06_coop_oracle` | Cooperation events and oracle accuracy |
| `fig07_reward` | Cumulative reward |
| `fig08_resources` | Active resource count |
| `fig09_collusion` | DPF vs AC: behavioral rates, cooperation, oracle accuracy |

All curves are smoothed with a Hanning window and shaded with ±0.6σ across seeds.

---

## Usage

```bash
# Full run — 50 000 episodes × 5 seeds × 5 conditions (~8–12 h on CPU)
conda run -n forager python pettingzoo_test.py

# Resume from existing checkpoints (skips completed seeds automatically)
conda run -n forager python pettingzoo_test.py --episodes 50000

# GPU acceleration
conda run -n forager python pettingzoo_test.py --device cuda

# Shorter smoke test
conda run -n forager python pettingzoo_test.py --episodes 500 --seeds 42

# Generate plots from saved checkpoints only
conda run -n forager python plot_from_checkpoints.py --checkpoint-dir checkpoints --outdir plots/v4_ppo
```

**Dependencies:** `torch`, `pettingzoo`, `gymnasium`, `numpy`, `matplotlib`
