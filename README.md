# Forager's Dilemma

Multi-agent reinforcement learning environment coupling epistemic (hallucination) and ethical (moral drift) failure modes within a shared resource economy. Evaluates four governance regimes: Scalar Reward Baseline (SRB), Exogenous Sanctions (ES), Decentralised Peer Feedback (DPF), and Dual-Enforcement Reinforcement Learning (DERL).

## Implementations

| File | Algorithm | Output |
|---|---|---|
| `main_v3_final.py` | Tabular Q-learning (128 states) | `plots/v3/` |
| `forager_ppo.py` | Independent PPO + PettingZoo | `plots/v4_ppo/` |

---

## Dependencies

### Tabular Q-learning (`main_v3_final.py`)

Only standard scientific Python is required:

```bash
pip install numpy matplotlib
```

### PPO expansion (`forager_ppo.py`)

```bash
pip install torch pettingzoo gymnasium matplotlib
```

For GPU acceleration, install the CUDA-enabled build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) matching your CUDA version. The CPU build works too but is slower for large episode counts.

#### Conda environment (recommended)

```bash
conda create -n forager python=3.11
conda activate forager
pip install torch pettingzoo gymnasium matplotlib
```

---

## Running Experiments

### Tabular Q-learning

```bash
python main_v3_final.py
```

Runs 50,000 episodes × 5 seeds across all 5 conditions. Saves 10 figures to `plots/v3/`. Expected runtime: ~10–20 minutes on a modern CPU.

### PPO (deep RL)

**Full experiment (matches paper episode count):**

```bash
# CPU
python forager_ppo.py --episodes 50000

# GPU (faster)
python forager_ppo.py --episodes 50000 --device cuda
```

**With conda environment:**

```bash
conda run -n forager python forager_ppo.py --episodes 50000 --device cuda
```

**Quick test run:**

```bash
python forager_ppo.py --episodes 500
```

**All options:**

```
--episodes   Number of training episodes per condition per seed  (default: 50000)
--seeds      Random seeds to average over                        (default: 42 43 44 45 46)
--device     PyTorch device: cpu or cuda                         (default: cpu)
--outdir     Directory to save figures                           (default: plots/v4_ppo)
```

Progress is printed every 1,000 episodes (for a 50k run) as:

```
ep   1000/50000  R=  342.1  T=0.120 L=0.210 G=0.310 M=0.180 ...
```

Figures are saved to `plots/v4_ppo/` on completion.

---

## Output Figures

Both scripts produce the same 10 figures:

| Figure | Metric |
|---|---|
| `fig01_epistemic` | Truth rate over training |
| `fig02_ethical` | Gather rate over training |
| `fig03_hallucination` | Lie rate over training |
| `fig04_moral_drift` | Mine rate over training |
| `fig05_social` | Punishment rate, verification rate, mean reputation |
| `fig06_coop_oracle` | Cooperation events and oracle accuracy |
| `fig07_reward` | Cumulative reward |
| `fig08_resources` | Active resources (commons sustainability) |
| `fig09_ablation_punish` | Punishment profitability ablation |
| `fig10_collusion` | Adversarial coalition robustness |

Each figure is saved as both PNG and PDF at 300 dpi.
