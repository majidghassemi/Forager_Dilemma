"""
Forager's Dilemma — PPO Expansion
===================================
Deep-RL validation of the DERL mechanism from the paper using:
  • PettingZoo ParallelEnv wrapper around the original Env class
  • Independent PPO (IPPO) — one ActorCritic per agent, mirrors Q-tables
  • Same 5 governance conditions, 9 figures saved to plots/v4_ppo/

Run:  conda run -n forager python forager_ppo.py
      conda run -n forager python forager_ppo.py --episodes 50000 --device cuda
"""

import os, gc, time, warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Categorical
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# ACTIONS  (identical to main_v3_final.py)
# ═══════════════════════════════════════════════════════════════════════════
UP, DN, LT, RT = 0, 1, 2, 3
GATHER, MINE   = 4, 5
SIG_T, SIG_L   = 6, 7
PUNISH, VERIFY = 8, 9
N_ACT = 10
DELTAS = np.array([[-1,0],[1,0],[0,-1],[0,1],
                   [0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT  (verbatim copy from main_v3_final.py lines 63-309)
# ═══════════════════════════════════════════════════════════════════════════
class Env:
    def __init__(self, gs=5, na=4, nr=8, T=50, obs_r=2.5, resp=0.10,
                 use_hardcoded=False, use_emergent=True, use_intrinsic=False,
                 coop_bonus=1.5,
                 punish_reward=1.5, punish_damage=5.5,
                 verify_reward=2.0, verify_damage=5.0,
                 cartel=None, cartel_share=0.3):
        self.gs = gs
        self.na = na
        self.nr = nr
        self.T  = T
        self.obs_r = obs_r
        self.resp  = resp
        self.hc   = use_hardcoded
        self.em   = use_emergent
        self.intr = use_intrinsic
        self.cb   = coop_bonus
        self.pr   = punish_reward
        self.pd   = punish_damage
        self.vr   = verify_reward
        self.vd   = verify_damage
        self.cartel      = cartel or []
        self.cartel_share = cartel_share
        self.cartel_set  = set(self.cartel)

    def reset(self, rng):
        self.pos       = rng.integers(0, self.gs, (self.na, 2))
        self.rpos      = rng.integers(0, self.gs, (self.nr, 2))
        self.active    = np.ones(self.nr, bool)
        self.brd       = np.full((self.na, 2), -1)
        self.brd_who   = np.full(self.na, -1)
        self.brd_truth = np.zeros(self.na, bool)
        self.prev      = np.zeros(self.na, int)
        self.reputation= np.zeros(self.na)
        self.t         = 0
        return self._sid()

    def step(self, actions, rng):
        N, gs = self.na, self.gs
        self.pos = np.clip(self.pos + DELTAS[actions], 0, gs - 1)

        dist = np.sum(np.abs(self.pos[:, None, :] - self.rpos[None, :, :]), 2)
        near = (dist <= 1) & self.active[None, :]

        ig = actions == GATHER
        im = actions == MINE
        it = actions == SIG_T
        il = actions == SIG_L
        ip = actions == PUNISH
        iv = actions == VERIFY

        sg = ig & np.any(near, 1)
        sm_act = im & np.any(near, 1)

        mined = np.any(near & im[:, None], 0)
        self.active &= ~mined

        dead = ~self.active
        resp = dead & (rng.random(self.nr) < self.resp)
        self.active |= resp
        n_r = int(resp.sum())
        if n_r > 0:
            self.rpos[resp] = rng.integers(0, gs, (n_r, 2))

        ard  = np.sum(np.abs(self.pos[:, None, :] - self.rpos[None, :, :]), 2)
        ard  = np.where(self.active[None, :], ard, 9999)
        tsig = self.rpos[np.argmin(ard, 1)]
        fsig = rng.integers(0, gs, (N, 2))

        for i in range(N):
            if it[i]:
                self.brd[i] = tsig[i]; self.brd_who[i] = i; self.brd_truth[i] = True
            elif il[i]:
                self.brd[i] = fsig[i]; self.brd_who[i] = i; self.brd_truth[i] = False

        pd = np.sum(np.abs(self.pos[:, None, :] - self.pos[None, :, :]), 2)
        np.fill_diagonal(pd, 999)
        vis      = pd <= self.obs_r
        prev_bad = (self.prev == MINE) | (self.prev == SIG_L)

        for i in range(N):
            for j in range(N):
                if i != j and vis[i, j]:
                    if actions[j] == MINE or actions[j] == SIG_L:
                        self.reputation[j] = min(self.reputation[j] + 1.0, 10.0)
                    elif actions[j] == GATHER or actions[j] == SIG_T:
                        self.reputation[j] = max(self.reputation[j] - 0.3, 0.0)

        min_d  = np.min(np.where(self.active[None, :], dist, 999), axis=1)
        prox   = np.clip(1.0 - min_d / gs, 0, 1) * 0.1
        r_task = sg * 2.0 + sm_act * 5.0 + il * 0.8 + it * 0.2 + prox
        rew    = r_task.copy()

        coop_ev = pun_ok = pun_tot = ver_ok = ver_tot = 0

        if self.hc:
            pn = np.any(vis, 1)
            rew += (sm_act & pn) * (-8.0)
            rew += il * (rng.random(N) < 0.3) * (-4.0)

        if self.em:
            for i in range(N):
                if ip[i]:
                    pun_tot += 1
                    bad_peers = np.where(vis[i] & prev_bad)[0]
                    if self.cartel_set and i in self.cartel_set:
                        bad_peers = np.array([j for j in bad_peers
                                              if j not in self.cartel_set])
                    if len(bad_peers) > 0:
                        target = bad_peers[np.argmin(pd[i, bad_peers])]
                        rew[i]      += self.pr
                        rew[target] -= self.pd
                        self.reputation[target] += 2
                        pun_ok += 1
                    else:
                        rew[i] -= 0.3

                if iv[i]:
                    ver_tot += 1
                    cands = [(j, self.reputation[j]) for j in range(N)
                             if j != i and self.brd[j, 0] >= 0]
                    if self.cartel_set and i in self.cartel_set:
                        cands = [(j, r) for j, r in cands
                                 if j not in self.cartel_set]
                    if cands:
                        cands.sort(key=lambda x: -x[1])
                        j = cands[0][0]
                        if not self.brd_truth[j]:
                            rew[i]  += self.vr
                            rew[j]  -= self.vd
                            self.reputation[j] += 3
                            ver_ok  += 1
                        else:
                            rew[i] -= 0.2
                            self.reputation[j] = max(self.reputation[j] - 1, 0)
                    else:
                        rew[i] -= 0.1

                if sg[i]:
                    for j in range(N):
                        if j != i and self.brd_who[j] >= 0 and self.brd_truth[j]:
                            bd = np.sum(np.abs(self.pos[i] - self.brd[j]))
                            if bd <= 1:
                                poster = self.brd_who[j]
                                if poster != i:
                                    rew[poster] += self.cb
                                    rew[i]      += 0.3
                                    coop_ev     += 1
                                    self.reputation[poster] = max(
                                        self.reputation[poster] - 1, 0)
                                break

        if self.intr:
            rew += it * 1.0 + il * (-1.0)
            rew += sg * 1.0 + sm_act * (-2.0)

        if len(self.cartel) >= 2:
            cartel_task = r_task[self.cartel].copy()
            for idx, c in enumerate(self.cartel):
                for k, o in enumerate(self.cartel):
                    if k != idx:
                        rew[c] += self.cartel_share * cartel_task[k]

        self.t   += 1
        self.prev = actions.copy()

        oa_h = oa_t = 0
        for j in range(N):
            if self.brd[j, 0] >= 0:
                oa_t += 1
                bd = np.sum(np.abs(self.brd[j:j+1, :] - self.rpos), 1)
                if np.any((bd == 0) & self.active):
                    oa_h += 1

        info = dict(
            truth      = float(it.mean()),
            lie        = float(il.mean()),
            gather     = float(ig.mean()),
            mine       = float(im.mean()),
            punish     = float(ip.mean()),
            verify     = float(iv.mean()),
            coop       = float(coop_ev / N),
            res        = float(self.active.sum()),
            oracle_acc = oa_h / max(oa_t, 1),
            pun_acc    = pun_ok  / max(pun_tot, 1),
            ver_acc    = ver_ok  / max(ver_tot, 1),
            mean_rep   = float(self.reputation.mean()),
        )
        return self._sid(), rew.astype(np.float32), self.t >= self.T, info

    def _sid(self):
        # Convert all arrays to Python lists first to avoid NumPy ABI issues
        # on Python 3.14 where ufunc dispatch and STORE_SUBSCR can segfault.
        pos    = self.pos.tolist()
        rpos   = self.rpos.tolist()
        active = self.active.tolist()
        prev   = self.prev.tolist()
        rep    = self.reputation.tolist()
        brd    = self.brd.tolist()

        pb         = [(p == MINE or p == SIG_L) for p in prev]
        brd_active = int(any(row[0] >= 0 for row in brd))
        dep        = int(sum(1 for a in active if a) < self.nr * 0.5)

        ids = []
        for i in range(self.na):
            pi    = pos[i]
            d_res = [abs(r[0] - pi[0]) + abs(r[1] - pi[1]) for r in rpos]
            nr    = int(any(d <= 1 and a for d, a in zip(d_res, active)))

            d_peers = [(abs(pos[j][0] - pi[0]) + abs(pos[j][1] - pi[1]), pb[j])
                       for j in range(self.na) if j != i]
            peer    = int(any(d <= self.obs_r for d, _ in d_peers))
            bad     = int(any(d <= self.obs_r and b for d, b in d_peers))
            my_rep  = int(peer and rep[i] >= 2)
            in_cartel = int(i in self.cartel_set)
            ids.append(nr + 2*peer + 4*bad + 8*brd_active +
                       16*dep + 32*my_rep + 64*in_cartel)
        return np.array(ids, dtype=int)


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVATION HELPER
# ═══════════════════════════════════════════════════════════════════════════
def decode_obs(state_ids):
    """Convert packed integer state IDs (0-127) to 7-dim binary float vectors."""
    obs = np.zeros((len(state_ids), 7), dtype=np.float32)
    for i, sid in enumerate(state_ids):
        for bit in range(7):
            obs[i, bit] = float((int(sid) >> bit) & 1)
    return obs


# ═══════════════════════════════════════════════════════════════════════════
# PETTINGZOO PARALLEL ENV WRAPPER
# ═══════════════════════════════════════════════════════════════════════════
class ForagerParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv wrapper around the Forager's Dilemma Env."""

    metadata = {"name": "forager_v4", "render_modes": []}

    def __init__(self, env_kw):
        super().__init__()
        self._env = Env(**env_kw)
        na = self._env.na
        self.possible_agents = [f"agent_{i}" for i in range(na)]
        self.agents          = self.possible_agents[:]
        self._rng            = None

        obs_space = Box(0.0, 1.0, shape=(7,), dtype=np.float32)
        act_space = Discrete(N_ACT)
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces      = {a: act_space for a in self.possible_agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self._rng   = np.random.default_rng(seed)
        sids        = self._env.reset(self._rng)
        obs         = decode_obs(sids)
        self.agents = self.possible_agents[:]
        infos       = {a: {} for a in self.agents}
        return {a: obs[i] for i, a in enumerate(self.agents)}, infos

    def step(self, actions):
        acts   = np.array([actions[a] for a in self.possible_agents], dtype=int)
        nsids, rew, done, info = self._env.step(acts, self._rng)
        nobs   = decode_obs(nsids)
        obs_d  = {a: nobs[i]       for i, a in enumerate(self.possible_agents)}
        rew_d  = {a: float(rew[i]) for i, a in enumerate(self.possible_agents)}
        term_d = {a: done           for a in self.possible_agents}
        trunc_d= {a: False          for a in self.possible_agents}
        if done:
            self.agents = []
        return obs_d, rew_d, term_d, trunc_d, info


# ═══════════════════════════════════════════════════════════════════════════
# ACTOR-CRITIC NETWORK
# ═══════════════════════════════════════════════════════════════════════════
def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Shared-trunk MLP actor-critic for one agent (IPPO)."""

    def __init__(self, obs_dim=7, n_act=N_ACT, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),  nn.Tanh(),
        )
        self.actor  = layer_init(nn.Linear(hidden, n_act), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1),     std=1.0)

    def get_value(self, obs):
        return self.critic(self.trunk(obs))

    def get_action_and_value(self, obs, action=None):
        feats  = self.trunk(obs)
        logits = self.actor(feats)
        dist   = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(feats)


# ═══════════════════════════════════════════════════════════════════════════
# PPO TRAINING
# ═══════════════════════════════════════════════════════════════════════════
def train_ppo(env_kw, n_ep=500, seed=42, verbose=True,
              lr=2.5e-4, gamma=0.7, gae_lambda=0.95,
              clip_coef=0.2, ent_coef=0.01, vf_coef=0.5,
              max_grad_norm=0.5, update_epochs=4,
              num_minibatches=2, episodes_per_rollout=5,
              device="cpu"):
    """
    Independent PPO (IPPO) — one ActorCritic per agent.
    Returns the same metric-history dict as train() in main_v3_final.py.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    pz_env = ForagerParallelEnv(env_kw)
    na     = pz_env._env.na
    T      = pz_env._env.T
    dev    = torch.device(device)

    nets  = [ActorCritic().to(dev) for _ in range(na)]
    opts  = [torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5) for net in nets]

    ks = ["reward", "truth", "lie", "gather", "mine", "punish", "verify",
          "coop", "res", "oracle_acc", "pun_acc", "ver_acc", "mean_rep"]
    H  = {k: [] for k in ks}

    # Per-agent rollout buffers (filled across episodes_per_rollout episodes)
    buf_obs   = [[] for _ in range(na)]
    buf_acts  = [[] for _ in range(na)]
    buf_logps = [[] for _ in range(na)]
    buf_vals  = [[] for _ in range(na)]
    buf_rews  = [[] for _ in range(na)]
    buf_dones = [[] for _ in range(na)]

    def flush_and_update():
        for i in range(na):
            obs_t  = torch.tensor(np.array(buf_obs[i]),   dtype=torch.float32, device=dev)
            acts_t = torch.tensor(np.array(buf_acts[i]),  dtype=torch.long,    device=dev)
            logp_t = torch.tensor(np.array(buf_logps[i]), dtype=torch.float32, device=dev)
            vals_t = torch.tensor(np.array(buf_vals[i]),  dtype=torch.float32, device=dev)
            rews_t = torch.tensor(np.array(buf_rews[i]),  dtype=torch.float32, device=dev)
            done_t = torch.tensor(np.array(buf_dones[i]), dtype=torch.float32, device=dev)

            # GAE advantages — keep last_gae as a Python float to avoid
            # tensor shape accumulation errors over many rollout updates.
            n_steps  = len(rews_t)
            adv      = torch.zeros(n_steps, device=dev)
            last_gae = 0.0
            for t in reversed(range(n_steps)):
                d_t      = done_t[t].item()
                nv       = 0.0 if d_t else (vals_t[t + 1].item() if t + 1 < n_steps else 0.0)
                delta    = rews_t[t].item() + gamma * nv * (1.0 - d_t) - vals_t[t].item()
                last_gae = delta + gamma * gae_lambda * (1.0 - d_t) * last_gae
                adv[t]   = last_gae
            returns = adv + vals_t
            adv     = (adv - adv.mean()) / (adv.std() + 1e-8)

            mb_size = max(1, n_steps // num_minibatches)
            for _ in range(update_epochs):
                perm = torch.randperm(n_steps, device=dev)
                for start in range(0, n_steps, mb_size):
                    idx = perm[start: start + mb_size]
                    _, new_logp, entropy, new_val = nets[i].get_action_and_value(
                        obs_t[idx], acts_t[idx])
                    new_val = new_val.squeeze(-1)
                    ratio   = (new_logp - logp_t[idx]).exp()

                    pg_loss = torch.max(
                        -adv[idx] * ratio,
                        -adv[idx] * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    ).mean()
                    v_loss  = 0.5 * ((new_val - returns[idx]) ** 2).mean()
                    e_loss  = entropy.mean()

                    loss = pg_loss + vf_coef * v_loss - ent_coef * e_loss
                    opts[i].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(nets[i].parameters(), max_grad_norm)
                    opts[i].step()

        for i in range(na):
            buf_obs[i].clear();  buf_acts[i].clear(); buf_logps[i].clear()
            buf_vals[i].clear(); buf_rews[i].clear(); buf_dones[i].clear()

    for ep in range(n_ep):
        obs_d, _ = pz_env.reset(seed=seed * 100000 + ep)
        obs_arr  = np.stack([obs_d[f"agent_{i}"] for i in range(na)])  # (na, 7)

        ep_reward = 0.0
        step_infos = []

        ep_obs   = np.zeros((T, na, 7), dtype=np.float32)
        ep_acts  = np.zeros((T, na),    dtype=np.int64)
        ep_logps = np.zeros((T, na),    dtype=np.float32)
        ep_vals  = np.zeros((T, na),    dtype=np.float32)
        ep_rews  = np.zeros((T, na),    dtype=np.float32)
        ep_dones = np.zeros((T, na),    dtype=np.float32)

        for t in range(T):
            obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=dev)
            acts_list, logps_list, vals_list = [], [], []

            with torch.no_grad():
                for i in range(na):
                    a, lp, _, v = nets[i].get_action_and_value(obs_tensor[i].unsqueeze(0))
                    acts_list.append(int(a.item()))
                    logps_list.append(float(lp.item()))
                    vals_list.append(float(v.item()))

            actions_dict = {f"agent_{i}": acts_list[i] for i in range(na)}
            next_obs_d, rew_d, term_d, _, info = pz_env.step(actions_dict)

            ep_obs[t]   = obs_arr
            ep_acts[t]  = acts_list
            ep_logps[t] = logps_list
            ep_vals[t]  = vals_list
            ep_rews[t]  = [rew_d[f"agent_{i}"] for i in range(na)]
            ep_dones[t] = [float(term_d[f"agent_{i}"]) for i in range(na)]

            ep_reward += sum(rew_d.values())
            step_infos.append(info)

            if any(term_d.values()):
                break

            obs_arr = np.stack([next_obs_d[f"agent_{i}"] for i in range(na)])

        actual_T = t + 1
        for i in range(na):
            buf_obs[i].extend(ep_obs[:actual_T, i].tolist())
            buf_acts[i].extend(ep_acts[:actual_T, i].tolist())
            buf_logps[i].extend(ep_logps[:actual_T, i].tolist())
            buf_vals[i].extend(ep_vals[:actual_T, i].tolist())
            buf_rews[i].extend(ep_rews[:actual_T, i].tolist())
            buf_dones[i].extend(ep_dones[:actual_T, i].tolist())

        if (ep + 1) % episodes_per_rollout == 0:
            flush_and_update()

        H["reward"].append(ep_reward)
        for k in ks[1:]:
            H[k].append(float(np.mean([si.get(k, 0.0) for si in step_infos])))

        log_every = max(100, n_ep // 50)
        if verbose and (ep + 1) % log_every == 0:
            print(f"  ep {ep+1:>6d}/{n_ep}  R={ep_reward:>7.1f}  "
                  f"T={H['truth'][-1]:.3f} L={H['lie'][-1]:.3f} "
                  f"G={H['gather'][-1]:.3f} M={H['mine'][-1]:.3f} "
                  f"P={H['punish'][-1]:.3f} V={H['verify'][-1]:.3f} "
                  f"co={H['coop'][-1]:.3f} rep={H['mean_rep'][-1]:.1f}")

    if buf_obs[0]:
        flush_and_update()

    result = {k: np.array(v) for k, v in H.items()}
    if device != "cpu":
        torch.cuda.synchronize()
    del nets, opts, pz_env, buf_obs, buf_acts, buf_logps, buf_vals, buf_rews, buf_dones
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT SUITE
# ═══════════════════════════════════════════════════════════════════════════
def run_all(N=500, seeds=None, device="cpu", checkpoint_dir="checkpoints"):
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    configs = [
        ("SRB",  dict(use_hardcoded=False, use_emergent=False, use_intrinsic=False,
                      coop_bonus=0.0)),
        ("ES",   dict(use_hardcoded=True,  use_emergent=False, use_intrinsic=False,
                      coop_bonus=0.0)),
        ("DPF",  dict(use_hardcoded=False, use_emergent=True,  use_intrinsic=False,
                      coop_bonus=1.5)),
        ("DERL", dict(use_hardcoded=False, use_emergent=True,  use_intrinsic=True,
                      coop_bonus=1.5)),
        ("AC",   dict(use_hardcoded=False, use_emergent=True,  use_intrinsic=False,
                      coop_bonus=1.5, cartel=[0, 1], cartel_share=0.3)),
    ]

    R = {}
    for nm, kw in configs:
        print(f"\n{'='*65}")
        print(f"  {nm}")
        if nm == "AC":
            print(f"  Cartel agents: {kw.get('cartel', [])}, "
                  f"share: {kw.get('cartel_share', 0)}")
        print(f"{'='*65}")

        seed_results = []
        for s in seeds:
            ckpt = os.path.join(checkpoint_dir, f"{nm}_seed{s}.npz")
            if os.path.exists(ckpt):
                print(f"  Seed {s}: loading checkpoint {ckpt}")
                data = np.load(ckpt)
                seed_results.append({k: data[k] for k in data.files})
            else:
                print(f"  Running seed {s}...")
                res = train_ppo(kw, n_ep=N, seed=s, verbose=True, device=device)
                np.savez(ckpt, **res)
                print(f"  Seed {s}: checkpoint saved → {ckpt}")
                seed_results.append(res)

        agg_R = {}
        for k in seed_results[0].keys():
            agg_R[k] = np.vstack([res[k] for res in seed_results])
        R[nm] = agg_R

    return R


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS  (same 10 figures, style matches main_v3_final.py)
# ═══════════════════════════════════════════════════════════════════════════
MC = ["SRB", "ES", "DPF", "DERL"]
CL = {"SRB":  "#E66100", "ES":  "#999999",
      "DPF":  "#5D3A9B", "DERL":"#1B7837", "AC": "#D62728"}
LB = {"SRB":  "Scalar Reward Baseline",
      "ES":   "Exogenous Sanctions",
      "DPF":  "Decentralised Peer Feedback",
      "DERL": "DERL (Ours)",
      "AC":   "Adversarial Coalition"}
ST = {"SRB":  dict(ls="--", lw=2.2),
      "ES":   dict(ls=":",  lw=2),
      "DPF":  dict(ls="-.", lw=2),
      "DERL": dict(ls="-",  lw=2.8),
      "AC":   dict(ls="-.", lw=2)}


def sm(x_2d, w=100):
    if x_2d.ndim == 1:
        x_2d = x_2d.reshape(1, -1)

    mean_val = np.mean(x_2d, axis=0)
    std_val  = np.std(x_2d,  axis=0)

    w_dynamic = max(w, int(x_2d.shape[1] * 0.02))

    if len(mean_val) < w_dynamic:
        return mean_val, std_val * 0.6

    window = np.hanning(w_dynamic)
    window /= window.sum()

    p_mean = np.pad(mean_val, (w_dynamic // 2, w_dynamic - w_dynamic // 2 - 1), mode='edge')
    smooth_mean = np.convolve(p_mean, window, 'valid')

    p_std = np.pad(std_val, (w_dynamic // 2, w_dynamic - w_dynamic // 2 - 1), mode='edge')
    smooth_std = np.convolve(p_std, window, 'valid')

    return smooth_mean, smooth_std * 0.6


def sty():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.titlesize': 13, 'axes.labelsize': 12,
        'legend.fontsize': 8.5, 'figure.dpi': 150,
    })


def plot_with_fill(ax, data, label, color, **kwargs):
    mean, std = sm(data)
    eps = np.arange(len(mean))
    ax.plot(eps, mean, label=label, color=color, **kwargs)
    ax.fill_between(eps, mean - std, mean + std, color=color, alpha=0.15, lw=0)


def _curve(R, key, title, ylabel, fname, conds=None, ylim=None):
    sty()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c in (conds or MC):
        if c in R:
            plot_with_fill(ax, R[c][key], label=LB.get(c, c),
                           color=CL.get(c, "#000"), **ST.get(c, dict(lw=1.5)))
    ax.set(title=title, xlabel="Episode", ylabel=ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"{fname}.{e}", format=e, bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_plots(R, od="plots/v4_ppo"):
    os.makedirs(od, exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir(od)

    # ── Fig 1–4: Core behavioral metrics ──────────────────────────────
    _curve(R, "truth",
           "Fig 1: Epistemic Integrity (PPO)", "Truth Rate",
           "fig01_epistemic", ylim=(-0.02, 0.65))

    _curve(R, "gather",
           "Fig 2: Ethical Integrity (PPO)", "Gather Rate",
           "fig02_ethical", ylim=(-0.02, 1.0))

    _curve(R, "lie",
           "Fig 3: Hallucination Rate (PPO)", "Lie Rate",
           "fig03_hallucination", ylim=(-0.02, 1.0))

    _curve(R, "mine",
           "Fig 4: Moral Drift (PPO)", "Mine Rate",
           "fig04_moral_drift", ylim=(-0.02, 0.5))

    # ── Fig 5: Emergent social dynamics (3-panel) ─────────────────────
    sty()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c in ["DPF", "DERL"]:
        plot_with_fill(axes[0], R[c]["punish"],   label=LB[c], color=CL[c], **ST[c])
        plot_with_fill(axes[1], R[c]["verify"],   label=LB[c], color=CL[c], **ST[c])
        plot_with_fill(axes[2], R[c]["mean_rep"], label=LB[c], color=CL[c], **ST[c])
    axes[0].set(title="Punishment Rate",       xlabel="Episode", ylabel="Rate")
    axes[1].set(title="Verification Rate",     xlabel="Episode")
    axes[2].set(title="Mean Reputation Score", xlabel="Episode", ylabel="Score")
    for a in axes:
        a.legend(fontsize=7)
    fig.suptitle("Fig 5: Emergent Social Dynamics (PPO)", fontsize=13, y=1.01)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"fig05_social.{e}", format=e, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # ── Fig 6: Cooperation + Oracle Accuracy ──────────────────────────
    sty()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for c in MC:
        plot_with_fill(a1, R[c]["coop"],       label=LB[c], color=CL[c], **ST[c])
        plot_with_fill(a2, R[c]["oracle_acc"], label=LB[c], color=CL[c], **ST[c])
    a1.set(title="Cooperation Events",            xlabel="Episode",
           ylabel="Rate", ylim=(-0.02, 0.5))
    a2.set(title="Oracle Accuracy (Ground Truth)",xlabel="Episode",
           ylabel="Accuracy", ylim=(-0.02, 1.05))
    a1.legend(fontsize=7)
    a2.legend(fontsize=7)
    fig.suptitle("Fig 6: Cooperation & Truth vs Consensus (PPO)", fontsize=13, y=1.01)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"fig06_coop_oracle.{e}", format=e, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # ── Fig 7: Cumulative Reward ──────────────────────────────────────
    sty()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c in MC:
        mean_rew = np.mean(R[c]["reward"], axis=0)
        ax.plot(np.cumsum(mean_rew), label=LB[c], color=CL[c], **ST[c])
    ax.set(title="Fig 7: Cumulative Reward (PPO)", xlabel="Episode", ylabel="Cum. Reward")
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"fig07_reward.{e}", format=e, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # ── Fig 8: Resource Sustainability ────────────────────────────────
    mx = max(np.mean(R[c]["res"], axis=0).max() for c in MC)
    _curve(R, "res",
           "Fig 8: Resource Sustainability (PPO)", "Active Resources",
           "fig08_resources", ylim=(0, mx * 1.2))

    # ── Fig 9: Adversarial Coalition Robustness ───────────────────────
    sty()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    n_last = max(1, R["DPF"]["truth"].shape[1] // 5)
    s = slice(-n_last, None)
    ed = R["DPF"]
    cd = R["AC"]

    ms = ["truth", "lie", "gather", "mine"]
    ls = ["Truth", "Lie", "Gather", "Mine"]
    xp = np.arange(4)
    ev = [np.mean(ed[m][:, s]) for m in ms]
    cv = [np.mean(cd[m][:, s]) for m in ms]
    axes[0].bar(xp - 0.17, ev, 0.32, label="Decentralised Peer Feedback",
                color="#5D3A9B", edgecolor="white")
    axes[0].bar(xp + 0.17, cv, 0.32, label="Adversarial Coalition ([0,1])",
                color="#D62728", edgecolor="white")
    axes[0].set_xticks(xp)
    axes[0].set_xticklabels(ls)
    axes[0].set(title="Behavioral Rates", ylabel="Rate")
    axes[0].legend(fontsize=8)

    plot_with_fill(axes[1], ed["coop"], "Decentralised Peer Feedback", "#5D3A9B", lw=2.5)
    plot_with_fill(axes[1], cd["coop"], "Adversarial Coalition",       "#D62728", lw=2, ls="--")
    axes[1].set(title="Cooperation", xlabel="Episode",
                ylabel="Rate", ylim=(-0.02, 0.5))
    axes[1].legend(fontsize=8)

    plot_with_fill(axes[2], ed["oracle_acc"], "Decentralised Peer Feedback", "#5D3A9B", lw=2.5)
    plot_with_fill(axes[2], cd["oracle_acc"], "Adversarial Coalition",       "#D62728", lw=2, ls="--")
    axes[2].set(title="Oracle Accuracy", xlabel="Episode",
                ylabel="Accuracy", ylim=(-0.02, 1.05))
    axes[2].legend(fontsize=8)

    fig.suptitle("Fig 9: Adversarial Coalition Robustness (PPO, Cartel = Agents 0,1)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"fig09_collusion.{e}", format=e, bbox_inches="tight", dpi=300)
    plt.close(fig)

    os.chdir(orig_dir)
    print(f"\n✓ 9 figures (PDF + PNG) → {od}/")
    return od


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",       type=int, default=50_000)
    parser.add_argument("--seeds",          type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--device",         type=str, default="cpu")
    parser.add_argument("--outdir",         type=str, default="plots/v4_ppo")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    t0 = time.time()
    R  = run_all(N=args.episodes, seeds=args.seeds, device=args.device,
                 checkpoint_dir=args.checkpoint_dir)
    make_plots(R, od=args.outdir)
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s  ({elapsed/3600:.2f}h)")

    # ── Summary table ─────────────────────────────────────────────────
    n_last = max(1, args.episodes // 5)
    s      = slice(-n_last, None)
    print(f"\n{'='*90}")
    print(f"  SUMMARY  (last {n_last} episodes, PPO)")
    print(f"{'='*90}")
    hk = ["truth", "lie", "gather", "mine", "punish", "verify",
          "coop", "oracle_acc", "mean_rep"]
    print(f"  {'Condition':<8}  Truth    Lie   Gath   Mine    Pun    Ver   Coop   Orac    Rep")
    print(f"  {'-'*82}")
    for c in MC + ["AC"]:
        d    = R[c]
        vals = [np.mean(d[k][:, s]) for k in hk]
        print(f"  {c:<8}  " + "  ".join(f"{v:>5.3f}" for v in vals))
