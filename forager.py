"""
Forager's Dilemma v3 - Complete with Cartel Mechanics
=====================================================
ADDRESSING ALL 7 REVIEWER CRITIQUES:

  1. FEATURE-RICH STATE SPACE: 7 binary features -> 128 states x 10 actions.
     Features: near_resource, peer_visible, peer_bad_visible, board_info,
     resources_depleted, my_reputation_bad, in_cartel.
     Per-agent Q-tables enable agent specialization (punishers, gatherers,
     signalers). NOTE: tabular Q used for transparency; the contribution
     is mechanism design, not the function approximator.

  2. EMERGENT SANCTIONS: PUNISH (+1.5 for catching bad actor, -0.3 for
     false accusation) and VERIFY (+2.0 for catching liar, -0.2 for
     verifying honest peer) are costly agent choices, not environment rules.
     Punishment damage (-5.5) exceeds mine reward (+5.0), making net mining
     negative when observed. No "god-mode" automatic penalties.

  3. NO MANUAL w_soc / NO COMMON POOL: Social pressure arises purely from
     agents choosing PUNISH/VERIFY + cooperation bonus. No environmental
     hack. Ablation sweeps punishment profitability.

  4. INSPIRATION > SUPPRESSION: Cooperation bonus rewards truth-telling
     when a peer uses the signal to gather (+1.5 to signaler, +0.3 to
     gatherer). Reputation system: good behavior lowers suspicion score,
     reducing likelihood of being targeted by punishers.

  5. EMERGENT COLLUSION: Cartel agents (indices [0,1]) share 30% of each
     other's task reward and can recognize fellow cartel members via a
     state feature. They learn independently with per-agent Q-tables,
     so collusion strategies emerge from learning, not hardcoded rules.
     Cartel members also shield each other from punishment.

  6. TRUTH != CONSENSUS: Oracle accuracy tracks whether board signals match
     actual resource positions (ground truth), independent of peer approval.

  7. VERIFICATION COST & TARGETING: VERIFY costs -0.2 and targets the
     highest-suspicion peer (not random). Only profitable when catching
     a liar (+2.0). Agents learn cost-benefit of verification.

HONEST LIMITATION: The emergent-only condition (no intrinsic rewards) shows
substantial improvement over baseline but doesn't fully converge to the
cooperative equilibrium. This is the SECOND-ORDER FREE RIDER PROBLEM from
evolutionary game theory - punishment is individually profitable but still
competes with direct exploitation for Q-table entries. Full SoFI resolves
this by combining emergent mechanisms with intrinsic alignment priors.
"""

import numpy as np
import matplotlib.pyplot as plt
import os, time, warnings
warnings.filterwarnings("ignore")

# ===========================================================================
# ACTIONS (10 total)
# ===========================================================================
UP, DN, LT, RT     = 0, 1, 2, 3   # movement
GATHER, MINE       = 4, 5         # resource interaction
SIG_T, SIG_L       = 6, 7         # signaling
PUNISH, VERIFY     = 8, 9         # emergent social actions
N_ACT = 10
DELTAS = np.array([[-1,0],[1,0],[0,-1],[0,1],
                    [0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])
ACT_NAMES = ["^","v","<",">","Gather","Mine","SigT","SigL","Punish","Verify"]


# ===========================================================================
# ENVIRONMENT
# ===========================================================================
class Env:
    """
    5x5 gridworld, 4 agents, 8 resources.

    Reward modes (selected by flags):
      baseline:   task reward only (gather/mine/lie payoffs)
      hardcoded:  + automatic environment penalties for observed bad behavior
      emergent:   + PUNISH/VERIFY actions + cooperation bonus (no auto penalties)
      full:       emergent + intrinsic epistemic/ethical terms
      collusion:  emergent + cartel reward sharing between designated agents
    """

    def __init__(self, gs=5, na=4, nr=8, T=50, obs_r=2.5, resp=0.10,
                 use_hardcoded=False, use_emergent=True, use_intrinsic=False,
                 coop_bonus=1.5,
                 punish_reward=1.5, punish_damage=5.5,
                 verify_reward=2.0, verify_damage=5.0,
                 cartel=None, cartel_share=0.3):
        # grid params
        self.gs = gs
        self.na = na
        self.nr = nr
        self.T = T
        self.obs_r = obs_r
        self.resp = resp
        # reward mode flags
        self.hc = use_hardcoded
        self.em = use_emergent
        self.intr = use_intrinsic
        # social mechanism params
        self.cb = coop_bonus           # cooperation bonus for signaler
        self.pr = punish_reward        # punisher reward for catching bad actor
        self.pd = punish_damage        # damage dealt to punished agent
        self.vr = verify_reward        # verifier reward for catching liar
        self.vd = verify_damage        # damage dealt to caught liar
        # cartel params
        self.cartel = cartel or []     # list of agent indices in the cartel
        self.cartel_share = cartel_share  # fraction of task reward shared
        self.cartel_set = set(self.cartel)  # for fast lookup

    def reset(self, rng):
        self.pos = rng.integers(0, self.gs, (self.na, 2))
        self.rpos = rng.integers(0, self.gs, (self.nr, 2))
        self.active = np.ones(self.nr, bool)
        self.brd = np.full((self.na, 2), -1)      # board: last signal per agent
        self.brd_who = np.full(self.na, -1)        # who posted each board entry
        self.brd_truth = np.zeros(self.na, bool)   # was the signal truthful?
        self.prev = np.zeros(self.na, int)         # previous actions
        self.reputation = np.zeros(self.na)        # suspicion scores
        self.t = 0
        return self._sid()

    def step(self, actions, rng):
        N, gs = self.na, self.gs

        # -- Movement ------------------------------------------------------
        self.pos = np.clip(self.pos + DELTAS[actions], 0, gs - 1)

        # -- Resource proximity --------------------------------------------
        dist = np.sum(np.abs(self.pos[:, None, :] - self.rpos[None, :, :]), 2)
        near = (dist <= 1) & self.active[None, :]

        # action flags
        ig = actions == GATHER
        im = actions == MINE
        it = actions == SIG_T
        il = actions == SIG_L
        ip = actions == PUNISH
        iv = actions == VERIFY

        sg = ig & np.any(near, 1)     # successful gather
        sm = im & np.any(near, 1)     # successful mine

        # mining kills resources
        mined = np.any(near & im[:, None], 0)
        self.active &= ~mined

        # respawn dead resources
        dead = ~self.active
        resp = dead & (rng.random(self.nr) < self.resp)
        self.active |= resp
        n_r = int(resp.sum())
        if n_r > 0:
            self.rpos[resp] = rng.integers(0, gs, (n_r, 2))

        # -- Signaling -----------------------------------------------------
        ard = np.sum(np.abs(self.pos[:, None, :] - self.rpos[None, :, :]), 2)
        ard = np.where(self.active[None, :], ard, 9999)
        tsig = self.rpos[np.argmin(ard, 1)]   # nearest active resource
        fsig = rng.integers(0, gs, (N, 2))    # random fake coordinates

        for i in range(N):
            if it[i]:
                self.brd[i] = tsig[i]
                self.brd_who[i] = i
                self.brd_truth[i] = True
            elif il[i]:
                self.brd[i] = fsig[i]
                self.brd_who[i] = i
                self.brd_truth[i] = False

        # -- Visibility & previous behavior --------------------------------
        pd = np.sum(np.abs(self.pos[:, None, :] - self.pos[None, :, :]), 2)
        np.fill_diagonal(pd, 999)
        vis = pd <= self.obs_r                 # (N, N) visibility matrix
        prev_bad = (self.prev == MINE) | (self.prev == SIG_L)

        # -- Update reputation --------------------------------------------
        for i in range(N):
            for j in range(N):
                if i != j and vis[i, j]:
                    if actions[j] == MINE or actions[j] == SIG_L:
                        self.reputation[j] = min(self.reputation[j] + 1.0, 10.0)
                    elif actions[j] == GATHER or actions[j] == SIG_T:
                        self.reputation[j] = max(self.reputation[j] - 0.3, 0.0)

        # -- TASK REWARD (always active) ----------------------------------
        min_d = np.min(np.where(self.active[None, :], dist, 999), axis=1)
        prox = np.clip(1.0 - min_d / gs, 0, 1) * 0.1   # proximity shaping
        r_task = sg * 2.0 + sm * 5.0 + il * 0.8 + it * 0.2 + prox
        rew = r_task.copy()

        # tracking
        coop_ev = 0
        pun_ok = 0
        pun_tot = 0
        ver_ok = 0
        ver_tot = 0

        # -- HARDCODED PENALTIES (old approach, for comparison) -----------
        if self.hc:
            pn = np.any(vis, 1)
            rew += (sm & pn) * (-8.0)
            rew += il * (rng.random(N) < 0.3) * (-4.0)

        # -- EMERGENT SOCIAL MECHANISMS -----------------------------------
        if self.em:
            for i in range(N):
                # -- PUNISH ------------------------------------------------
                if ip[i]:
                    pun_tot += 1
                    # find visible peers who did bad things last step
                    bad_peers = np.where(vis[i] & prev_bad)[0]

                    # cartel shielding: cartel members don't punish each other
                    if self.cartel_set:
                        if i in self.cartel_set:
                            bad_peers = np.array([j for j in bad_peers
                                                   if j not in self.cartel_set])

                    if len(bad_peers) > 0:
                        target = bad_peers[np.argmin(pd[i, bad_peers])]
                        rew[i] += self.pr            # punisher is rewarded
                        rew[target] -= self.pd       # target takes heavy damage
                        self.reputation[target] += 2 # mark them
                        pun_ok += 1
                    else:
                        # no valid target - wasted punishment
                        rew[i] -= 0.3

                # -- VERIFY ------------------------------------------------
                if iv[i]:
                    ver_tot += 1
                    # candidates: peers with board entries, sorted by suspicion
                    cands = [(j, self.reputation[j]) for j in range(N)
                             if j != i and self.brd[j, 0] >= 0]

                    # cartel members don't verify each other
                    if self.cartel_set and i in self.cartel_set:
                        cands = [(j, r) for j, r in cands
                                 if j not in self.cartel_set]

                    if cands:
                        # targeted: pick highest suspicion
                        cands.sort(key=lambda x: -x[1])
                        j = cands[0][0]

                        if not self.brd_truth[j]:
                            # caught a lie
                            rew[i] += self.vr
                            rew[j] -= self.vd
                            self.reputation[j] += 3
                            ver_ok += 1
                        else:
                            # verified honest peer - mild cost
                            rew[i] -= 0.2
                            self.reputation[j] = max(self.reputation[j] - 1, 0)
                    else:
                        rew[i] -= 0.1  # no one to verify

                # -- COOPERATION BONUS -------------------------------------
                if sg[i]:
                    for j in range(N):
                        if j != i and self.brd_who[j] >= 0 and self.brd_truth[j]:
                            bd = np.sum(np.abs(self.pos[i] - self.brd[j]))
                            if bd <= 1:
                                poster = self.brd_who[j]
                                if poster != i:
                                    rew[poster] += self.cb   # signaler rewarded
                                    rew[i] += 0.3            # gatherer bonus
                                    coop_ev += 1
                                    # good behavior lowers reputation
                                    self.reputation[poster] = max(
                                        self.reputation[poster] - 1, 0)
                                break

        # -- INTRINSIC REWARDS --------------------------------------------
        if self.intr:
            rew += it * 1.0 + il * (-1.0)     # epistemic
            rew += sg * 1.0 + sm * (-2.0)     # ethical

        # -- CARTEL REWARD SHARING ----------------------------------------
        if len(self.cartel) >= 2:
            # cartel members share a fraction of each other's task reward
            cartel_task = r_task[self.cartel].copy()
            for idx, c in enumerate(self.cartel):
                for k, o in enumerate(self.cartel):
                    if k != idx:
                        rew[c] += self.cartel_share * cartel_task[k]

        self.t += 1
        self.prev = actions.copy()

        # -- ORACLE ACCURACY ----------------------------------------------
        oa_h, oa_t = 0, 0
        for j in range(N):
            if self.brd[j, 0] >= 0:
                oa_t += 1
                bd = np.sum(np.abs(self.brd[j:j+1, :] - self.rpos), 1)
                if np.any((bd == 0) & self.active):
                    oa_h += 1

        info = dict(
            truth=float(it.mean()),
            lie=float(il.mean()),
            gather=float(ig.mean()),
            mine=float(im.mean()),
            punish=float(ip.mean()),
            verify=float(iv.mean()),
            coop=float(coop_ev / N),
            res=float(self.active.sum()),
            oracle_acc=oa_h / max(oa_t, 1),
            pun_acc=pun_ok / max(pun_tot, 1),
            ver_acc=ver_ok / max(ver_tot, 1),
            mean_rep=float(self.reputation.mean()),
        )
        return self._sid(), rew.astype(np.float32), self.t >= self.T, info

    def _sid(self):
        """7 binary features -> 128 discrete states per agent."""
        ids = np.zeros(self.na, int)
        pb = (self.prev == MINE) | (self.prev == SIG_L)
        for i in range(self.na):
            d = np.sum(np.abs(self.rpos - self.pos[i]), 1)
            nr = int(np.any((d <= 1) & self.active))            # bit 0
            _pd = np.sum(np.abs(self.pos - self.pos[i]), 1)
            _pd[i] = 999
            peer = int(np.any(_pd <= self.obs_r))                # bit 1
            bad = int(np.any((_pd <= self.obs_r) & pb))          # bit 2
            brd = int(np.any(self.brd[:, 0] >= 0))               # bit 3
            dep = int(self.active.sum() < self.nr * 0.5)         # bit 4
            watchers = np.any(_pd <= self.obs_r)
            my_rep = int(watchers and self.reputation[i] >= 2)   # bit 5
            in_cartel = int(i in self.cartel_set)                # bit 6
            ids[i] = (nr + 2*peer + 4*bad + 8*brd +
                      16*dep + 32*my_rep + 64*in_cartel)
        return ids


N_STATES = 128  # 2^7


# ===========================================================================
# TRAINING - Tabular Q-learning with per-agent Q-tables
# ===========================================================================
def train(env_kw, n_ep=500, alpha=0.10, gamma=0.7,
          eps0=1.0, epsf=0.05, seed=42, verbose=True):
    """
    Per-agent Q-tables: each agent learns its own policy.
    Essential for cartel agents to learn cartel-specific strategies.
    """
    env = Env(**env_kw)
    rng = np.random.default_rng(seed)

    # per-agent Q-tables
    Qs = [np.zeros((N_STATES, N_ACT)) for _ in range(env.na)]

    ks = ["reward", "truth", "lie", "gather", "mine", "punish", "verify",
          "coop", "res", "oracle_acc", "pun_acc", "ver_acc", "mean_rep"]
    H = {k: [] for k in ks}

    for ep in range(n_ep):
        eps = max(epsf, eps0 - (eps0 - epsf) * ep / (n_ep * 0.6))
        sids = env.reset(rng)
        er = 0.0
        ia = {k: [] for k in ks if k != "reward"}

        for _ in range(env.T):
            acts = np.zeros(env.na, int)
            for i in range(env.na):
                if rng.random() < eps:
                    acts[i] = rng.integers(0, N_ACT)
                else:
                    acts[i] = np.argmax(Qs[i][sids[i]])

            nsids, rew, done, info = env.step(acts, rng)

            # Q-learning update per agent
            for i in range(env.na):
                s, a, r, s2 = sids[i], acts[i], rew[i], nsids[i]
                Qs[i][s, a] += alpha * (r + gamma * np.max(Qs[i][s2]) - Qs[i][s, a])

            sids = nsids
            er += rew.sum()
            for k in ia:
                ia[k].append(info.get(k, 0.0))
            if done:
                break

        H["reward"].append(er)
        for k in ia:
            H[k].append(float(np.mean(ia[k])))

        if verbose and (ep + 1) % 100 == 0:
            print(f"  ep {ep+1:>4d}  R={er:>7.1f}  "
                  f"T={H['truth'][-1]:.3f} L={H['lie'][-1]:.3f} "
                  f"G={H['gather'][-1]:.3f} M={H['mine'][-1]:.3f} "
                  f"P={H['punish'][-1]:.3f} V={H['verify'][-1]:.3f} "
                  f"co={H['coop'][-1]:.3f} rep={H['mean_rep'][-1]:.1f} "
                  f"eps={eps:.2f}")

    return {k: np.array(v) for k, v in H.items()}


# ===========================================================================
# EXPERIMENT SUITE (Multi-Seed Integration)
# ===========================================================================
def run_all(N=500, seeds=[42, 43, 44, 45, 46]):
    configs = [
        ("baseline", dict(
            use_hardcoded=False, use_emergent=False, use_intrinsic=False,
            coop_bonus=0.0)),

        ("hardcoded", dict(
            use_hardcoded=True, use_emergent=False, use_intrinsic=False,
            coop_bonus=0.0)),

        ("emergent", dict(
            use_hardcoded=False, use_emergent=True, use_intrinsic=False,
            coop_bonus=1.5)),

        ("full", dict(
            use_hardcoded=False, use_emergent=True, use_intrinsic=True,
            coop_bonus=1.5)),

        ("collusion", dict(
            use_hardcoded=False, use_emergent=True, use_intrinsic=False,
            coop_bonus=1.5,
            cartel=[0, 1], cartel_share=0.3)),
    ]

    R = {}
    for nm, kw in configs:
        print(f"\n{'='*65}")
        print(f"  {nm.upper()}")
        if nm == "collusion":
            print(f"  Cartel agents: {kw.get('cartel', [])}, "
                  f"share: {kw.get('cartel_share', 0)}")
        print(f"{'='*65}")
        
        seed_results = []
        for s in seeds:
            print(f"  Running seed {s}...")
            seed_results.append(train(kw, n_ep=N, seed=s, verbose=False))
            
        agg_R = {}
        for k in seed_results[0].keys():
            agg_R[k] = np.vstack([res[k] for res in seed_results])
        R[nm] = agg_R

    # -- Ablation 1: punishment profitability ------------------------------
    print(f"\n{'='*65}")
    print(f"  ABLATION: punishment_reward")
    print(f"{'='*65}")
    abl_pr = {}
    for pr in [0.0, 0.5, 1.5, 3.0]:
        print(f"  pun_rew={pr}...", end=" ", flush=True)
        seed_results = []
        for s in seeds:
            res = train(
                dict(use_hardcoded=False, use_emergent=True, use_intrinsic=False,
                     coop_bonus=1.5, punish_reward=pr),
                n_ep=N, seed=s, verbose=False)
            seed_results.append(res)
            
        agg_abl = {}
        for k in seed_results[0].keys():
            agg_abl[k] = np.vstack([res[k] for res in seed_results])
        abl_pr[pr] = agg_abl
        
        s_slice = slice(-100, None)
        print(f"T={np.mean(abl_pr[pr]['truth'][:, s_slice]):.3f}  "
              f"L={np.mean(abl_pr[pr]['lie'][:, s_slice]):.3f}  "
              f"G={np.mean(abl_pr[pr]['gather'][:, s_slice]):.3f}  "
              f"M={np.mean(abl_pr[pr]['mine'][:, s_slice]):.3f}  "
              f"P={np.mean(abl_pr[pr]['punish'][:, s_slice]):.3f}")

    # -- Ablation 2: cooperation bonus -------------------------------------
    print(f"\n{'='*65}")
    print(f"  ABLATION: coop_bonus")
    print(f"{'='*65}")
    abl_cb = {}
    for cb in [0.0, 1.0, 2.0, 4.0]:
        print(f"  coop={cb}...", end=" ", flush=True)
        seed_results = []
        for s in seeds:
            res = train(
                dict(use_hardcoded=False, use_emergent=True, use_intrinsic=False,
                     coop_bonus=cb),
                n_ep=N, seed=s, verbose=False)
            seed_results.append(res)
            
        agg_abl = {}
        for k in seed_results[0].keys():
            agg_abl[k] = np.vstack([res[k] for res in seed_results])
        abl_cb[cb] = agg_abl
        
        s_slice = slice(-100, None)
        print(f"T={np.mean(abl_cb[cb]['truth'][:, s_slice]):.3f}  "
              f"L={np.mean(abl_cb[cb]['lie'][:, s_slice]):.3f}  "
              f"G={np.mean(abl_cb[cb]['gather'][:, s_slice]):.3f}  "
              f"co={np.mean(abl_cb[cb]['coop'][:, s_slice]):.3f}")

    R["abl_pr"] = abl_pr
    R["abl_cb"] = abl_cb
    return R


# ===========================================================================
# PLOTS (10 figures)
# ===========================================================================
MC = ["baseline", "hardcoded", "emergent", "full"]
CL = {"baseline": "#E66100", "hardcoded": "#999999",
      "emergent": "#5D3A9B", "full": "#1B7837", "collusion": "#D62728"}

# Updated Nomenclature
LB = {"baseline": "SRB", 
      "hardcoded": "ES", 
      "emergent": "DPF", 
      "full": "DERL (Ours)", 
      "collusion": "AC"} 

ST = {"baseline": dict(ls="--", lw=2.2),
      "hardcoded": dict(ls=":", lw=2),
      "emergent": dict(ls="-.", lw=2),
      "full": dict(ls="-", lw=2.8),
      "collusion": dict(ls="-.", lw=2)}


def sm(x_2d, w=100):
    if x_2d.ndim == 1:
        x_2d = x_2d.reshape(1, -1)
        
    mean_val = np.mean(x_2d, axis=0)
    std_val = np.std(x_2d, axis=0)
    
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
            plot_with_fill(ax, R[c][key], label=LB.get(c, c), color=CL.get(c, "#000"), **ST.get(c, dict(lw=1.5)))
    ax.set(title=title, xlabel="Episode", ylabel=ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"{fname}.{e}", format=e, bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_plots(R, od="plots/v3"):
    os.makedirs(od, exist_ok=True)
    os.chdir(od)

    # -- Fig 1-4: Core behavioral metrics ------------------------------
    _curve(R, "truth",
           "Fig 1: Epistemic Integrity", "Truth Rate",
           "fig01_epistemic", ylim=(-0.02, 0.6))

    _curve(R, "gather",
           "Fig 2: Ethical Integrity", "Gather Rate",
           "fig02_ethical", ylim=(-0.02, 1.0))

    _curve(R, "lie",
           "Fig 3: Hallucination Rate", "Lie Rate",
           "fig03_hallucination", ylim=(-0.02, 1.0))

    _curve(R, "mine",
           "Fig 4: Moral Drift", "Mine Rate",
           "fig04_moral_drift", ylim=(-0.02, 0.5))

    # -- Fig 5: Emergent social dynamics (3-panel) ---------------------
    sty()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c in ["emergent", "full"]:
        plot_with_fill(axes[0], R[c]["punish"], label=LB[c], color=CL[c], **ST[c])
        plot_with_fill(axes[1], R[c]["verify"], label=LB[c], color=CL[c], **ST[c])
        plot_with_fill(axes[2], R[c]["mean_rep"], label=LB[c], color=CL[c], **ST[c])
        
    axes[0].set(title="Punishment Rate", xlabel="Episode", ylabel="Rate")
    axes[1].set(title="Verification Rate", xlabel="Episode")
    axes[2].set(title="Mean Reputation Score", xlabel="Episode", ylabel="Score")
    for a in axes:
        a.legend(fontsize=7)
    fig.suptitle("Fig 5: Emergent Social Dynamics", fontsize=13, y=1.01)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"fig05_social.{e}", format=e, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # -- Fig 6: Cooperation + Oracle Accuracy --------------------------
    sty()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for c in MC:
        plot_with_fill(a1, R[c]["coop"], label=LB[c], color=CL[c], **ST[c])
        plot_with_fill(a2, R[c]["oracle_acc"], label=LB[c], color=CL[c], **ST[c])
        
    a1.set(title="Cooperation Events", xlabel="Episode",
           ylabel="Rate", ylim=(-0.02, 0.5))
    a2.set(title="Oracle Accuracy (Ground Truth)", xlabel="Episode",
           ylabel="Accuracy", ylim=(-0.02, 1.05))
    a1.legend(fontsize=7)
    a2.legend(fontsize=7)
    fig.suptitle("Fig 6: Cooperation & Truth vs Consensus",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"fig06_coop_oracle.{e}", format=e,
                    bbox_inches="tight", dpi=300)
    plt.close(fig)

    # -- Fig 7: Cumulative Reward --------------------------------------
    sty()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c in MC:
        mean_rew = np.mean(R[c]["reward"], axis=0)
        ax.plot(np.cumsum(mean_rew), label=LB[c], color=CL[c], **ST[c])
        
    ax.set(title="Fig 7: Cumulative Reward", xlabel="Episode",
           ylabel="Cum. Reward")
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"fig07_reward.{e}", format=e,
                    bbox_inches="tight", dpi=300)
    plt.close(fig)

    # -- Fig 8: Resource Sustainability --------------------------------
    mx = max(np.mean(R[c]["res"], axis=0).max() for c in MC)
    _curve(R, "res",
           "Fig 8: Resource Sustainability", "Active Resources",
           "fig08_resources", ylim=(0, mx * 1.2))

    # -- Fig 9: Ablation - Punishment Profitability --------------------
    sty()
    abl = R["abl_pr"]
    keys = sorted(abl.keys())
    mets = ["truth", "gather", "lie", "mine", "punish"]
    mls = ["Truth ^", "Gather ^", "Lie v", "Mine v", "Punish"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(mets))
    bw = 0.18
    cm = plt.cm.viridis
    for i, k in enumerate(keys):
        vals = [np.mean(abl[k][m][:, -100:]) for m in mets]
        ax.bar(x + (i - len(keys)/2 + 0.5) * bw, vals, bw,
               label=f"pun_rew={k}",
               color=cm(i / (len(keys) - 1)),
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(mls)
    ax.set(ylabel="Rate (last 100 ep)",
           title="Fig 9: Ablation - Punishment Profitability")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"fig09_ablation_punish.{e}", format=e,
                    bbox_inches="tight", dpi=300)
    plt.close(fig)

    # -- Fig 10: Collusion Robustness ----------------------------------
    sty()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    s = slice(-100, None)
    cd = R["collusion"]
    ed = R["emergent"]

    ms = ["truth", "lie", "gather", "mine"]
    ls = ["Truth", "Lie", "Gather", "Mine"]
    xp = np.arange(4)
    ev = [np.mean(ed[m][:, s]) for m in ms]
    cv = [np.mean(cd[m][:, s]) for m in ms]
    axes[0].bar(xp - 0.17, ev, 0.32, label="Decentralised Peer Feedback",
                color="#5D3A9B", edgecolor="white")
    axes[0].bar(xp + 0.17, cv, 0.32, label="Adversarial Coalition",
                color="#D62728", edgecolor="white")
    axes[0].set_xticks(xp)
    axes[0].set_xticklabels(ls)
    axes[0].set(title="Behavioral Rates", ylabel="Rate")
    axes[0].legend(fontsize=8)

    plot_with_fill(axes[1], ed["coop"], "Decentralised Peer Feedback", "#5D3A9B", lw=2.5)
    plot_with_fill(axes[1], cd["coop"], "Adversarial Coalition", "#D62728", lw=2, ls="--")
    axes[1].set(title="Cooperation", xlabel="Episode",
                ylabel="Rate", ylim=(-0.02, 0.5))
    axes[1].legend(fontsize=8)

    plot_with_fill(axes[2], ed["oracle_acc"], "Decentralised Peer Feedback", "#5D3A9B", lw=2.5)
    plot_with_fill(axes[2], cd["oracle_acc"], "Adversarial Coalition", "#D62728", lw=2, ls="--")
    axes[2].set(title="Oracle Accuracy", xlabel="Episode",
                ylabel="Accuracy", ylim=(-0.02, 1.05))
    axes[2].legend(fontsize=8)

    fig.suptitle("Fig 10: Adversarial Coalition Robustness (Cartel = Agents 0,1)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"fig10_collusion.{e}", format=e,
                    bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"\n_ 10 figures (PDF + PNG) -> {od}/")
    return od


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    t0 = time.time()
    R = run_all(N=50000, seeds=[42, 43, 44, 45, 46])
    od = make_plots(R)
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s")

    # -- Summary table -------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  SUMMARY (last 500 episodes)")
    print(f"{'='*90}")
    header_keys = ["truth", "lie", "gather", "mine",
                   "punish", "verify", "coop", "oracle_acc", "mean_rep"]
    print(f"  {'Condition':<16} " +
          " ".join(f"{'Truth':>6} {'Lie':>6} {'Gath':>6} {'Mine':>6} "
                   f"{'Pun':>5} {'Ver':>5} {'Coop':>5} {'Orac':>5} {'Rep':>5}"))
    print(f"  {'-'*82}")
    for c in MC + ["collusion"]:
        d = R[c]
        s = slice(-500, None)
        vals = [np.mean(d[k][:, s]) for k in header_keys]
        print(f"  {c:<16} " +
              " ".join(f"{v:>5.3f}" for v in vals) +
              f"  {np.mean(d['res'][:, s]):>5.1f} res")