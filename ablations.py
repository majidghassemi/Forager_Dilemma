"""
Channel Ablation: DERL vs DERL-Eth vs DERL-Ep
==============================================
Runs the pure channel splits to test whether epistemic and ethical
enforcement are complementary or redundant.

Conditions:
  DERL      : r_task + r_ep + r_eth + r_soc  (full)
  DERL-Eth  : r_task + r_eth + r_soc          (ethical only; no epistemic)
  DERL-Ep   : r_task + r_ep + r_soc           (epistemic only; no ethical)
"""

import numpy as np
import matplotlib.pyplot as plt
import os, time, warnings
warnings.filterwarnings("ignore")

# ===========================================================================
# ACTIONS (10 total)
# ===========================================================================
UP, DN, LT, RT     = 0, 1, 2, 3
GATHER, MINE       = 4, 5
SIG_T, SIG_L       = 6, 7
PUNISH, VERIFY     = 8, 9
N_ACT = 10
DELTAS = np.array([[-1,0],[1,0],[0,-1],[0,1],
                    [0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])
ACT_NAMES = ["^","v","<",">","Gather","Mine","SigT","SigL","Punish","Verify"]


# ===========================================================================
# ENVIRONMENT (modified to split epistemic / ethical flags)
# ===========================================================================
class Env:
    """
    5x5 gridworld, 4 agents, 8 resources.
    Now supports separate use_epistemic and use_ethical flags.
    """

    def __init__(self, gs=5, na=4, nr=8, T=50, obs_r=2.5, resp=0.10,
                 use_hardcoded=False, use_emergent=True, use_intrinsic=False,
                 use_epistemic=True, use_ethical=True,
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
        # intrinsic: backward compatible (use_intrinsic sets both)
        self.intr_ep = use_epistemic if use_intrinsic else False
        self.intr_eth = use_ethical if use_intrinsic else False
        # social mechanism params
        self.cb = coop_bonus
        self.pr = punish_reward
        self.pd = punish_damage
        self.vr = verify_reward
        self.vd = verify_damage
        # cartel params
        self.cartel = cartel or []
        self.cartel_share = cartel_share
        self.cartel_set = set(self.cartel)

    def reset(self, rng):
        self.pos = rng.integers(0, self.gs, (self.na, 2))
        self.rpos = rng.integers(0, self.gs, (self.nr, 2))
        self.active = np.ones(self.nr, bool)
        self.brd = np.full((self.na, 2), -1)
        self.brd_who = np.full(self.na, -1)
        self.brd_truth = np.zeros(self.na, bool)
        self.prev = np.zeros(self.na, int)
        self.reputation = np.zeros(self.na)
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

        sg = ig & np.any(near, 1)
        sm = im & np.any(near, 1)

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
        tsig = self.rpos[np.argmin(ard, 1)]
        fsig = rng.integers(0, gs, (N, 2))

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
        vis = pd <= self.obs_r
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
        prox = np.clip(1.0 - min_d / gs, 0, 1) * 0.1
        r_task = sg * 2.0 + sm * 5.0 + il * 0.8 + it * 0.2 + prox
        rew = r_task.copy()

        # tracking
        coop_ev = 0
        pun_ok = 0
        pun_tot = 0
        ver_ok = 0
        ver_tot = 0

        # -- HARDCODED PENALTIES ------------------------------------------
        if self.hc:
            pn = np.any(vis, 1)
            rew += (sm & pn) * (-8.0)
            rew += il * (rng.random(N) < 0.3) * (-4.0)

        # -- EMERGENT SOCIAL MECHANISMS -----------------------------------
        if self.em:
            for i in range(N):
                # -- PUNISH ----------------------------------------------
                if ip[i]:
                    pun_tot += 1
                    bad_peers = np.where(vis[i] & prev_bad)[0]

                    if self.cartel_set and i in self.cartel_set:
                        bad_peers = np.array([j for j in bad_peers
                                               if j not in self.cartel_set])

                    if len(bad_peers) > 0:
                        target = bad_peers[np.argmin(pd[i, bad_peers])]
                        rew[i] += self.pr
                        rew[target] -= self.pd
                        self.reputation[target] += 2
                        pun_ok += 1
                    else:
                        rew[i] -= 0.3

                # -- VERIFY ----------------------------------------------
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
                            rew[i] += self.vr
                            rew[j] -= self.vd
                            self.reputation[j] += 3
                            ver_ok += 1
                        else:
                            rew[i] -= 0.2
                            self.reputation[j] = max(self.reputation[j] - 1, 0)
                    else:
                        rew[i] -= 0.1

                # -- COOPERATION BONUS -----------------------------------
                if sg[i]:
                    for j in range(N):
                        if j != i and self.brd_who[j] >= 0 and self.brd_truth[j]:
                            bd = np.sum(np.abs(self.pos[i] - self.brd[j]))
                            if bd <= 1:
                                poster = self.brd_who[j]
                                if poster != i:
                                    rew[poster] += self.cb
                                    rew[i] += 0.3
                                    coop_ev += 1
                                    self.reputation[poster] = max(
                                        self.reputation[poster] - 1, 0)
                                break

        # -- INTRINSIC REWARDS (split by channel) -------------------------
        if self.intr_ep:
            rew += it * 1.0 + il * (-1.0)     # epistemic
        if self.intr_eth:
            rew += sg * 1.0 + sm * (-2.0)     # ethical

        # -- CARTEL REWARD SHARING ----------------------------------------
        if len(self.cartel) >= 2:
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
            nr = int(np.any((d <= 1) & self.active))
            _pd = np.sum(np.abs(self.pos - self.pos[i]), 1)
            _pd[i] = 999
            peer = int(np.any(_pd <= self.obs_r))
            bad = int(np.any((_pd <= self.obs_r) & pb))
            brd = int(np.any(self.brd[:, 0] >= 0))
            dep = int(self.active.sum() < self.nr * 0.5)
            watchers = np.any(_pd <= self.obs_r)
            my_rep = int(watchers and self.reputation[i] >= 2)
            in_cartel = int(i in self.cartel_set)
            ids[i] = (nr + 2*peer + 4*bad + 8*brd +
                      16*dep + 32*my_rep + 64*in_cartel)
        return ids


N_STATES = 128


# ===========================================================================
# TRAINING (single seed, per-agent Q-tables)
# ===========================================================================
def train(env_kw, n_ep=50000, alpha=0.10, gamma=0.7,
          eps0=1.0, epsf=0.05, seed=42, verbose=True):
    """
    Per-agent Q-tables. Single seed run for ablation study.
    """
    env = Env(**env_kw)
    rng = np.random.default_rng(seed)

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

        if verbose and (ep + 1) % 5000 == 0:
            print(f"  ep {ep+1:>5d}  R={er:>8.1f}  "
                  f"T={H['truth'][-1]:.3f} L={H['lie'][-1]:.3f} "
                  f"G={H['gather'][-1]:.3f} M={H['mine'][-1]:.3f} "
                  f"P={H['punish'][-1]:.3f} V={H['verify'][-1]:.3f} "
                  f"co={H['coop'][-1]:.3f} rep={H['mean_rep'][-1]:.1f} "
                  f"eps={eps:.3f}")

    return {k: np.array(v) for k, v in H.items()}


# ===========================================================================
# ABLATION RUNNER
# ===========================================================================
def run_ablations(n_ep=50000, seed=42):
    configs = [
        ("DERL (full)", dict(
            use_hardcoded=False, use_emergent=True, use_intrinsic=True,
            use_epistemic=True, use_ethical=True,
            coop_bonus=1.5)),

        ("DERL-Eth (ethical only)", dict(
            use_hardcoded=False, use_emergent=True, use_intrinsic=True,
            use_epistemic=False, use_ethical=True,
            coop_bonus=1.5)),

        ("DERL-Ep (epistemic only)", dict(
            use_hardcoded=False, use_emergent=True, use_intrinsic=True,
            use_epistemic=True, use_ethical=False,
            coop_bonus=1.5)),
    ]

    R = {}
    for nm, kw in configs:
        print(f"\n{'='*60}")
        print(f"  {nm}")
        print(f"  w_ep={'ON' if kw['use_epistemic'] else 'OFF'}  "
              f"w_eth={'ON' if kw['use_ethical'] else 'OFF'}  "
              f"w_soc=ON")
        print(f"{'='*60}")
        res = train(kw, n_ep=n_ep, seed=seed, verbose=True)
        R[nm] = res

    return R


# ===========================================================================
# PLOTTING
# ===========================================================================
def smooth(x, w=500):
    if len(x) < w:
        return x
    window = np.hanning(w)
    window /= window.sum()
    p = np.pad(x, (w//2, w - w//2 - 1), mode='edge')
    return np.convolve(p, window, 'valid')


def plot_ablations(R, od="plots/ablation"):
    os.makedirs(od, exist_ok=True)

    colors = {
        "DERL (full)": "#1B7837",
        "DERL-Eth (ethical only)": "#E66100",
        "DERL-Ep (epistemic only)": "#5D3A9B",
    }
    styles = {
        "DERL (full)": dict(ls="-", lw=2.5),
        "DERL-Eth (ethical only)": dict(ls="--", lw=2),
        "DERL-Ep (epistemic only)": dict(ls="-.", lw=2),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.titlesize': 13, 'axes.labelsize': 12,
        'legend.fontsize': 9,
    })

    metrics = [
        ("truth", "Truth-Telling Rate", "Rate"),
        ("mine", "Mining Rate", "Rate"),
    ]

    # Panels (a) and (b): behavioral rates
    for ax, (key, title, ylabel) in zip(axes[:2], metrics):
        for nm in ["DERL (full)", "DERL-Eth (ethical only)", "DERL-Ep (epistemic only)"]:
            data = R[nm][key]
            s = smooth(data)
            ax.plot(s, label=nm.split(" (")[0], color=colors[nm], **styles[nm])
        ax.set(title=title, xlabel="Episode", ylabel=ylabel)
        ax.set_ylim(-0.02, max(0.6, ax.get_ylim()[1]))
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)

    # Panel (c): cumulative reward
    ax = axes[2]
    for nm in ["DERL (full)", "DERL-Eth (ethical only)", "DERL-Ep (epistemic only)"]:
        data = R[nm]["reward"]
        ax.plot(np.cumsum(data), label=nm.split(" (")[0], color=colors[nm], **styles[nm])
    ax.set(title="Cumulative Reward", xlabel="Episode", ylabel="Cumulative Reward")
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Channel Ablation: Epistemic vs Ethical Enforcement",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{od}/ablation_channels.{fmt}", format=fmt,
                    bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"\n  Figure saved to {od}/ablation_channels.pdf")
    return od


# ===========================================================================
# SUMMARY TABLE
# ===========================================================================
def print_summary(R):
    print(f"\n{'='*65}")
    print(f"  SUMMARY (last 5000 episodes)")
    print(f"{'='*65}")
    header_keys = ["truth", "lie", "gather", "mine",
                   "punish", "verify", "coop", "oracle_acc", "mean_rep"]
    print(f"  {'Condition':<20} " +
          " ".join(f"{'Truth':>6} {'Lie':>6} {'Gath':>6} {'Mine':>6} "
                   f"{'Pun':>5} {'Ver':>5} {'Coop':>5} {'Orac':>5} {'Rep':>5}"))
    print(f"  {'-'*72}")
    for nm in ["DERL (full)", "DERL-Eth (ethical only)", "DERL-Ep (epistemic only)"]:
        d = R[nm]
        s = slice(-5000, None)
        vals = [np.mean(d[k][s]) for k in header_keys]
        print(f"  {nm:<20} " +
              " ".join(f"{v:>5.3f}" for v in vals))

    # Key comparison lines
    print(f"\n  {'KEY FINDINGS:'}")
    full_t = np.mean(R["DERL (full)"]["truth"][-5000:])
    ep_t = np.mean(R["DERL-Ep (epistemic only)"]["truth"][-5000:])
    eth_m = np.mean(R["DERL-Eth (ethical only)"]["mine"][-5000:])
    full_m = np.mean(R["DERL (full)"]["mine"][-5000:])

    print(f"    Truth rate:  DERL={full_t:.3f} vs DERL-Ep={ep_t:.3f} "
          f"(delta = {full_t - ep_t:+.3f})")
    print(f"    Mining rate: DERL={full_m:.3f} vs DERL-Eth={eth_m:.3f} "
          f"(delta = {full_m - eth_m:+.3f})")

    if ep_t < full_t * 0.5 and eth_m > full_m * 2:
        print(f"    -> Both channels are necessary. Neither compensates for the other.")
    elif ep_t < full_t * 0.5:
        print(f"    -> Epistemic channel is essential; ethical channel partially compensates.")
    elif eth_m > full_m * 2:
        print(f"    -> Ethical channel is essential; epistemic channel partially compensates.")
    else:
        print(f"    -> One channel may be dominant; inspect full results carefully.")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    t0 = time.time()
    R = run_ablations(n_ep=25000, seed=42)
    plot_ablations(R)
    print_summary(R)
    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")