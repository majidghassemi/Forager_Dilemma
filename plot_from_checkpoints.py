"""
Generate plots from saved .npz checkpoints.
Figs 1-8 and 10 use the 5 main conditions.
Fig 9 uses whatever ablation files exist (partial is fine).

Usage:
    conda run -n forager python plot_from_checkpoints.py
    conda run -n forager python plot_from_checkpoints.py --checkpoint-dir checkpoints --outdir plots/v4_ppo
"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt

CONDITIONS = ["SRB", "ES", "DPF", "DERL", "AC"]
ABL_PRS    = [0.0, 0.5, 1.5, 3.0]
MC         = ["SRB", "ES", "DPF", "DERL"]

CL = {"SRB": "#E66100", "ES": "#999999",
      "DPF": "#5D3A9B", "DERL": "#1B7837", "AC": "#D62728"}
LB = {"SRB":  "SRB",
      "ES":   "ES",
      "DPF":  "DPF",
      "DERL": "DERL (Ours)",
      "AC":   "AC"}
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
    w_dyn = max(w, int(x_2d.shape[1] * 0.02))
    if len(mean_val) < w_dyn:
        return mean_val, std_val * 0.6
    window = np.hanning(w_dyn); window /= window.sum()
    pad = (w_dyn // 2, w_dyn - w_dyn // 2 - 1)
    smooth_mean = np.convolve(np.pad(mean_val, pad, mode='edge'), window, 'valid')
    smooth_std  = np.convolve(np.pad(std_val,  pad, mode='edge'), window, 'valid')
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


def _curve(R, key, ylabel, fname, conds=None, ylim=None):
    sty()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c in (conds or MC):
        if c in R:
            plot_with_fill(ax, R[c][key], label=LB.get(c, c),
                           color=CL.get(c, "#000"), **ST.get(c, dict(lw=1.5)))
    ax.set(xlabel="Episode", ylabel=ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"{fname}.{e}", format=e, bbox_inches="tight", dpi=300)
    plt.close(fig)


def load_checkpoints(ckpt_dir, seeds):
    R = {}

    for nm in CONDITIONS:
        seed_results = []
        for s in seeds:
            ckpt = os.path.join(ckpt_dir, f"{nm}_seed{s}.npz")
            if os.path.exists(ckpt):
                data = np.load(ckpt)
                seed_results.append({k: data[k] for k in data.files})
            else:
                print(f"  WARNING: {os.path.basename(ckpt)} missing")
        if seed_results:
            R[nm] = {k: np.vstack([r[k] for r in seed_results]) for k in seed_results[0]}
            print(f"  {nm}: {len(seed_results)}/{len(seeds)} seeds")
        else:
            print(f"  {nm}: NO checkpoints found, skipping")

    abl_pr = {}
    for pr in ABL_PRS:
        seed_results = []
        for s in seeds:
            ckpt = os.path.join(ckpt_dir, f"abl_pr{pr}_seed{s}.npz")
            if os.path.exists(ckpt):
                data = np.load(ckpt)
                seed_results.append({k: data[k] for k in data.files})
        if seed_results:
            abl_pr[pr] = {k: np.vstack([r[k] for r in seed_results]) for k in seed_results[0]}
            print(f"  ablation pr={pr}: {len(seed_results)}/{len(seeds)} seeds")
    if abl_pr:
        R["abl_pr"] = abl_pr
        print(f"  ablation: {len(abl_pr)}/4 punish_reward values loaded")
    else:
        print("  ablation: no checkpoints found — fig 9 will be skipped")

    return R


def make_plots(R, od):
    os.makedirs(od, exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir(od)
    n_saved = 0

    _curve(R, "truth",  "Truth Rate",  "fig01_epistemic",    ylim=(-0.02, 0.65))
    _curve(R, "gather", "Gather Rate", "fig02_ethical",       ylim=(-0.02, 1.0))
    _curve(R, "lie",    "Lie Rate",    "fig03_hallucination", ylim=(-0.02, 1.0))
    _curve(R, "mine",   "Mine Rate",   "fig04_moral_drift",   ylim=(-0.02, 0.5))
    n_saved += 4

    # Fig 5
    sty()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c in ["DPF", "DERL"]:
        if c not in R:
            continue
        plot_with_fill(axes[0], R[c]["punish"],   label=LB[c], color=CL[c], **ST[c])
        plot_with_fill(axes[1], R[c]["verify"],   label=LB[c], color=CL[c], **ST[c])
        plot_with_fill(axes[2], R[c]["mean_rep"], label=LB[c], color=CL[c], **ST[c])
    axes[0].set(xlabel="Episode", ylabel="Punishment Rate")
    axes[1].set(xlabel="Episode", ylabel="Verification Rate")
    axes[2].set(xlabel="Episode", ylabel="Mean Reputation Score")
    for a in axes:
        a.legend(fontsize=7)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"fig05_social.{e}", format=e, bbox_inches="tight", dpi=300)
    plt.close(fig)
    n_saved += 1

    # Fig 6
    sty()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for c in MC:
        if c not in R:
            continue
        plot_with_fill(a1, R[c]["coop"],       label=LB[c], color=CL[c], **ST[c])
        plot_with_fill(a2, R[c]["oracle_acc"], label=LB[c], color=CL[c], **ST[c])
    a1.set(xlabel="Episode", ylabel="Cooperation Rate",  ylim=(-0.02, 0.5))
    a2.set(xlabel="Episode", ylabel="Oracle Accuracy",   ylim=(-0.02, 1.05))
    a1.legend(fontsize=7); a2.legend(fontsize=7)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"fig06_coop_oracle.{e}", format=e, bbox_inches="tight", dpi=300)
    plt.close(fig)
    n_saved += 1

    # Fig 7
    sty()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c in MC:
        if c not in R:
            continue
        ax.plot(np.cumsum(np.mean(R[c]["reward"], axis=0)), label=LB[c], color=CL[c], **ST[c])
    ax.set(xlabel="Episode", ylabel="Cumulative Reward")
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useMathText=True)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"fig07_reward.{e}", format=e, bbox_inches="tight", dpi=300)
    plt.close(fig)
    n_saved += 1

    # Fig 8
    mx = max(np.mean(R[c]["res"], axis=0).max() for c in MC if c in R)
    _curve(R, "res", "Active Resources", "fig08_resources", ylim=(0, mx * 1.2))
    n_saved += 1

    # Fig 9 — skip if no ablation data
    if "abl_pr" in R:
        abl  = R["abl_pr"]
        keys = sorted(abl.keys())
        mets = ["truth", "gather", "lie", "mine", "punish"]
        mls  = ["Truth ↑", "Gather ↑", "Lie ↓", "Mine ↓", "Punish"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x  = np.arange(len(mets))
        bw = 0.18
        cm = plt.cm.viridis
        n_last = max(1, abl[keys[0]]["truth"].shape[1] // 5)
        for i, k in enumerate(keys):
            vals = [np.mean(abl[k][m][:, -n_last:]) for m in mets]
            ax.bar(x + (i - len(keys) / 2 + 0.5) * bw, vals, bw,
                   label=f"pun_rew={k}",
                   color=cm(i / max(len(keys) - 1, 1)),
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(mls)
        ax.set(ylabel=f"Rate (last {n_last} ep)")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        for e in ["pdf", "png"]:
            fig.savefig(f"fig09_ablation_punish.{e}", format=e, bbox_inches="tight", dpi=300)
        plt.close(fig)
        n_saved += 1
        print(f"  fig09 generated ({len(keys)}/4 punish_reward values)")
    else:
        print("  fig09 SKIPPED — run forager_ppo.py to complete ablation")

    # Fig 10
    if "DPF" in R and "AC" in R:
        sty()
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        n_last = max(1, R["DPF"]["truth"].shape[1] // 5)
        s = slice(-n_last, None)
        ed, cd = R["DPF"], R["AC"]
        ms = ["truth", "lie", "gather", "mine"]
        ls = ["Truth", "Lie", "Gather", "Mine"]
        xp = np.arange(4)
        ev = [np.mean(ed[m][:, s]) for m in ms]
        cv = [np.mean(cd[m][:, s]) for m in ms]
        axes[0].bar(xp - 0.17, ev, 0.32, label="DPF",
                    color="#5D3A9B", edgecolor="white")
        axes[0].bar(xp + 0.17, cv, 0.32, label="AC",
                    color="#D62728", edgecolor="white")
        axes[0].set_xticks(xp); axes[0].set_xticklabels(ls)
        axes[0].set(ylabel="Rate"); axes[0].legend(fontsize=8)
        plot_with_fill(axes[1], ed["coop"], "DPF", "#5D3A9B", lw=2.5)
        plot_with_fill(axes[1], cd["coop"], "AC",  "#D62728", lw=2, ls="--")
        axes[1].set(xlabel="Episode", ylabel="Cooperation Rate", ylim=(-0.02, 0.5))
        axes[1].legend(fontsize=8)
        plot_with_fill(axes[2], ed["oracle_acc"], "DPF", "#5D3A9B", lw=2.5)
        plot_with_fill(axes[2], cd["oracle_acc"], "AC",  "#D62728", lw=2, ls="--")
        axes[2].set(xlabel="Episode", ylabel="Oracle Accuracy", ylim=(-0.02, 1.05))
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        for e in ["pdf", "png"]:
            fig.savefig(f"fig10_collusion.{e}", format=e, bbox_inches="tight", dpi=300)
        plt.close(fig)
        n_saved += 1

    os.chdir(orig_dir)
    print(f"\n  {n_saved} figures (PDF + PNG) saved → {od}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--outdir",         default="plots/v4_ppo")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    print(f"Loading checkpoints from {args.checkpoint_dir} ...")
    R = load_checkpoints(args.checkpoint_dir, args.seeds)

    avail = [c for c in CONDITIONS if c in R]
    if not avail:
        print("No checkpoints found. Run forager_ppo.py first.")
        raise SystemExit(1)

    print(f"\nGenerating plots → {args.outdir} ...")
    make_plots(R, args.outdir)
