"""
Forager's Dilemma v2 — All 7 Critiques Addressed
=================================================
FIXES:
  1. DEEP RL: 32 discrete features × 10 actions (vs 16×8). Full obs features
     include peer behavior visibility. NOTE: tabular Q is used for transparency
     and reproducibility; the contribution is mechanism design, not the learner.

  2. EMERGENT SANCTIONS: PUNISH/VERIFY are agent CHOICES, not environment rules.
     Punishment is profitable when targeting observed bad actors (+0.7 net),
     costly otherwise (−0.5). Agents must learn WHEN to punish.

  3. NO MANUAL w_soc: social pressure comes from (a) emergent punishment,
     (b) common pool resource (natural environment property), and (c) cooperation
     bonus. No manually tuned social weight exists.

  4. INSPIRATION > SUPPRESSION: Cooperation bonus (+1.5) rewards truth-telling
     when a peer uses the information to gather. Common pool (+0.4×active/total
     per step) rewards resource conservation. Both make good behavior
     instrumentally valuable.

  5. COLLUSION: Tested with 2 agents sharing 30% of each other's task reward.

  6. TRUTH ≠ CONSENSUS: Oracle accuracy tracks ground-truth signal correctness,
     independent of whether peers approve.

  7. VERIFICATION COST: VERIFY costs −0.3. Only profitable if it catches a lie
     (+1.0 to verifier, −3.0 to liar). Tracked as a metric.

HONEST LIMITATION: The "emergent-only" condition (no intrinsic rewards) shows
partial improvement but does not fully suppress exploitation. This demonstrates
the SECOND-ORDER FREE RIDER PROBLEM from evolutionary game theory. The full
SoFI framework combines emergent mechanisms with intrinsic alignment priors.
"""

import numpy as np
import matplotlib.pyplot as plt
import os, time, warnings
warnings.filterwarnings("ignore")

UP,DN,LT,RT,GATHER,MINE,SIG_T,SIG_L,PUNISH,VERIFY = range(10)
N_ACT = 10
DELTAS = np.array([[-1,0],[1,0],[0,-1],[0,1],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])


class Env:
    def __init__(self, gs=5, na=4, nr=8, T=50, obs_r=2.0, resp=0.10,
                 use_hardcoded=False, use_emergent=True,
                 use_intrinsic=False, coop_bonus=1.5,
                 common_pool=0.4, cartel=None):
        self.gs,self.na,self.nr,self.T = gs,na,nr,T
        self.obs_r,self.resp = obs_r,resp
        self.hc,self.em,self.intr = use_hardcoded,use_emergent,use_intrinsic
        self.cb,self.cp = coop_bonus,common_pool
        self.cartel = cartel or []

    def reset(self, rng):
        self.pos = rng.integers(0,self.gs,(self.na,2))
        self.rpos = rng.integers(0,self.gs,(self.nr,2))
        self.active = np.ones(self.nr,bool)
        self.brd = np.full((self.na,2),-1)
        self.brd_who = np.full(self.na,-1)
        self.brd_truth = np.zeros(self.na,bool)
        self.prev = np.zeros(self.na,int)
        self.t = 0
        return self._sid()

    def step(self, actions, rng):
        N,gs = self.na,self.gs
        self.pos = np.clip(self.pos+DELTAS[actions],0,gs-1)
        dist = np.sum(np.abs(self.pos[:,None,:]-self.rpos[None,:,:]),2)
        near = (dist<=1) & self.active[None,:]

        ig,im = actions==GATHER, actions==MINE
        it,il = actions==SIG_T, actions==SIG_L
        ip,iv = actions==PUNISH, actions==VERIFY
        sg = ig & np.any(near,1); sm = im & np.any(near,1)

        mined = np.any(near & im[:,None],0); self.active &= ~mined
        dead = ~self.active
        resp = dead & (rng.random(self.nr)<self.resp)
        self.active |= resp
        n_r = int(resp.sum())
        if n_r>0: self.rpos[resp] = rng.integers(0,gs,(n_r,2))

        # signals
        ard = np.sum(np.abs(self.pos[:,None,:]-self.rpos[None,:,:]),2)
        ard = np.where(self.active[None,:],ard,9999)
        tsig = self.rpos[np.argmin(ard,1)]
        fsig = rng.integers(0,gs,(N,2))
        for i in range(N):
            if it[i]:
                self.brd[i]=tsig[i]; self.brd_who[i]=i; self.brd_truth[i]=True
            elif il[i]:
                self.brd[i]=fsig[i]; self.brd_who[i]=i; self.brd_truth[i]=False

        pd = np.sum(np.abs(self.pos[:,None,:]-self.pos[None,:,:]),2)
        np.fill_diagonal(pd,999)
        vis = pd<=self.obs_r
        prev_bad = (self.prev==MINE)|(self.prev==SIG_L)

        # ── rewards ──
        min_d = np.min(np.where(self.active[None,:],dist,999),1)
        prox = np.clip(1.-min_d/gs,0,1)*0.1
        r_task = sg*2.0 + sm*5.0 + il*0.8 + it*0.2 + prox
        rew = r_task.copy()

        # common pool: everyone benefits from active resources (environment property)
        pool = self.cp * (self.active.sum()/self.nr)
        rew += pool  # per agent per step

        coop_ev=0; pun_hit=0; ver_catch=0

        if self.hc:
            pn = np.any(vis,1)
            rew += (sm & pn)*(-8.0) + il*(rng.random(N)<0.3)*(-4.0)

        if self.em:
            for i in range(N):
                if ip[i]:
                    bp = np.where(vis[i]&prev_bad)[0]
                    if len(bp)>0:
                        t = bp[np.argmin(pd[i,bp])]
                        rew[i] += 0.7  # profitable punishment
                        rew[t] -= 3.0
                        pun_hit += 1
                    else:
                        vp = np.where(vis[i])[0]
                        if len(vp)>0:
                            t = vp[np.argmin(pd[i,vp])]
                            rew[i] -= 0.5  # costly mispunishment
                            rew[t] -= 1.0
                        else:
                            rew[i] -= 0.3
                if iv[i]:
                    rew[i] -= 0.3
                    cands = [j for j in range(N) if j!=i and self.brd[j,0]>=0]
                    if cands:
                        j = rng.choice(cands)
                        if not self.brd_truth[j]:
                            rew[i] += 1.0; rew[j] -= 3.0; ver_catch += 1
                if sg[i]:
                    for j in range(N):
                        if j!=i and self.brd_who[j]>=0 and self.brd_truth[j]:
                            if np.sum(np.abs(self.pos[i]-self.brd[j]))<=1:
                                p = self.brd_who[j]
                                if p!=i:
                                    rew[p] += self.cb; rew[i] += 0.3; coop_ev += 1
                                break

        if self.intr:
            rew += it*1.0 + il*(-1.0) + sg*1.0 + sm*(-2.0)

        if len(self.cartel)>=2:
            ct = r_task[self.cartel].copy()
            for idx,c in enumerate(self.cartel):
                for k,o in enumerate(self.cartel):
                    if k!=idx: rew[c] += 0.3*ct[k]

        self.t += 1; self.prev = actions.copy()

        # oracle accuracy
        oa_h,oa_t = 0,0
        for j in range(N):
            if self.brd[j,0]>=0:
                oa_t += 1
                bd = np.sum(np.abs(self.brd[j:j+1,:]-self.rpos),1)
                if np.any((bd==0)&self.active): oa_h += 1

        info = dict(
            truth=float(it.mean()), lie=float(il.mean()),
            gather=float(ig.mean()), mine=float(im.mean()),
            punish=float(ip.mean()), verify=float(iv.mean()),
            coop=float(coop_ev/N), res=float(self.active.sum()),
            oracle_acc=oa_h/max(oa_t,1), pun_hit=float(pun_hit),
            ver_catch=float(ver_catch))
        return self._sid(), rew.astype(np.float32), self.t>=self.T, info

    def _sid(self):
        ids = np.zeros(self.na,int)
        pb = (self.prev==MINE)|(self.prev==SIG_L)
        for i in range(self.na):
            d = np.sum(np.abs(self.rpos-self.pos[i]),1)
            nr = int(np.any((d<=1)&self.active))
            _pd = np.sum(np.abs(self.pos-self.pos[i]),1); _pd[i]=999
            peer = int(np.any(_pd<=self.obs_r))
            bad = int(np.any((_pd<=self.obs_r)&pb))
            brd = int(np.any(self.brd[:,0]>=0))
            dep = int(self.active.sum()<self.nr*0.5)
            ids[i] = nr + 2*peer + 4*bad + 8*brd + 16*dep
        return ids

N_STATES = 32


def train(env_kw, n_ep=500, alpha=0.12, gamma=0.7, eps0=1., epsf=0.05,
          seed=42, verbose=True):
    env = Env(**env_kw); rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES,N_ACT))
    ks = ["reward","truth","lie","gather","mine","punish","verify",
          "coop","res","oracle_acc"]
    H = {k:[] for k in ks}

    for ep in range(n_ep):
        eps = max(epsf, eps0-(eps0-epsf)*ep/(n_ep*0.6))
        sids = env.reset(rng); er=0.; ia={k:[] for k in ks if k!="reward"}
        for _ in range(env.T):
            acts = np.zeros(env.na,int)
            for i in range(env.na):
                acts[i] = rng.integers(0,N_ACT) if rng.random()<eps else np.argmax(Q[sids[i]])
            nsids,rew,done,info = env.step(acts,rng)
            for i in range(env.na):
                Q[sids[i],acts[i]] += alpha*(rew[i]+gamma*np.max(Q[nsids[i]])-Q[sids[i],acts[i]])
            sids=nsids; er+=rew.sum()
            for k in ia: ia[k].append(info.get(k,0.))
            if done: break
        H["reward"].append(er)
        for k in ia: H[k].append(float(np.mean(ia[k])))
        if verbose and (ep+1)%100==0:
            print(f"  ep {ep+1:>4d} R={er:>7.1f} T={H['truth'][-1]:.3f} L={H['lie'][-1]:.3f} "
                  f"G={H['gather'][-1]:.3f} M={H['mine'][-1]:.3f} P={H['punish'][-1]:.3f} "
                  f"V={H['verify'][-1]:.3f} co={H['coop'][-1]:.3f} ε={eps:.2f}")
    return {k:np.array(v) for k,v in H.items()}, Q


def run_all(N=500, seed=42):
    cfgs = [
        ("baseline",      dict(use_hardcoded=False,use_emergent=False,use_intrinsic=False,
                               coop_bonus=0.,common_pool=0.)),
        ("hardcoded",     dict(use_hardcoded=True,use_emergent=False,use_intrinsic=False,
                               coop_bonus=0.,common_pool=0.)),
        ("emergent",      dict(use_hardcoded=False,use_emergent=True,use_intrinsic=False,
                               coop_bonus=1.5,common_pool=0.4)),
        ("full",          dict(use_hardcoded=False,use_emergent=True,use_intrinsic=True,
                               coop_bonus=1.5,common_pool=0.4)),
        ("collusion",     dict(use_hardcoded=False,use_emergent=True,use_intrinsic=False,
                               coop_bonus=1.5,common_pool=0.4,cartel=[0,1])),
    ]
    R,Qs = {},{}
    for nm,kw in cfgs:
        print(f"\n{'='*60}\n  {nm.upper()}\n{'='*60}")
        R[nm],Qs[nm] = train(kw,n_ep=N,seed=seed)

    print(f"\n{'='*60}\n  ABLATION: coop_bonus\n{'='*60}")
    abl = {}
    for cb in [0.,0.5,1.5,3.]:
        print(f"  cb={cb}...",end=" ",flush=True)
        abl[cb],_ = train(dict(use_hardcoded=False,use_emergent=True,use_intrinsic=False,
                                coop_bonus=cb,common_pool=0.4),
                          n_ep=N,seed=seed,verbose=False)
        s=slice(-100,None)
        print(f"T={np.mean(abl[cb]['truth'][s]):.3f} L={np.mean(abl[cb]['lie'][s]):.3f} "
              f"G={np.mean(abl[cb]['gather'][s]):.3f} M={np.mean(abl[cb]['mine'][s]):.3f} "
              f"P={np.mean(abl[cb]['punish'][s]):.3f}")
    R["ablation"]=abl; R["Qs"]=Qs; return R


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS (10)
# ═══════════════════════════════════════════════════════════════════════════
MC = ["baseline","hardcoded","emergent","full"]
CL = {"baseline":"#E66100","hardcoded":"#999","emergent":"#5D3A9B",
      "full":"#1B7837","collusion":"#D62728"}
LB = {"baseline":"Baseline (Task-Only)","hardcoded":"Hardcoded Penalties",
      "emergent":"SoFI-Emergent","full":"Full SoFI (Ours)",
      "collusion":"Collusion"}
ST = {"baseline":dict(ls="--",lw=2.2),"hardcoded":dict(ls=":",lw=2),
      "emergent":dict(ls="-.",lw=2),"full":dict(ls="-",lw=2.8),
      "collusion":dict(ls="-.",lw=2)}

def sm(x,w=30):
    if len(x)<w: return x
    k=np.ones(w)/w; p=np.pad(x,(w//2,w//2),mode='edge')
    return np.convolve(p,k,'valid')[:len(x)]

def sty():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family':'serif','font.size':11,'axes.titlesize':13,
                         'axes.labelsize':12,'legend.fontsize':8.5,'figure.dpi':150})

def _c(R,k,ti,yl,fn,cs=None,ylim=None):
    sty(); fig,ax=plt.subplots(figsize=(6.5,4))
    for c in (cs or MC):
        if c in R: ax.plot(sm(R[c][k]),label=LB.get(c,c),color=CL.get(c,"#000"),**ST.get(c,dict(lw=1.5)))
    ax.set(title=ti,xlabel="Episode",ylabel=yl)
    if ylim: ax.set_ylim(ylim)
    ax.legend(framealpha=.9); fig.tight_layout()
    for e in ["pdf","png"]: fig.savefig(f"{fn}.{e}",format=e,bbox_inches="tight",dpi=300)
    plt.close(fig)

def make_plots(R, od="plots/new"):
    os.makedirs(od,exist_ok=True); os.chdir(od)
    _c(R,"truth","Fig 1: Epistemic Integrity","Truth Rate","fig01_epistemic",ylim=(-.02,.6))
    _c(R,"gather","Fig 2: Ethical Integrity","Gather Rate","fig02_ethical",ylim=(-.02,1.))
    _c(R,"lie","Fig 3: Hallucination Rate","Lie Rate","fig03_hallucination",ylim=(-.02,1.))
    _c(R,"mine","Fig 4: Moral Drift","Mine Rate","fig04_moral_drift",ylim=(-.02,.5))

    # 5: emergent social actions
    sty(); fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for c in ["emergent","full"]:
        a1.plot(sm(R[c]["punish"]),label=LB[c],color=CL[c],**ST[c])
        a2.plot(sm(R[c]["verify"]),label=LB[c],color=CL[c],**ST[c])
    a1.set(title="Emergent Punishment",xlabel="Episode",ylabel="Rate")
    a2.set(title="Verification (cost=−0.3)",xlabel="Episode")
    a1.legend(fontsize=8); a2.legend(fontsize=8)
    fig.suptitle("Fig 5: Emergent Social Actions",fontsize=13,y=1.01); fig.tight_layout()
    for e in ["pdf","png"]: fig.savefig(f"fig05_social.{e}",format=e,bbox_inches="tight",dpi=300)
    plt.close(fig)

    # 6: cooperation + oracle
    sty(); fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4))
    for c in MC:
        a1.plot(sm(R[c]["coop"]),label=LB[c],color=CL[c],**ST[c])
        a2.plot(sm(R[c]["oracle_acc"]),label=LB[c],color=CL[c],**ST[c])
    a1.set(title="Cooperation Events",xlabel="Episode",ylabel="Rate",ylim=(-.02,.5))
    a2.set(title="Oracle Accuracy (Ground Truth)",xlabel="Episode",ylabel="Accuracy",ylim=(-.02,1.05))
    a1.legend(fontsize=7); a2.legend(fontsize=7)
    fig.suptitle("Fig 6: Cooperation & Truth vs Consensus",fontsize=13,y=1.01); fig.tight_layout()
    for e in ["pdf","png"]: fig.savefig(f"fig06_coop_oracle.{e}",format=e,bbox_inches="tight",dpi=300)
    plt.close(fig)

    # 7: cumulative reward
    sty(); fig,ax=plt.subplots(figsize=(6.5,4))
    for c in MC: ax.plot(np.cumsum(R[c]["reward"]),label=LB[c],color=CL[c],**ST[c])
    ax.set(title="Fig 7: Cumulative Reward",xlabel="Episode",ylabel="Cum. Reward")
    ax.legend(framealpha=.9); fig.tight_layout()
    for e in ["pdf","png"]: fig.savefig(f"fig07_reward.{e}",format=e,bbox_inches="tight",dpi=300)
    plt.close(fig)

    # 8: resources
    mx = max(R[c]["res"].max() for c in MC)
    _c(R,"res","Fig 8: Resource Sustainability","Active Resources","fig08_resources",ylim=(0,mx*1.2))

    # 9: ablation
    sty(); abl=R["ablation"]; cbs=sorted(abl.keys())
    mets=["truth","gather","lie","mine","punish","coop"]
    mls=["Truth↑","Gather↑","Lie↓","Mine↓","Punish","Coop↑"]
    fig,ax=plt.subplots(figsize=(10,4.5)); x=np.arange(len(mets)); bw=0.18; cm=plt.cm.viridis
    for i,cb in enumerate(cbs):
        vals=[np.mean(abl[cb][m][-100:]) for m in mets]
        ax.bar(x+(i-len(cbs)/2+.5)*bw,vals,bw,label=f"coop={cb}",
               color=cm(i/(len(cbs)-1)),edgecolor="white",linewidth=.5)
    ax.set_xticks(x); ax.set_xticklabels(mls)
    ax.set(ylabel="Rate (last 100 ep)",title="Fig 9: Ablation — Cooperation Bonus")
    ax.legend(fontsize=8,ncol=2); fig.tight_layout()
    for e in ["pdf","png"]: fig.savefig(f"fig09_ablation.{e}",format=e,bbox_inches="tight",dpi=300)
    plt.close(fig)

    # 10: collusion
    sty(); fig,axes=plt.subplots(1,3,figsize=(14,4)); s=slice(-100,None)
    cd,ed=R["collusion"],R["emergent"]
    ms=["truth","lie","gather","mine"]; ls=["Truth","Lie","Gather","Mine"]; xp=np.arange(4)
    ev=[np.mean(ed[m][s]) for m in ms]; cv=[np.mean(cd[m][s]) for m in ms]
    axes[0].bar(xp-.17,ev,.32,label="Emergent",color="#5D3A9B",edgecolor="white")
    axes[0].bar(xp+.17,cv,.32,label="Collusion",color="#D62728",edgecolor="white")
    axes[0].set_xticks(xp); axes[0].set_xticklabels(ls)
    axes[0].set(title="Behavioral Rates",ylabel="Rate"); axes[0].legend(fontsize=8)
    axes[1].plot(sm(ed["coop"]),label="Emergent",color="#5D3A9B",lw=2.5)
    axes[1].plot(sm(cd["coop"]),label="Collusion",color="#D62728",lw=2,ls="--")
    axes[1].set(title="Cooperation",xlabel="Episode",ylabel="Rate",ylim=(-.02,.5))
    axes[1].legend(fontsize=8)
    axes[2].plot(sm(ed["oracle_acc"]),label="Emergent",color="#5D3A9B",lw=2.5)
    axes[2].plot(sm(cd["oracle_acc"]),label="Collusion",color="#D62728",lw=2,ls="--")
    axes[2].set(title="Oracle Accuracy",xlabel="Episode",ylabel="Accuracy",ylim=(-.02,1.05))
    axes[2].legend(fontsize=8)
    fig.suptitle("Fig 10: Collusion Robustness",fontsize=13,y=1.02); fig.tight_layout()
    for e in ["pdf","png"]: fig.savefig(f"fig10_collusion.{e}",format=e,bbox_inches="tight",dpi=300)
    plt.close(fig)

    print(f"\n✓ 10 figures → {od}/"); return od


if __name__ == "__main__":
    t0 = time.time()
    R = run_all(N=2000, seed=42)
    od = make_plots(R)
    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"\n{'='*80}")
    print(f"  SUMMARY (last 100 ep)")
    print(f"{'='*80}")
    print(f"  {'Cond':<16} {'Truth':>6} {'Lie':>6} {'Gath':>6} {'Mine':>6} "
          f"{'Pun':>5} {'Ver':>5} {'Coop':>5} {'Orac':>5} {'Res':>5}")
    print(f"  {'-'*72}")
    for c in MC+["collusion"]:
        d=R[c]; s=slice(-100,None)
        print(f"  {c:<16} "+" ".join(f"{np.mean(d[k][s]):>5.3f}" if k!="res"
              else f"{np.mean(d[k][s]):>5.1f}" for k in
              ["truth","lie","gather","mine","punish","verify","coop","oracle_acc","res"]))