"""Heads-up push/fold equilibrium by CFR+ (regret matching+ with linear averaging).

This is the papers' core idea (Zinkevich 2007; Tammelin's CFR+ as used for
Cepheus) applied to the one subgame that matters most in a fast-blind SNG:
short stacks. Small-blind player either shoves S big blinds or folds; big
blind calls or folds. Card removal is handled by combo-pair weights W.

    python pushfold_cfr.py   -> pushfold.pkl (charts for S = 1..30 BB)
"""
import os, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STACKS = [x / 2 for x in range(2, 61)]  # 1.0 .. 30.0 BB


def _rm_plus(r_a, r_b, ga, gb):
    """Regret matching+: add instantaneous regrets, clip at zero, return P(action a)."""
    r_a = np.maximum(r_a + ga, 0); r_b = np.maximum(r_b + gb, 0)
    tot = r_a + r_b
    prob = np.where(tot > 0, r_a / np.where(tot > 0, tot, 1), 0.5)
    return r_a, r_b, prob


def solve(M, W, S, iters=2000):
    """M[i,j] = equity of SB class i vs BB class j; W = combo-pair weights.
    Returns average push prob per SB class and call prob per BB class."""
    n = M.shape[0]
    sb_called = M * 2 * S - S          # SB net (BB units) when shove is called
    bb_called = (1 - M) * 2 * S - S    # BB net when calling
    rp = np.zeros(n); rf = np.zeros(n); rc = np.zeros(n); rcf = np.zeros(n)
    p = np.full(n, 0.5); c = np.full(n, 0.5)
    sp = np.zeros(n); sc = np.zeros(n); wsum = 0.0
    Wsb = W.sum(1)
    for t in range(1, iters + 1):
        u_push = (W * ((1 - c)[None, :] * 1.0 + c[None, :] * sb_called)).sum(1)
        u_fold = -0.5 * Wsb
        u = p * u_push + (1 - p) * u_fold
        rp, rf, p = _rm_plus(rp, rf, u_push - u, u_fold - u)
        wp = W * p[:, None]
        u_call = (wp * bb_called).sum(0)
        u_bf = -1.0 * wp.sum(0)
        ub = c * u_call + (1 - c) * u_bf
        rc, rcf, c = _rm_plus(rc, rcf, u_call - ub, u_bf - ub)
        sp += t * p; sc += t * c; wsum += t   # linear averaging (CFR+ style)
    return sp / wsum, sc / wsum


def exploitability(M, W, S, p, c):
    """Best-response gains for each side, in BB per hand (should be ~0)."""
    Z = W.sum()
    pay_called = M * 2 * S - S
    bb_call = (1 - M) * 2 * S - S
    u_push = (W * ((1 - c)[None, :] * 1.0 + c[None, :] * pay_called)).sum(1)
    u_fold = -0.5 * W.sum(1)
    v_sb = (p * u_push + (1 - p) * u_fold).sum() / Z
    br_sb = np.maximum(u_push, u_fold).sum() / Z
    wp = W * p[:, None]
    u_call = (wp * bb_call).sum(0); u_bf = -1.0 * wp.sum(0)
    u_bb_nopush = (W * (1 - p)[:, None] * 0.5).sum(0)  # BB wins SB's 0.5 when SB folds
    v_bb = (c * u_call + (1 - c) * u_bf + u_bb_nopush).sum() / Z
    br_bb = (np.maximum(u_call, u_bf) + u_bb_nopush).sum() / Z
    return br_sb - v_sb, br_bb - v_bb, v_sb


def main():
    with open(os.path.join(HERE, "tables.pkl"), "rb") as f:
        tables = pickle.load(f)
    classes = tables["matrix_classes"]
    M = np.load(os.path.join(HERE, "matrix.npy"))
    W = np.load(os.path.join(HERE, "matrix_w.npy"))
    out = {"classes": classes, "stacks": STACKS, "push": {}, "call": {},
           "push_freq": {}, "call_freq": {}, "eq_vs_call": {}, "eq_vs_push": {}}
    wcls = W.sum(1) / W.sum()
    for S in STACKS:
        p, c = solve(M, W, S)
        e1, e2, v = exploitability(M, W, S, p, c)
        out["push"][S] = dict(zip(classes, p.round(3)))
        out["call"][S] = dict(zip(classes, c.round(3)))
        out["push_freq"][S] = float((wcls * p).sum())
        out["call_freq"][S] = float((wcls * c).sum())
        wc = W * c[None, :]
        out["eq_vs_call"][S] = dict(zip(classes, ((wc * M).sum(1) / np.maximum(wc.sum(1), 1e-9)).round(4)))
        wpush = W * p[:, None]
        out["eq_vs_push"][S] = dict(zip(classes, ((wpush * (1 - M)).sum(0) / np.maximum(wpush.sum(0), 1e-9)).round(4)))
        if S in (2, 5, 10, 15, 20, 30):
            print("S=%4.1fBB push %4.1f%% call %4.1f%%  SB value %+.3f BB  exploitability %.1e / %.1e BB"
                  % (S, 100 * out["push_freq"][S], 100 * out["call_freq"][S], v, e1, e2))
    with open(os.path.join(HERE, "pushfold.pkl"), "wb") as f:
        pickle.dump(out, f)
    print("wrote pushfold.pkl")


if __name__ == "__main__":
    main()
