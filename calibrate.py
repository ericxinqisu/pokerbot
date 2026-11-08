"""Offline Monte Carlo calibration -> tables.pkl (the bot's 'blueprint').

Run on event day as soon as the variant is announced (edit cards.VARIANT first):
    python calibrate.py --fast      # ~2-4 min on a laptop
    python calibrate.py             # better tables, ~10 min
Outputs:
  pre_eq[n][key]       preflop equity vs n random opponents
  pre_pct[key]         strength percentile (0 = best) by combo share
  pre_vs_top[x][key]   HU equity vs an opponent holding a top-x% range
  post_eq[n][key]      postflop bucket equity vs n random opponents
  matrix.npy           169x169 class-vs-class equity (2-card hold'em only, for push/fold CFR)
"""
import argparse, os, pickle, random, time
from collections import defaultdict
from multiprocessing import Pool

import cards
from cards import deck, evaluate, total_board, VARIANT
from features import preflop_key, postflop_key

OPP_NS = (1, 2, 3, 5, 7)
POST_NS = (1, 2, 3, 5)
TOP_X = (2, 5, 8, 12, 16, 20, 25, 30, 40, 50, 65, 80, 100)
HERE = os.path.dirname(os.path.abspath(__file__))


def _showdown_shares(hero, opps, run):
    """Hero share vs first k opps for every k in OPP_NS, one runout."""
    hv = evaluate(hero, run)
    out = {}
    best = hv; ties = 1; alive = True
    for i, o in enumerate(opps, 1):
        v = evaluate(o, run)
        if v > best:
            best = v; alive = False
        elif v == best and alive:
            ties += 1
        if i in OPP_NS or i in POST_NS:
            out[i] = (1.0 / ties) if alive else 0.0
    return out


def _pre_worker(args):
    n, seed, variant = args
    VARIANT.update(variant)
    rng = random.Random(seed)
    d = deck(); H = VARIANT["hole"]; B = total_board(); maxo = max(OPP_NS)
    s = defaultdict(lambda: [0.0] * (maxo + 1)); c = defaultdict(int)
    for _ in range(n):
        cs = rng.sample(d, H * (maxo + 1) + B)
        hero = cs[:H]
        opps = [cs[H * (i + 1):H * (i + 2)] for i in range(maxo)]
        run = cs[H * (maxo + 1):]
        k = preflop_key(hero)
        c[k] += 1
        acc = s[k]
        for i, v in _showdown_shares(hero, opps, run).items():
            acc[i] += v
    return dict(s), dict(c)


def _top_worker(args):
    n, seed, variant, pct, x = args
    VARIANT.update(variant)
    rng = random.Random(seed)
    d = deck(); H = VARIANT["hole"]; B = total_board()
    s = defaultdict(float); c = defaultdict(int)
    lim = x / 100.0
    for _ in range(n):
        hero = rng.sample(d, H)
        hs = set(hero)
        rest0 = [q for q in d if q not in hs]
        opp = None
        for _t in range(60):
            cand = rng.sample(rest0, H)
            if pct.get(preflop_key(cand), 1.0) <= lim:
                opp = cand; break
        if opp is None:
            continue
        os_ = set(opp)
        run = rng.sample([q for q in rest0 if q not in os_], B)
        hv = evaluate(hero, run); ov = evaluate(opp, run)
        k = preflop_key(hero)
        s[k] += 1.0 if hv > ov else (0.5 if hv == ov else 0.0)
        c[k] += 1
    return x, dict(s), dict(c)


def _post_worker(args):
    n, seed, variant, street_cards = args
    VARIANT.update(variant)
    rng = random.Random(seed)
    d = deck(); H = VARIANT["hole"]; B = total_board(); maxo = max(POST_NS)
    tabs = [defaultdict(lambda: [0.0] * (maxo + 1)) for _ in range(3)]
    cnts = [defaultdict(int) for _ in range(3)]
    for _ in range(n):
        cs = rng.sample(d, H * (maxo + 1) + B)
        hero = cs[:H]
        opps = [cs[H * (i + 1):H * (i + 2)] for i in range(maxo)]
        run = cs[H * (maxo + 1):]
        keys = postflop_key(hero, run[:street_cards])
        res = _showdown_shares(hero, opps, run)
        for t in range(3):
            cnts[t][keys[t]] += 1
            acc = tabs[t][keys[t]]
            for i in POST_NS:
                acc[i] += res[i]
    return [dict(t) for t in tabs], [dict(c) for c in cnts]


def _merge(results):
    S = defaultdict(lambda: None); C = defaultdict(int)
    for s, c in results:
        for k, v in s.items():
            if S[k] is None:
                S[k] = list(v)
            else:
                S[k] = [a + b for a, b in zip(S[k], v)]
        for k, v in c.items():
            C[k] += v
    return S, C


def _matrix_worker(args):
    rows, classes, combos, samples, seed, variant = args
    VARIANT.update(variant)
    rng = random.Random(seed)
    d = deck(); B = total_board()
    out = []
    for i in rows:
        row = []
        for j in range(len(classes)):
            if j < i:
                row.append(None); continue
            w = 0.0; n = 0
            ci = combos[classes[i]]; cj = combos[classes[j]]
            tries = 0
            while n < samples and tries < samples * 20:
                tries += 1
                a = rng.choice(ci); b = rng.choice(cj)
                if set(a) & set(b):
                    continue
                used = set(a) | set(b)
                run = rng.sample([q for q in d if q not in used], B)
                va = evaluate(a, run); vb = evaluate(b, run)
                w += 1.0 if va > vb else (0.5 if va == vb else 0.0)
                n += 1
            row.append(w / max(1, n))
        out.append((i, row))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--no-matrix", action="store_true")
    a = ap.parse_args()
    scale = 0.35 if a.fast else 1.0
    variant = dict(VARIANT)
    P = a.procs
    t0 = time.time()
    tables = {"variant": variant}
    with Pool(P) as pool:
        # 1) preflop vs n random
        n = int(600_000 * scale) // P
        S, C = _merge(pool.map(_pre_worker, [(n, 11 + i, variant) for i in range(P)]))
        pre_eq = {k: {key: S[key][k] / C[key] for key in C} for k in OPP_NS}
        tables["pre_eq"] = pre_eq
        order = sorted(C, key=lambda k: -pre_eq[1][k])
        tot = sum(C.values()); cum = 0.0; pct = {}
        for key in order:
            cum += C[key]
            pct[key] = cum / tot
        tables["pre_pct"] = pct
        tables["pre_order"] = order
        print("preflop done %.0fs, %d classes" % (time.time() - t0, len(C)))

        # 2) HU equity vs top-x% ranges
        n = int(160_000 * scale) // P
        jobs = [(n, 101 + i * 31 + x, variant, pct, x) for x in TOP_X for i in range(P)]
        vs = defaultdict(lambda: [defaultdict(float), defaultdict(int)])
        for x, s, c in pool.map(_top_worker, jobs):
            for k, v in s.items():
                vs[x][0][k] += v
            for k, v in c.items():
                vs[x][1][k] += v
        pre_vs_top = {}
        for x in TOP_X:
            s, c = vs[x]
            pre_vs_top[x] = {k: s[k] / c[k] for k in c if c[k] >= 8}
            for k in order:  # fill sparse classes from vs-random
                pre_vs_top[x].setdefault(k, pre_eq[1][k])
        tables["pre_vs_top"] = pre_vs_top
        print("vs-top done %.0fs" % (time.time() - t0))

        # 3) postflop buckets
        post = {k: {} for k in POST_NS}
        post_n = {}
        streets = []
        acc = 0
        for add in VARIANT["board"]:
            acc += add
            if acc >= 3 and add:
                streets.append(acc)
        for sc in streets:
            n = int(450_000 * scale) // P
            res = pool.map(_post_worker, [(n, 1000 + sc * 7 + i, variant, sc) for i in range(P)])
            for t in range(3):
                S, C = _merge([(r[0][t], r[1][t]) for r in res])
                for key, cnt in C.items():
                    post_n[key] = cnt
                    for k in POST_NS:
                        post[k][key] = S[key][k] / cnt
            print("street %d done %.0fs" % (sc, time.time() - t0))
        tables["post_eq"] = post
        tables["post_n"] = post_n

        # 4) class-vs-class matrix for push/fold CFR
        if VARIANT["hole"] == 2 and not a.no_matrix:
            import numpy as np
            from itertools import combinations
            combos = defaultdict(list)
            for h in combinations(deck(), 2):
                combos[preflop_key(list(h))].append(h)
            classes = order
            samples = int(220 * scale) + 40
            rows = list(range(len(classes)))
            chunks = [rows[i::P * 4] for i in range(P * 4)]
            M = np.zeros((len(classes), len(classes)))
            for part in pool.map(_matrix_worker, [(ch, classes, dict(combos), samples, 7 + i, variant) for i, ch in enumerate(chunks)]):
                for i, row in part:
                    for j, v in enumerate(row):
                        if v is not None:
                            M[i, j] = v; M[j, i] = 1 - v
            for i in range(len(classes)):
                M[i, i] = 0.5
            # combo-pair weights with card removal
            W = np.zeros_like(M)
            for i, ki in enumerate(classes):
                for j, kj in enumerate(classes):
                    W[i, j] = sum(1 for x in combos[ki] for y in combos[kj] if not set(x) & set(y))
            np.save(os.path.join(HERE, "matrix.npy"), M)
            np.save(os.path.join(HERE, "matrix_w.npy"), W)
            tables["matrix_classes"] = classes
            print("matrix done %.0fs" % (time.time() - t0))

    with open(os.path.join(HERE, "tables.pkl"), "wb") as f:
        pickle.dump(tables, f)
    print("wrote tables.pkl in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
