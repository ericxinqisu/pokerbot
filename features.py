"""Hand abstraction: maps a concrete hand to a bucket key in microseconds.

Preflop: suit-isomorphic class (169 classes for 2-card hold'em, generic for more).
Postflop: made-hand class + sub-type + draws + board texture. The equity of each
bucket is learned offline by Monte Carlo (calibrate.py), so the in-game cost is a
hand evaluation plus a dict lookup, well under the 1 ms limit.
"""
from cards import (VARIANT, RANK_CHARS, evaluate, eval_any, category,
                   HIGH, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH)


def preflop_key(hole):
    ranks = sorted((c >> 2 for c in hole), reverse=True)
    if len(hole) == 2:
        a, b = ranks
        if a == b:
            return RANK_CHARS[a] * 2
        suited = (hole[0] & 3) == (hole[1] & 3)
        return RANK_CHARS[a] + RANK_CHARS[b] + ("s" if suited else "o")
    # generic: ranks + sorted suit-count signature (approximate isomorphism)
    sc = [0, 0, 0, 0]
    for c in hole:
        sc[c & 3] += 1
    sig = "".join(str(x) for x in sorted(sc, reverse=True) if x)
    return "".join(RANK_CHARS[r] for r in ranks) + ":" + sig


def _kick(k):
    return 2 if k >= 11 else (1 if k >= 8 else 0)


def _straight_outs(mask, low):
    outs = 0
    for r in range(low, 13):
        if mask >> r & 1:
            continue
        m = mask | (1 << r)
        for top in range(12, low + 3, -1):
            w = 0b11111 << (top - 4)
            if m & w == w and (w >> r) & 1:
                outs += 1
                break
        else:
            wheel = (1 << 12) | sum(1 << (low + i) for i in range(4))
            if m & wheel == wheel and (wheel >> r) & 1:
                outs += 1
    return outs


def postflop_key(hole, board):
    """Returns (fine_key, coarse_key, cat_key)."""
    street = len(board)  # 3, 4, 5 for hold'em
    v = evaluate(hole, board)
    cat = category(v)
    hr = [c >> 2 for c in hole]
    br = [c >> 2 for c in board]
    bset = sorted(set(br), reverse=True)
    bmax = bset[0]
    r0 = (v >> 16) & 15
    r1 = (v >> 12) & 15
    sub = 0
    if cat == HIGH:
        sub = "o%d" % min(2, sum(1 for r in set(hr) if r > bmax))
    elif cat == PAIR:
        if r0 in br and r0 in hr:
            pos = min(2, bset.index(r0))
            k = max([r for r in hr if r != r0] or [0])
            sub = "p%d_k%d" % (pos, _kick(k))
        elif r0 in br:
            sub = "bp_o%d" % min(2, sum(1 for r in set(hr) if r > bmax))
        else:
            sub = "over" if r0 > bmax else "under%d" % min(2, sum(1 for r in bset if r > r0))
    elif cat == TWO_PAIR:
        n_in = (r0 in hr) + (r1 in hr)
        top_in = r0 in hr
        sub = "h%d%s" % (n_in, "t" if top_in else "")
    elif cat == TRIPS:
        if hr.count(r0) >= 2:
            sub = "set"
        elif r0 in hr:
            sub = "trips_k%d" % _kick(max([r for r in hr if r != r0] or [0]))
        else:
            sub = "board"
    elif cat == FLUSH:
        suit_counts = [0] * 4
        for c in board:
            suit_counts[c & 3] += 1
        fs = max(range(4), key=lambda s: suit_counts[s] + sum(1 for c in hole if c & 3 == s))
        hs = [c >> 2 for c in hole if c & 3 == fs]
        sub = "nut" if hs and max(hs) >= 11 else ("hole" if hs else "board")
    elif cat == STRAIGHT:
        sub = "top" if r0 >= max(br) + 1 else "low"
    # board-plays check on river
    if street >= 5 and v == eval_any(board):
        sub = "plays_board"

    # draws
    fd = 0; sd = 0
    if street < sum(VARIANT["board"]) and cat < FLUSH:
        for s in range(4):
            hcnt = sum(1 for c in hole if c & 3 == s)
            bcnt = sum(1 for c in board if c & 3 == s)
            need_h = VARIANT["must_use_hole"] or 1
            if hcnt >= need_h and hcnt + bcnt >= 4 and bcnt <= 5 - need_h:
                fd = 1
        if cat < STRAIGHT:
            low = VARIANT["low_rank"]
            mask = 0
            for r in hr + br:
                mask |= 1 << r
            bmask = 0
            for r in br:
                bmask |= 1 << r
            outs = _straight_outs(mask, low) - _straight_outs(bmask, low)
            sd = 2 if outs >= 2 else (1 if outs == 1 else 0)

    # board texture
    bpaired = int(len(bset) < len(br))
    sc = [0, 0, 0, 0]
    for c in board:
        sc[c & 3] += 1
    bsuit = min(3, max(sc))
    conn = 0
    for top in range(12, 3, -1):
        conn = max(conn, sum(1 for r in bset if top - 4 <= r <= top))
    bconn = min(3, conn)

    fine = (street, cat, sub, fd, sd, bpaired, bsuit, bconn)
    coarse = (street, cat, sub, fd, sd)
    return fine, coarse, (street, cat)
