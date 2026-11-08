"""Cards, deck config and a fast pure-Python hand evaluator.

Cards are ints: rank = c // 4 (0 = lowest rank in the deck), suit = c % 4.
The variant is unknown until event day, so everything reads from VARIANT.
Change the dict (or call set_variant) to match what organizers announce.
"""
import random

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"

VARIANT = {
    "name": "nlhe",
    "hole": 2,             # hole cards per player
    "board": [0, 3, 1, 1],  # cards dealt on preflop/flop/turn/river
    "low_rank": 0,          # 0 = deuce. Short deck (6+) -> 4
    "must_use_hole": None,  # None = best 5 of any. Omaha -> 2 (exactly 2 hole + 3 board)
    "short_deck": False,    # True: flush beats full house, A-6-7-8-9 straight
}


def set_variant(**kw):
    VARIANT.update(kw)


def deck():
    lo = VARIANT["low_rank"]
    return [r * 4 + s for r in range(lo, 13) for s in range(4)]


def parse(s):
    """'AsKd' -> [cards]."""
    s = s.replace(" ", "")
    return [RANK_CHARS.index(s[i]) * 4 + SUIT_CHARS.index(s[i + 1]) for i in range(0, len(s), 2)]


def show(cards):
    return " ".join(RANK_CHARS[c // 4] + SUIT_CHARS[c % 4] for c in cards)


# ---------------------------------------------------------------- evaluator
# Categories (higher is better)
HIGH, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CAT_NAMES = ["high", "pair", "two_pair", "trips", "straight", "flush", "full_house", "quads", "straight_flush"]


def _straight_high(mask, low_rank):
    """Highest straight top-rank in a 13-bit rank mask, or -1."""
    # wheel: ace plays low. Standard A2345, short deck A6789
    for top in range(12, low_rank + 3, -1):
        m = 0b11111 << (top - 4)
        if mask & m == m:
            return top
    wheel = (1 << 12) | sum(1 << (low_rank + i) for i in range(4))
    if mask & wheel == wheel:
        return low_rank + 3
    return -1


def _pack(cat, ranks):
    v = cat
    for r in ranks[:5]:
        v = (v << 4) | r
    for _ in range(5 - min(5, len(ranks))):
        v <<= 4
    return v


def eval_any(cards):
    """Best 5-card value from 5..7+ cards. Comparable int, higher wins."""
    short = VARIANT["short_deck"]
    low = VARIANT["low_rank"]
    counts = [0] * 13
    suits = [0, 0, 0, 0]
    smask = [0, 0, 0, 0]
    for c in cards:
        r = c >> 2
        s = c & 3
        counts[r] += 1
        suits[s] += 1
        smask[s] |= 1 << r
    flush_suit = -1
    for s in range(4):
        if suits[s] >= 5:
            flush_suit = s
    if flush_suit >= 0:
        sh = _straight_high(smask[flush_suit], low)
        if sh >= 0:
            return _pack(STRAIGHT_FLUSH, [sh]) + ((1 << 41) if short else 0)
    quads = []; trips = []; pairs = []; singles = []
    mask = 0
    for r in range(12, -1, -1):
        n = counts[r]
        if n:
            mask |= 1 << r
            if n == 4: quads.append(r)
            elif n == 3: trips.append(r)
            elif n == 2: pairs.append(r)
            else: singles.append(r)
    if quads:
        q = quads[0]
        k = max([r for r in range(13) if counts[r] and r != q])
        return _pack(QUADS, [q, k])
    fh = None
    if trips and (len(trips) > 1 or pairs):
        t = trips[0]
        p = max(trips[1:] + pairs)
        fh = _pack(FULL_HOUSE, [t, p])
    fl = None
    if flush_suit >= 0:
        fr = [r for r in range(12, -1, -1) if smask[flush_suit] >> r & 1][:5]
        fl = _pack(FLUSH, fr)
    if short:
        if fl: return fl + (1 << 40)  # flush above full house in short deck
        if fh: return fh
    else:
        if fh: return fh
        if fl: return fl
    sh = _straight_high(mask, low)
    if sh >= 0:
        return _pack(STRAIGHT, [sh])
    if trips:
        t = trips[0]
        ks = [r for r in range(12, -1, -1) if counts[r] and r != t][:2]
        return _pack(TRIPS, [t] + ks)
    if len(pairs) >= 2:
        a, b = pairs[0], pairs[1]
        k = max([r for r in range(13) if counts[r] and r != a and r != b])
        return _pack(TWO_PAIR, [a, b, k])
    if pairs:
        p = pairs[0]
        ks = [r for r in range(12, -1, -1) if counts[r] and r != p][:3]
        return _pack(PAIR, [p] + ks)
    return _pack(HIGH, singles[:5])


def category(value):
    v = value >> 20
    return min(v, 8) if v < 16 else 5  # short-deck flush offset folds back to FLUSH


def evaluate(hole, board):
    """Respects Omaha-style 'must use N hole cards' if configured."""
    need = VARIANT["must_use_hole"]
    if not need:
        return eval_any(list(hole) + list(board))
    from itertools import combinations
    best = -1
    for h in combinations(hole, need):
        for b in combinations(board, 5 - need):
            v = eval_any(list(h) + list(b))
            if v > best:
                best = v
    return best


def total_board():
    return sum(VARIANT["board"])


def showdown_equity(hero, opps, board, rng=random, deck_cards=None):
    """One random runout: returns hero share of pot (1, 0, or split)."""
    used = set(hero) | set(board)
    for o in opps:
        used |= set(o)
    rest = [c for c in (deck_cards or deck()) if c not in used]
    need = total_board() - len(board)
    run = list(board) + rng.sample(rest, need)
    hv = evaluate(hero, run)
    best = hv; ties = 1; hero_best = True
    for o in opps:
        v = evaluate(o, run)
        if v > best:
            best = v; hero_best = False; ties = 1
        elif v == best:
            ties += 1
    return (1.0 / ties) if (hero_best and hv == best) else 0.0
