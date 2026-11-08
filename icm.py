"""Independent Chip Model (Malmuth-Harville) with a bitmask DP.

icm(stacks, payouts) -> expected points per player.
P(player i finishes 1st) = s_i / S; then recursively among the rest.
Cost ~ 2^n * n; ~0.05 ms for 6 players, ~0.3 ms for 8 in CPython,
so the agent only calls it when it matters (all-in sized decisions).
"""
from functools import lru_cache


def icm(stacks, payouts):
    n = len(stacks)
    payouts = list(payouts[:n]) + [0.0] * max(0, n - len(payouts))
    total = float(sum(stacks))
    if total <= 0:
        return [0.0] * n
    ev = [0.0] * n
    full = (1 << n) - 1
    prob = {0: 1.0}
    placed_sum = {0: 0.0}
    # process masks in order of popcount (number already placed above)
    frontier = [0]
    for place in range(n):
        nxt = {}
        pay = payouts[place]
        for mask in frontier:
            pm = prob[mask]
            rem = total - placed_sum[mask]
            if rem <= 0:
                continue
            for i in range(n):
                bit = 1 << i
                if mask & bit or stacks[i] <= 0:
                    continue
                p = pm * stacks[i] / rem
                ev[i] += p * pay
                m2 = mask | bit
                if m2 in nxt:
                    prob[m2] += p
                else:
                    nxt[m2] = 1
                    prob[m2] = p
                    placed_sum[m2] = placed_sum[mask] + stacks[i]
        frontier = list(nxt)
        if not frontier:
            break
    return ev


@lru_cache(maxsize=4096)
def icm_cached(stacks, payouts):
    return icm(stacks, payouts)


if __name__ == "__main__":
    import time
    pay = (50, 30, 20, 10, 5, 3, 1, 0)
    st = (5000, 3000, 2000, 1500, 1200, 1000, 800, 500)
    t = time.perf_counter()
    for _ in range(200):
        v = icm(list(st), pay)
    print("8 players: %.3f ms" % ((time.perf_counter() - t) / 200 * 1e3), [round(x, 2) for x in v])
    t = time.perf_counter()
    for _ in range(200):
        v = icm(list(st[:5]), pay)
    print("5 players: %.3f ms" % ((time.perf_counter() - t) / 200 * 1e3), [round(x, 2) for x in v])
