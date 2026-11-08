"""Vanilla CFR on Kuhn poker (Neller & Lanctot tutorial, section 3).
Sanity check: player 1's game value converges to -1/18 = -0.0556.
    python kuhn_cfr.py
"""
import random

PASS, BET = 0, 1
nodes = {}


class Node:
    def __init__(self):
        self.regret = [0.0, 0.0]
        self.strat_sum = [0.0, 0.0]

    def strategy(self, weight):
        s = [max(r, 0.0) for r in self.regret]
        tot = sum(s)
        s = [x / tot for x in s] if tot > 0 else [0.5, 0.5]
        for a in range(2):
            self.strat_sum[a] += weight * s[a]
        return s

    def average(self):
        tot = sum(self.strat_sum)
        return [x / tot for x in self.strat_sum] if tot > 0 else [0.5, 0.5]


def cfr(cards, history, p0, p1):
    plays = len(history)
    player = plays % 2
    if plays > 1:
        terminal_pass = history[-1] == "p"
        double_bet = history[-2:] == "bb"
        higher = cards[player] > cards[1 - player]
        if terminal_pass:
            if history == "pp":
                return 1 if higher else -1
            return 1
        if double_bet:
            return 2 if higher else -2
    key = str(cards[player]) + history
    node = nodes.setdefault(key, Node())
    strat = node.strategy(p0 if player == 0 else p1)
    util = [0.0, 0.0]
    node_util = 0.0
    for a, ch in enumerate("pb"):
        if player == 0:
            util[a] = -cfr(cards, history + ch, p0 * strat[a], p1)
        else:
            util[a] = -cfr(cards, history + ch, p0, p1 * strat[a])
        node_util += strat[a] * util[a]
    cf_reach = p1 if player == 0 else p0
    for a in range(2):
        node.regret[a] += cf_reach * (util[a] - node_util)
    return node_util


def main(iters=200_000):
    cards = [1, 2, 3]
    total = 0.0
    rng = random.Random(0)
    for _ in range(iters):
        rng.shuffle(cards)
        total += cfr(cards, "", 1, 1)
    print("average game value for player 1: %.4f (theory -1/18 = -0.0556)" % (total / iters))
    for k in sorted(nodes):
        print("%-4s bet prob %.3f" % (k, nodes[k].average()[1]))


if __name__ == "__main__":
    main()
