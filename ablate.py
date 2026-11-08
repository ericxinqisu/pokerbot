"""Ablations: switch off one layer at a time.  python ablate.py"""
import random, pickle, statistics, sys
from multiprocessing import Pool
import engine, agent
from bench import field

def run(args):
    mode, fk, seed, n = args
    if mode == "no_icm":
        agent.ICM_MAX_PLAYERS = 0
    if mode.startswith("set:"):  # e.g. set:PUSH_FOLD_BB=10
        k, v = mode[4:].split("=")
        setattr(agent, k, type(getattr(agent, k))(v))
    if mode.startswith("icm_from_"):
        agent.ICM_FROM_HAND = int(mode.split("_")[-1])
    if mode == "no_model":
        agent.Agent.on_hand_end = lambda self, info: None
    rng = random.Random(seed)
    pct = pickle.load(open("tables.pkl", "rb"))["pre_pct"]
    pts = []
    for _ in range(n):
        hero = agent.Agent(seed=rng.random())
        bots = field(pct, rng, fk) + [hero]; rng.shuffle(bots)
        res, _, _ = engine.run_table(bots, rng)
        pts.append(res[bots.index(hero)])
    return pts

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    for fk in ("mixed", "charts"):
        for mode in sys.argv[2].split(",") if len(sys.argv) > 2 else ("full", "no_icm", "no_model"):
            with Pool(2) as p:
                pts = sum(p.map(run, [(mode, fk, 77 + i, n // 2) for i in range(2)]), [])
            print("%-7s %-9s %.2f +- %.2f" % (fk, mode, statistics.mean(pts), statistics.stdev(pts) / len(pts) ** .5))
