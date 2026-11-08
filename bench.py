"""Benchmark: average points per table vs a field of stand-in bots + decision timing.
    python bench.py --tables 400
"""
import argparse, os, pickle, random, statistics, time
from multiprocessing import Pool

import engine
from agent import Agent
from bots import RandomBot, CallingStation, Maniac, ChartBot, Nit, ShortShover

HERE = os.path.dirname(os.path.abspath(__file__))


def field(pct, rng, kind="mixed"):
    if kind == "charts":
        return [ChartBot(pct, open_top=rng.uniform(0.1, 0.35), call_top=rng.uniform(0.06, 0.2),
                         seed=rng.random(), aggro=rng.uniform(0.3, 0.9)) for _ in range(4)] + \
               [ShortShover(pct, seed=rng.random()) for _ in range(2)] + [Nit(pct, seed=rng.random())]
    if kind == "self":
        return [Agent(seed=rng.random()) for _ in range(7)]
    return [ChartBot(pct, open_top=0.25, call_top=0.15, seed=rng.random()),
            ChartBot(pct, open_top=0.15, call_top=0.10, seed=rng.random(), aggro=0.4),
            Nit(pct, seed=rng.random()), CallingStation(), Maniac(),
            ShortShover(pct, seed=rng.random()), RandomBot(rng.random())]


def worker(args):
    n, seed, hero_kind, fkind = args
    rng = random.Random(seed)
    pct = pickle.load(open(os.path.join(HERE, "tables.pkl"), "rb"))["pre_pct"]
    pts = []; places = []; times = []; hands = []
    for t in range(n):
        hero = Agent(seed=rng.random()) if hero_kind == "agent" else ChartBot(pct, seed=rng.random())
        bots = field(pct, rng, fkind) + [hero]
        rng.shuffle(bots)
        timing = {}
        res, fin, nh = engine.run_table(bots, rng, timing=timing)
        i = bots.index(hero)
        pts.append(res[i]); places.append(fin[i]); hands.append(nh)
        times.extend(timing.get(id(hero), []))
    return pts, places, times, hands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, default=400)
    ap.add_argument("--hero", default="agent", choices=["agent", "chart"])
    ap.add_argument("--field", default="mixed", choices=["mixed", "charts", "self"])
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    t0 = time.time()
    per = a.tables // a.procs
    with Pool(a.procs) as p:
        out = p.map(worker, [(per, 1234 + i, a.hero, a.field) for i in range(a.procs)])
    pts = [x for o in out for x in o[0]]
    places = [x for o in out for x in o[1]]
    times = sorted(x for o in out for x in o[2])
    hands = [x for o in out for x in o[3]]
    m = statistics.mean(pts); se = statistics.stdev(pts) / len(pts) ** 0.5
    fair = sum(engine.POINTS) / 8
    print("field=%s hero=%s tables=%d  points/table %.2f +- %.2f  (field average %.1f)" % (a.field, a.hero, len(pts), m, se, fair))
    won = sum(1 for p in places if p == 0)
    print("won table outright %.1f%%, avg hands per table %.0f" % (100 * won / len(pts), statistics.mean(hands)))
    if times:
        q = lambda f: times[min(len(times) - 1, int(f * len(times)))] * 1e3
        print("decisions %d  median %.3f ms  p99 %.3f ms  max %.3f ms  over 1ms: %d"
              % (len(times), q(0.5), q(0.99), times[-1] * 1e3, sum(1 for x in times if x > 1e-3)))
    print("%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
