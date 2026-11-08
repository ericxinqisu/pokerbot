"""The competition bot. Every decision is table lookups + arithmetic (<1 ms).

Layers (see the strategy doc):
  1. Blueprint tables from Monte Carlo (calibrate.py) and push/fold CFR+ (pushfold_cfr.py)
  2. Rules that turn equity + pot odds into actions, with sizes from a small action set
  3. Tournament layer: ICM-adjusted call/shove thresholds
  4. Per-seat opponent model: widen/tighten ranges, bluff only seats that fold
"""
import gc, os, pickle, random, time

from features import preflop_key, postflop_key
from opponent import OpponentModel
from icm import icm

HERE = os.path.dirname(os.path.abspath(__file__))
TOP_X = (2, 5, 8, 12, 16, 20, 25, 30, 40, 50, 65, 80, 100)
POST_NS = (1, 2, 3, 5)

# --- knobs worth tuning on the day ----------------------------------------
PUSH_FOLD_BB = 10          # at or below this effective depth, unopened pots are shove-or-fold
OPEN_PCT = {1: 0.48, 2: 0.42, 3: 0.30, 4: 0.24, 5: 0.20, 6: 0.17, 7: 0.14}  # by players left to act
OPEN_SIZE_BB = 2.5
VALUE_EQ_HU = 0.62         # postflop value-bet equity threshold heads-up
RAISE_EQ = 0.74            # postflop raise-for-value threshold vs a bet
ICM_MAX_PLAYERS = 6        # exact ICM only when <= this many alive (speed)
ICM_FROM_HAND = 0          # use ICM only from this hand on (or when <= 3 remain). With a skill
                           # edge, chips early are worth more than ICM says; tune with ablate.py
TIME_BUDGET = 0.0006       # seconds; above this skip optional work


def _load(name):
    with open(os.path.join(HERE, name), "rb") as f:
        return pickle.load(f)


class Agent:
    def __init__(self, seed=0):
        t = _load("tables.pkl")
        self.pre_eq = t["pre_eq"]; self.pct = t["pre_pct"]; self.vs_top = t["pre_vs_top"]
        self.post_eq = t["post_eq"]; self.post_n = t["post_n"]
        try:
            pf = _load("pushfold.pkl")
        except FileNotFoundError:
            pf = None
        self.pf = pf
        self.rng = random.Random(seed)
        self.opp = OpponentModel(8)
        self.seat = None
        self.max_dt = 0.0
        # the big lookup tables are permanent: move them out of the GC's view so a
        # collection cannot stall a decision past the 1 ms limit
        gc.collect(); gc.freeze()

    # ------------------------------------------------------------ lifecycle
    def new_table(self, seat, n_seats):
        self.opp = OpponentModel(n_seats)
        self.seat = seat

    def on_hand_end(self, info):
        gc.collect(0)  # pay for young-gen GC between hands, not inside act()
        self.opp.update_hand(info["history"], info["seats_in_hand"], info["shown"], info["bb"],
                             lambda h: self.pct.get(preflop_key(h)), info["stacks_start"])

    def act(self, o):
        t0 = time.perf_counter()
        self._t0 = t0
        try:
            a = self._act(o)
        except Exception:
            a = ("check", 0) if "check" in o["legal"] else ("fold", 0)
        dt = time.perf_counter() - t0
        if dt > self.max_dt:
            self.max_dt = dt
        return a

    # ------------------------------------------------------------ helpers
    def _late(self):
        return time.perf_counter() - self._t0 > TIME_BUDGET

    @staticmethod
    def _icm_on(o, n_alive):
        return o["hand_no"] >= ICM_FROM_HAND or n_alive <= 3

    def _vs_top(self, key, x):
        x100 = x * 100
        for g in TOP_X:
            if g >= x100:
                return self.vs_top[g].get(key, 0.5)
        return self.vs_top[100].get(key, 0.5)

    def _post_eq(self, hole, board, n):
        fine, coarse, catk = postflop_key(hole, board)
        k = fine if self.post_n.get(fine, 0) >= 40 else (coarse if self.post_n.get(coarse, 0) >= 30 else catk)
        def get(m):
            return self.post_eq[m].get(k, self.post_eq[m].get(catk, 0.3))
        if n in POST_NS:
            eq = get(n)
        elif n == 4:
            eq = 0.5 * (get(3) + get(5))
        else:
            eq = get(5) ** (n / 5.0)
        return eq, fine

    @staticmethod
    def _order(o):
        n = len(o["stacks"])
        alive = [i for i in range(n) if o["alive"][i]]
        b = o["button"]
        order = [(b + k) % n for k in range(1, n + 1)]
        order = [i for i in order if i in alive]
        if len(alive) == 2:
            sbs, bbs = b, [i for i in alive if i != b][0]
            pre = [b, bbs]
        else:
            sbs, bbs = order[0], order[1]
            pre = order[2:] + order[:2]
        return pre, sbs, bbs

    def _icm_required(self, o, villain, risk, pot):
        """Equity needed to call off `risk` chips vs `villain` for a pot of `pot` (ICM)."""
        n_alive = sum(o["alive"])
        chip_req = risk / (pot + risk)
        me = o["seat"]
        if n_alive > ICM_MAX_PLAYERS or self._late() or not self._icm_on(o, n_alive):
            stacks = [s + b for s, b in zip(o["stacks"], o["bets"]) if s + b > 0]
            avg = sum(stacks) / max(1, len(stacks))
            prem = 0.0 if n_alive <= 2 else min(0.06, 0.025 * risk / max(1, avg))
            return min(0.99, chip_req + prem)
        ids = [i for i in range(len(o["stacks"])) if o["alive"][i]]
        base = [o["stacks"][i] for i in ids]
        mi = ids.index(me); vi = ids.index(villain) if villain in ids else None
        pay = o["points"][:n_alive]

        def ev(hero_delta, vil_delta):
            s = list(base)
            s[mi] += hero_delta
            if vi is not None:
                s[vi] += vil_delta
            s = [max(0, x) for x in s]
            return icm(s, pay)[mi]
        e_fold = ev(0, pot)
        e_win = ev(pot, 0)
        e_lose = ev(-risk, pot + risk)
        if e_win - e_lose <= 1e-9:
            return chip_req
        return min(0.99, max(0.0, (e_fold - e_lose) / (e_win - e_lose)))

    def _raise(self, o, to):
        to = int(max(o["min_raise_to"], min(o["max_raise_to"], to)))
        # if the raise leaves us with < 30% of it behind, just shove
        if o["max_raise_to"] - to < 0.3 * to:
            to = o["max_raise_to"]
        return ("raise", to)

    @staticmethod
    def _passive(o):
        return ("check", 0) if "check" in o["legal"] else ("fold", 0)

    # ------------------------------------------------------------ decision
    def _act(self, o):
        if o["street"] == 0:
            return self._preflop(o)
        return self._postflop(o)

    def _preflop(self, o):
        me = o["seat"]; bb = o["bb"]
        hole = o["hole"]; key = preflop_key(hole)
        pct = self.pct.get(key, 0.5)
        pre, sbs, bbs = self._order(o)
        hist = [h for h in o["history"] if h[1] == 0]
        raises = [h for h in hist if h[2] in ("raise", "allin") and h[3] > bb]
        limpers = [h for h in hist if h[2] == "call" and h[3] <= bb]
        others = [i for i in range(len(o["stacks"])) if i != me and o["in_hand"][i]]
        my_total = o["stacks"][me] + o["bets"][me]
        eff = min(my_total, max(o["stacks"][i] + o["bets"][i] for i in others)) if others else my_total
        eff_bb = eff / bb
        idx = pre.index(me) if me in pre else 0
        behind = [i for i in pre[idx + 1:] if o["in_hand"][i] and not o["allin"][i]]

        # ---------------- unopened (maybe limpers)
        if not raises:
            if eff_bb <= PUSH_FOLD_BB and "raise" in o["legal"]:
                return self._shove_or_fold(o, key, pct, behind, eff_bb)
            if "raise" in o["legal"]:
                k = max(1, len(behind))
                thr = OPEN_PCT.get(k, 0.14)
                if behind:
                    v = sum(self.opp[i].vpip_rate() for i in behind) / len(behind)
                    if v < 0.18: thr *= 1.3      # they fold too much: steal wider
                    elif v > 0.5: thr *= 0.75    # they call everything: value only
                if limpers:
                    thr *= 0.6
                if pct <= thr:
                    size = OPEN_SIZE_BB + len(limpers) + (0.5 if me == sbs else 0)
                    return self._raise(o, size * bb)
            return self._passive(o)

        # ---------------- facing a raise
        aggr = raises[-1][0]
        st = self.opp[aggr]
        aggr_total = o["bets"][aggr] + o["stacks"][aggr]
        aggr_shove = o["allin"][aggr] or o["bets"][aggr] >= 0.9 * aggr_total
        if aggr_shove:
            S = min(30.0, max(1.0, round(2 * o["bets"][aggr] / bb) / 2))
            prior = self.pf["push_freq"][S] if self.pf else 0.3
            if o["bets"][aggr] / bb > 30:
                prior = 0.08  # deep shoves from a balanced player are tight
            if len(pre) > 2 and aggr not in (sbs,):
                prior *= 0.7   # earlier-position shoves are tighter
            x = st.shove_range_pct(prior)
        else:
            x = st.raise_range_pct()
        x *= 0.45 ** (len(raises) - 1)
        x = max(0.02, x)
        eq = self._vs_top(key, x)
        n_in = len(others)
        if n_in > 1:
            eq *= 0.85 ** (n_in - 1)
        to_call = o["to_call"]; pot = o["pot"]
        behind_me = o["stacks"][me]
        commit = to_call >= 0.4 * behind_me or aggr_shove

        if commit:
            req = self._icm_required(o, aggr, to_call, pot)
            if eq >= req:
                if "raise" in o["legal"] and not aggr_shove:
                    return ("raise", o["max_raise_to"])
                return ("call", 0)
            return self._passive(o)

        # normal-sized raise, deep stacks
        if (eq >= 0.60 or pct <= 0.02) and "raise" in o["legal"]:
            to = o["current_bet"] * 3 + sum(1 for h in hist if h[2] == "call") * o["current_bet"]
            return self._raise(o, to)
        in_pos = me == o["button"]
        realize = 0.9 if in_pos else 0.75
        if eq * realize * (pot + to_call) - to_call > 0 and to_call <= 0.15 * behind_me and eff_bb >= 15:
            return ("call", 0)
        return self._passive(o)

    def _shove_or_fold(self, o, key, pct, behind, eff_bb):
        me = o["seat"]; bb = o["bb"]
        pot = o["pot"]; my_in = o["bets"][me]; h = o["stacks"][me]
        if not behind:
            return ("check", 0) if "check" in o["legal"] else ("raise", o["max_raise_to"])
        p_nocall = 1.0
        best_v = behind[0]; best_c = -1
        x_call = 0.2
        for i in behind:
            si = min(h + my_in, o["stacks"][i] + o["bets"][i]) / bb
            S = min(30.0, max(1.0, round(2 * si) / 2))
            base_c = self.pf["call_freq"][S] if self.pf else 0.25
            st = self.opp[i]
            if st.shove_rate() > 0.4:
                c = 0.9
            else:
                c = min(0.95, base_c * (st.vpip_rate() / 0.25) ** 0.5)
            p_nocall *= (1 - c)
            if c > best_c:
                best_c = c; best_v = i; x_call = c
        eq = self._vs_top(key, x_call)
        if len(behind) > 2:
            eq *= 0.97
        steal = pot  # relative to folding: hero's posted blind is already in the pot
        vstack = o["stacks"][best_v]
        c_amt = min(h, vstack + o["bets"][best_v] - o["bets"][best_v])
        n_alive = sum(o["alive"])
        if n_alive <= ICM_MAX_PLAYERS and not self._late() and self._icm_on(o, n_alive):
            ids = [i for i in range(len(o["stacks"])) if o["alive"][i]]
            base = [o["stacks"][i] for i in ids]
            mi = ids.index(me); vi = ids.index(best_v)
            pay = o["points"][:n_alive]

            def ev(hd, vd):
                s = list(base); s[mi] += hd; s[vi] += vd
                return icm([max(0, x) for x in s], pay)[mi]
            e_fold = ev(0, 0)
            e_steal = ev(steal, 0)
            e_win = ev(steal + c_amt, -c_amt)
            e_lose = ev(-c_amt, steal + c_amt)
            e_push = p_nocall * e_steal + (1 - p_nocall) * (eq * e_win + (1 - eq) * e_lose)
            good = e_push > e_fold
        else:
            chip = p_nocall * steal + (1 - p_nocall) * (eq * (steal + c_amt) - (1 - eq) * c_amt)
            good = chip > 0.02 * c_amt  # small risk premium without exact ICM
        if good:
            return ("raise", o["max_raise_to"])
        return self._passive(o)

    def _postflop(self, o):
        me = o["seat"]; bb = o["bb"]
        hole = o["hole"]; board = o["board"]
        others = [i for i in range(len(o["stacks"])) if i != me and o["in_hand"][i]]
        n = max(1, len(others))
        eq, fine = self._post_eq(hole, board, n)
        street, cat, sub, fd, sd, bpaired, bsuit, bconn = fine
        last_street = len(board) >= 5
        # opponents who stayed in are stronger than random; tighter players more so
        v = sum(self.opp[i].vpip_rate() for i in others) / n
        eq -= 0.06 * (1 - v)
        pot = o["pot"]; to_call = o["to_call"]; h = o["stacks"][me]
        legal = o["legal"]
        draw = (fd or sd == 2) and not last_street

        if to_call > 0:
            bettor = max(others, key=lambda i: o["bets"][i])
            agg = self.opp[bettor].aggression()
            bet_frac = to_call / max(1, pot - to_call)
            disc = (0.08 + 0.05 * (street - 3)) * min(1.5, bet_frac) * max(0.2, 1.2 - agg)
            eqf = eq - disc
            if draw:
                eqf += 0.04
            commit = to_call >= 0.45 * h or o["allin"][bettor]
            if commit:
                req = self._icm_required(o, bettor, to_call, pot)
                if eqf >= req:
                    return ("call", 0) if "call" in legal else ("raise", o["max_raise_to"])
                return self._passive(o)
            if eqf >= RAISE_EQ and "raise" in legal and self.rng.random() < 0.8:
                return self._raise(o, o["current_bet"] + 0.8 * (pot + to_call))
            req = to_call / (pot + to_call)
            if eqf >= req:
                return ("call", 0)
            return self._passive(o)

        # checked to us / first to act
        v_th = VALUE_EQ_HU / (1 + 0.25 * (n - 1))
        wet = bsuit >= 2 and bconn >= 3
        frac = 0.75 if wet else 0.6
        if "raise" not in legal:
            return ("check", 0)
        if eq >= v_th:
            size = frac * pot
            if h <= 1.3 * pot:
                return ("raise", o["max_raise_to"])
            return self._raise(o, o["current_bet"] + max(bb, size))
        if draw and self.rng.random() < 0.55:
            return self._raise(o, o["current_bet"] + max(bb, 0.6 * pot))
        # bluff seats that fold: HU only, never versus stations
        if n == 1:
            ftb = self.opp[others[0]].fold_to_bet()
            p_bluff = 0.0
            if ftb > 0.5:
                p_bluff = min(0.7, (ftb - 0.4) * 1.5)
            elif ftb > 0.4:
                p_bluff = 0.15
            if cat == 0 and self.rng.random() < p_bluff:
                return self._raise(o, o["current_bet"] + max(bb, 0.5 * pot))
        return ("check", 0)
