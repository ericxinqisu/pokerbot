"""Stand-in opponents: the kinds of bots a 60-minute build tends to produce."""
import random

from cards import evaluate, category
from features import preflop_key


class RandomBot:
    def __init__(self, seed=None): self.r = random.Random(seed)
    def act(self, o):
        a = self.r.choice(o["legal"])
        if a == "raise":
            return a, self.r.randint(o["min_raise_to"], o["max_raise_to"])
        return a, 0


class CallingStation:
    def act(self, o):
        return ("call", 0) if "call" in o["legal"] else ("check", 0)


class Maniac:
    """Shoves every hand."""
    def act(self, o):
        if "raise" in o["legal"]:
            return "raise", o["max_raise_to"]
        return ("call", 0) if "call" in o["legal"] else ("check", 0)


class ChartBot:
    """Typical quick bot: preflop by hand-percentile chart, postflop by made-hand class."""
    def __init__(self, pct, open_top=0.2, call_top=0.12, seed=None, aggro=0.6):
        self.pct = pct; self.open_top = open_top; self.call_top = call_top
        self.r = random.Random(seed); self.aggro = aggro

    def act(self, o):
        p = self.pct.get(preflop_key(o["hole"]), 1.0)
        bb = o["bb"]; legal = o["legal"]
        if o["street"] == 0:
            facing = o["current_bet"] > bb
            if not facing and p <= self.open_top and "raise" in legal:
                return "raise", o["current_bet"] + 2 * bb
            if facing and p <= self.call_top * 0.4 and "raise" in legal:
                return "raise", o["current_bet"] * 3
            if facing and p <= self.call_top:
                return "call", 0
            if "check" in legal:
                return "check", 0
            if o["to_call"] <= bb and p <= 0.5:
                return "call", 0
            return "fold", 0
        cat = category(evaluate(o["hole"], o["board"]))
        if cat >= 2 and "raise" in legal and self.r.random() < self.aggro:
            return "raise", o["current_bet"] + max(bb, o["pot"] * 2 // 3)
        if cat >= 1:
            return ("call", 0) if "call" in legal else ("check", 0)
        if "check" in legal:
            if self.r.random() < 0.15 and "raise" in legal:
                return "raise", o["current_bet"] + max(bb, o["pot"] // 2)
            return "check", 0
        return "fold", 0


class Nit(ChartBot):
    def __init__(self, pct, seed=None):
        super().__init__(pct, open_top=0.07, call_top=0.05, seed=seed, aggro=0.8)


class ShortShover:
    """Push/fold heuristic once short: shove top 30% under 12 BB, else chart bot."""
    def __init__(self, pct, seed=None):
        self.pct = pct; self.base = ChartBot(pct, seed=seed)
    def act(self, o):
        me = o["seat"]
        if o["street"] == 0 and (o["stacks"][me] + o["bets"][me]) / o["bb"] < 12:
            p = self.pct.get(preflop_key(o["hole"]), 1.0)
            if p <= 0.3 and "raise" in o["legal"]:
                return "raise", o["max_raise_to"]
            if p <= 0.12 and "call" in o["legal"]:
                return "call", 0
            return ("check", 0) if "check" in o["legal"] else ("fold", 0)
        return self.base.act(o)
