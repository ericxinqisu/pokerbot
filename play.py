"""Play the bot yourself in the terminal.

    python3 play.py                      # you + 7 copies of the bot, full SNG rules
    python3 play.py --seats 2            # heads-up vs the bot
    python3 play.py --opponents mixed    # you + the bot + the stand-in field from bench.py
    python3 play.py --hint               # show what the bot would do in your spot
    python3 play.py --watch              # don't play: watch the bot play a whole table

Commands on your turn:  f = fold, k = check, c = call, r 300 = raise to 300,
                        a = all-in, p = pot-size raise, h = hint
"""
import argparse, os, pickle, random

import engine
from agent import Agent
from bots import RandomBot, CallingStation, Maniac, ChartBot, Nit, ShortShover
from cards import RANK_CHARS

HERE = os.path.dirname(os.path.abspath(__file__))
SUITS = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
STREETS = ["Preflop", "Flop", "Turn", "River"]


def cs(cards):
    return " ".join(RANK_CHARS[c // 4] + SUITS[c % 4] for c in cards) or "-"


class Narrator:
    """Prints the action. Shared by the human seat and watch mode."""

    def __init__(self, names):
        self.names = names; self.hand = -1; self.seen = 0; self.street = -1

    def _header(self, st, board):
        import cards
        n = sum(cards.VARIANT["board"][:st + 1])
        print("  -- %s --  board: %s" % (STREETS[min(3, st)], cs(board[:n])))
        self.street = st; self.street_bet = 0

    def history(self, o_or_info, upto=None):
        h = o_or_info["history"]
        for seat, st, act, amt in h[self.seen:upto]:
            if st != self.street:
                self._header(st, o_or_info["board"])
            if act == "raise":
                word = ("bets %d" if st > 0 and self.street_bet == 0 else "raises to %d") % amt
            else:
                word = {"fold": "folds", "check": "checks", "call": "calls %d" % amt,
                        "allin": "ALL-IN (%d)" % amt}[act]
            if act in ("raise", "allin"):
                self.street_bet = max(self.street_bet, amt)
            print("   %-10s %s" % (self.names[seat], word))
        self.seen = len(h)

    def catch_up(self, o):
        """Print the street header if a new street started with no action yet."""
        if o["street"] != self.street:
            self._header(o["street"], o["board"])

    def new_hand(self, o):
        if o["hand_no"] != self.hand:
            self.hand = o["hand_no"]; self.seen = 0; self.street = -1
            n = len(o["stacks"])
            print("\n" + "=" * 60)
            print("Hand %d   blinds %d/%d   button: %s" % (o["hand_no"] + 1, o["sb"], o["bb"], self.names[o["button"]]))
            print("  " + "  ".join("%s %d" % (self.names[i], o["stacks"][i] + o["bets"][i])
                                   for i in range(n) if o["alive"][i]))

    def hand_end(self, info):
        self.history(info)
        if info["shown"]:
            print("  Board: %s" % cs(info["board"]))
            for seat, hole in info["shown"].items():
                print("   %-10s shows %s" % (self.names[seat], cs(hole)))
        for seat, end in info["stacks_end"].items():
            start = info["stacks_start"][seat]
            if end > start:
                verb = "win" if self.names[seat] == "You" else "wins"
                print("  >> %s %s %d" % (self.names[seat], verb, end - start))
        for seat, end in info["stacks_end"].items():
            if end == 0:
                print("  xx %s %s eliminated" % (self.names[seat], "are" if self.names[seat] == "You" else "is"))


class Human:
    def __init__(self, hint=False):
        self.hint_on = hint
        self.advisor = Agent(seed=7)
        self.narr = None

    def new_table(self, seat, n):
        self.seat = seat
        self.advisor.new_table(seat, n)

    def on_hand_end(self, info):
        self.advisor.on_hand_end(info)
        self.narr.hand_end(info)

    def act(self, o):
        self.narr.new_hand(o)
        self.narr.history(o)
        self.narr.catch_up(o)
        me = o["seat"]
        print("\n  Your cards: %s    pot %d    your stack %d" % (cs(o["hole"]), o["pot"], o["stacks"][me]))
        if o["to_call"]:
            print("  To call: %d" % o["to_call"])
        opts = []
        if "fold" in o["legal"]: opts.append("f")
        if "check" in o["legal"]: opts.append("k")
        if "call" in o["legal"]: opts.append("c")
        if "raise" in o["legal"]:
            opts.append("r <%d-%d>" % (o["min_raise_to"], o["max_raise_to"])); opts += ["p", "a"]
        if self.hint_on:
            print("  Bot would: %s" % self._hint(o))
        while True:
            try:
                s = input("  Action [%s, h]: " % ", ".join(opts)).strip().lower()
            except EOFError:
                s = "f"
            if s == "h":
                print("  Bot would: %s" % self._hint(o)); continue
            if s == "f" and "fold" in o["legal"]: return ("fold", 0)
            if s == "f" and "check" in o["legal"]:
                print("  (you can check for free; checking)"); return ("check", 0)
            if s == "k" and "check" in o["legal"]: return ("check", 0)
            if s == "c" and "call" in o["legal"]: return ("call", 0)
            if s in ("c", "k") and "check" in o["legal"]: return ("check", 0)
            if "raise" in o["legal"]:
                if s == "a": return ("raise", o["max_raise_to"])
                if s == "p": return ("raise", o["current_bet"] + o["pot"] + o["to_call"])
                if s.startswith("r"):
                    try:
                        amt = int(s.split()[1])
                        if o["min_raise_to"] <= amt <= o["max_raise_to"]:
                            return ("raise", amt)
                        print("  Raise must be between %d and %d" % (o["min_raise_to"], o["max_raise_to"]))
                        continue
                    except (IndexError, ValueError):
                        pass
            print("  Not a legal action here.")

    def _hint(self, o):
        a, amt = self.advisor.act(dict(o))
        if a != "raise":
            return a
        return ("bet %d" if o["current_bet"] == 0 else "raise to %d") % amt


class Watched:
    """Wraps a bot so watch mode can print the action as it happens."""
    def __init__(self, bot, narr, name):
        self.bot = bot; self.narr = narr; self.name = name

    def new_table(self, seat, n):
        self.seat = seat
        if hasattr(self.bot, "new_table"): self.bot.new_table(seat, n)

    def on_hand_end(self, info):
        if hasattr(self.bot, "on_hand_end"): self.bot.on_hand_end(info)
        if self.seat == min(info["seats_in_hand"]):  # one seat narrates the result
            self.narr.hand_end(info)

    def act(self, o):
        self.narr.new_hand(o)
        self.narr.history(o)
        self.narr.catch_up(o)
        a = self.bot.act(o)
        if self.name == "Bot":
            print("   [Bot holds %s, board %s]" % (cs(o["hole"]), cs(o["board"])))
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seats", type=int, default=8)
    ap.add_argument("--opponents", default="bots", choices=["bots", "mixed"])
    ap.add_argument("--hint", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pct = pickle.load(open(os.path.join(HERE, "tables.pkl"), "rb"))["pre_pct"]

    if a.opponents == "mixed":
        pool = [("ChartBot", ChartBot(pct, seed=1)), ("Nit", Nit(pct)), ("Station", CallingStation()),
                ("Maniac", Maniac()), ("Shover", ShortShover(pct)), ("Random", RandomBot(3)),
                ("ChartBot2", ChartBot(pct, open_top=0.15, seed=2))]
        others = [("Bot", Agent(seed=1))] + pool
    else:
        others = [("Bot%d" % (i + 1), Agent(seed=i + 1)) for i in range(7)]
    others = others[:a.seats - 1]

    if a.watch:
        seats = [("Bot", Agent(seed=99))] + others
    else:
        seats = [("You", Human(hint=a.hint))] + others
    rng.shuffle(seats)
    names = [n for n, _ in seats]
    narr = Narrator(names)
    bots = []
    for name, b in seats:
        if isinstance(b, Human):
            b.narr = narr; bots.append(b)
        elif a.watch:
            bots.append(Watched(b, narr, name))
        else:
            bots.append(b)

    hero = names.index("You" if not a.watch else "Bot")
    result, finish, hands = engine.run_table(bots, rng)
    print("\n" + "=" * 60)
    print("Table over after %d hands." % hands)
    order = sorted(range(len(names)), key=lambda i: -result[i])
    for i in order:
        place = "%d" % (finish[i] + 1) if finish[i] is not None else "alive (ICM)"
        print("  %-10s place %-12s points %.1f" % (names[i], place, result[i]))
    print("\n%s scored %.1f points (field average %.1f)."
          % (names[hero], result[hero], sum(engine.POINTS[:len(names)]) / len(names)))


if __name__ == "__main__":
    main()
