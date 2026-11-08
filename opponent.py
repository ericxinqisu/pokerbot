"""Per-seat opponent model. Bot state resets each table, and opponents are
anonymous, so we learn each seat from scratch over <=200 hands with priors
(Bayesian shrinkage) so early estimates do not swing wildly.
"""


class SeatStats:
    __slots__ = ("hands", "vpip", "pfr", "shove", "bets", "calls", "folds_vs_bet",
                 "faced_bet", "shown_pcts", "shove_pcts", "limp")

    def __init__(self):
        self.hands = 0
        self.vpip = 0; self.pfr = 0; self.shove = 0; self.limp = 0
        self.bets = 0; self.calls = 0
        self.folds_vs_bet = 0; self.faced_bet = 0
        self.shown_pcts = []   # preflop strength percentile of hands shown at showdown
        self.shove_pcts = []   # percentile of hands shown after a preflop all-in

    # --- shrunk estimates -------------------------------------------------
    @staticmethod
    def _est(k, n, prior, w):
        return (k + prior * w) / (n + w)

    def vpip_rate(self):
        return self._est(self.vpip, self.hands, 0.25, 12)

    def pfr_rate(self):
        return self._est(self.pfr, self.hands, 0.15, 12)

    def shove_rate(self):
        return self._est(self.shove, self.hands, 0.03, 15)

    def fold_to_bet(self):
        return self._est(self.folds_vs_bet, self.faced_bet, 0.45, 8)

    def aggression(self):
        """Share of postflop actions that are bets/raises (vs calls)."""
        return self._est(self.bets, self.bets + self.calls, 0.45, 8)

    def raise_range_pct(self):
        """Estimated top-x% (as fraction) of hands this seat raises preflop."""
        x = self.pfr_rate()
        if self.shown_pcts:
            # uniform over top-x has mean x/2 -> mix in evidence from showdowns
            m = 2 * sum(self.shown_pcts) / len(self.shown_pcts)
            k = len(self.shown_pcts)
            x = (x * 4 + m * k) / (4 + k)
        return min(1.0, max(0.02, x))

    def shove_range_pct(self, prior_pct):
        """Shove range; prior from the push/fold equilibrium at that depth."""
        x = prior_pct
        n = self.hands
        if n:
            obs = self.shove / n
            x = (prior_pct * 10 + obs * n) / (10 + n)
        if self.shove_pcts:
            m = min(1.0, 2 * sum(self.shove_pcts) / len(self.shove_pcts))
            k = len(self.shove_pcts)
            x = (x * 3 + m * k) / (3 + k)
        return min(1.0, max(0.02, x))

    def kind(self):
        """Coarse label, handy in logs."""
        if self.hands < 10:
            return "unknown"
        if self.shove_rate() > 0.4:
            return "maniac"
        if self.vpip_rate() > 0.6 and self.aggression() < 0.3:
            return "station"
        if self.vpip_rate() < 0.12:
            return "nit"
        return "reg"


class OpponentModel:
    def __init__(self, n_seats=8):
        self.s = [SeatStats() for _ in range(n_seats)]

    def __getitem__(self, seat):
        return self.s[seat]

    def update_hand(self, history, seats_in_hand, shown, big_blind, pct_lookup, stacks_start):
        """history: list of (seat, street, action, amount_to); shown: {seat: hole}."""
        pre_actions = {}
        shoved = set()
        for seat, street, act, amt in history:
            st = self.s[seat]
            if street == 0:
                if act in ("call", "raise", "allin"):
                    pre_actions.setdefault(seat, set()).add(act)
                if act in ("raise", "allin") and amt >= 0.9 * stacks_start.get(seat, 1e9):
                    shoved.add(seat)
            else:
                if act in ("raise", "bet", "allin"):
                    st.bets += 1
                elif act == "call":
                    st.calls += 1
        # fold-to-bet: an aggressive action followed by this seat's response
        facing = False
        street_now = 0
        for seat, street, act, amt in history:
            if street != street_now:
                facing = False; street_now = street
            if street > 0:
                if act in ("raise", "bet", "allin"):
                    facing = True
                    continue
                if facing and act in ("fold", "call", "raise"):
                    self.s[seat].faced_bet += 1
                    if act == "fold":
                        self.s[seat].folds_vs_bet += 1
        for seat in seats_in_hand:
            st = self.s[seat]
            st.hands += 1
            acts = pre_actions.get(seat, set())
            if acts:
                st.vpip += 1
            if acts & {"raise", "allin"}:
                st.pfr += 1
            if seat in shoved:
                st.shove += 1
        for seat, hole in shown.items():
            pct = pct_lookup(hole)
            if pct is None:
                continue
            if seat in shoved:
                self.s[seat].shove_pcts.append(pct)
            elif seat in pre_actions and "raise" in pre_actions[seat]:
                self.s[seat].shown_pcts.append(pct)
