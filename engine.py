"""Local 8-seat no-limit sit-and-go simulator that mirrors the competition format:
2,000 chips each, blinds 10/20 rising every 25 hands, stop at one survivor or
200 hands, points by finishing place, ICM split of points at the hand cap.

The real starter kit ships its own simulator and Agent interface; this one exists
so the strategy can be tested before event day. Agents implement:
    act(obs) -> (action, amount)   action in fold/check/call/raise, amount = raise-to
    on_hand_end(info)              optional, for opponent modelling
    new_table(seat, n_seats)       optional, state reset per table
"""
import random, time

import cards
from cards import deck, evaluate
from icm import icm

BLINDS = [(10, 20), (15, 30), (25, 50), (50, 100), (75, 150), (100, 200),
          (150, 300), (200, 400), (300, 600)]
POINTS = [100, 60, 40, 25, 15, 8, 3, 0]  # placeholder; real table announced on the day


class Seat:
    def __init__(self, bot, stack):
        self.bot = bot; self.stack = stack
        self.hole = []; self.folded = False; self.allin = False
        self.bet = 0; self.total = 0; self.alive = True


def run_table(bots, rng, start=2000, max_hands=200, level_hands=25, blinds=BLINDS,
              points=POINTS, time_limit=None, timing=None):
    n = len(bots)
    seats = [Seat(b, start) for b in bots]
    for i, b in enumerate(bots):
        if hasattr(b, "new_table"):
            b.new_table(i, n)
    finish = [None] * n
    places_left = n
    button = rng.randrange(n)
    hand_no = 0
    while hand_no < max_hands and sum(s.alive for s in seats) > 1:
        lvl = min(hand_no // level_hands, len(blinds) - 1)
        sb, bb = blinds[lvl]
        alive = [i for i in range(n) if seats[i].alive]
        # move button to next alive seat
        button = next(((button + k) % n for k in range(1, n + 1) if seats[(button + k) % n].alive))
        stacks_start = {i: seats[i].stack for i in alive}
        _play_hand(seats, alive, button, sb, bb, rng, hand_no, points, time_limit, timing, stacks_start)
        # eliminations: bigger starting stack finishes higher
        busted = [i for i in alive if seats[i].stack == 0]
        busted.sort(key=lambda i: stacks_start[i])
        for i in busted:
            seats[i].alive = False
            finish[i] = places_left - 1
            places_left -= 1
        hand_no += 1
    # remaining players: ICM over remaining places
    result = [0.0] * n
    for i in range(n):
        if finish[i] is not None:
            result[i] = points[finish[i]]
    survivors = [i for i in range(n) if seats[i].alive]
    if len(survivors) == 1:
        result[survivors[0]] = points[0]
        finish[survivors[0]] = 0
    else:
        ev = icm([seats[i].stack for i in survivors], points[:len(survivors)])
        for i, v in zip(survivors, ev):
            result[i] = v
    return result, finish, hand_no


def _obs(seats, me, button, sb, bb, street, board, pot, cur_bet, min_raise, history,
         hand_no, points, alive_ids):
    s = seats[me]
    to_call = min(cur_bet - s.bet, s.stack)
    legal = []
    if to_call > 0:
        legal += ["fold", "call"]
    else:
        legal.append("check")
    max_to = s.bet + s.stack
    others_can_act = any(not seats[i].folded and not seats[i].allin and i != me for i in alive_ids)
    if s.stack > to_call and others_can_act:
        legal.append("raise")
    return {
        "hole": list(s.hole), "board": list(board), "street": street,
        "seat": me, "button": button, "sb": sb, "bb": bb,
        "stacks": [x.stack if x.alive else 0 for x in seats],
        "bets": [x.bet for x in seats],
        "in_hand": [x.alive and not x.folded for x in seats],
        "allin": [x.allin for x in seats],
        "alive": [x.alive for x in seats],
        "pot": pot, "to_call": to_call, "current_bet": cur_bet,
        "min_raise_to": min(cur_bet + min_raise, max_to), "max_raise_to": max_to,
        "legal": legal, "history": list(history), "hand_no": hand_no,
        "points": list(points),
    }


def _ask(bot, obs, time_limit, timing):
    t = time.perf_counter()
    try:
        act = bot.act(obs)
        err = False
    except Exception:
        act = None; err = True
    dt = time.perf_counter() - t
    if timing is not None:
        timing.setdefault(id(bot), []).append(dt)
    if err or act is None or (time_limit and dt > time_limit):
        return ("check", 0) if "check" in obs["legal"] else ("fold", 0)
    a, amt = act if isinstance(act, tuple) else (act, 0)
    if a not in obs["legal"]:
        if a == "raise":
            a = "call" if "call" in obs["legal"] else "check"
        elif a == "check" and "call" in obs["legal"]:
            a = "fold"
        elif a == "call" and "check" in obs["legal"]:
            a = "check"
        elif a == "fold" and "check" in obs["legal"]:
            a = "check"
        else:
            a = obs["legal"][0]
    if a == "raise":
        amt = int(max(obs["min_raise_to"], min(obs["max_raise_to"], amt)))
    return a, amt


def _play_hand(seats, alive, button, sb, bb, rng, hand_no, points, time_limit, timing, stacks_start):
    d = deck(); rng.shuffle(d)
    H = cards.VARIANT["hole"]
    for i in alive:
        s = seats[i]
        s.hole = [d.pop() for _ in range(H)]
        s.folded = False; s.allin = False; s.bet = 0; s.total = 0
    order = [(button + k) % len(seats) for k in range(1, len(seats) + 1)]
    order = [i for i in order if i in alive]  # left of button ... button last
    if len(alive) == 2:
        sb_seat = button; bb_seat = [i for i in alive if i != button][0]
    else:
        sb_seat, bb_seat = order[0], order[1]
    history = []

    def post(i, amt):
        s = seats[i]
        a = min(amt, s.stack)
        s.stack -= a; s.bet += a; s.total += a
        if s.stack == 0:
            s.allin = True
        return a

    post(sb_seat, sb); post(bb_seat, bb)
    board = []
    streets = cards.VARIANT["board"]
    for street in range(len(streets)):
        if street > 0:
            for _ in range(streets[street]):
                board.append(d.pop())
            for i in alive:
                seats[i].bet = 0
        live = [i for i in alive if not seats[i].folded]
        if len(live) == 1:
            break
        can_act = [i for i in live if not seats[i].allin]
        if street > 0 and len(can_act) < 2:
            continue
        cur_bet = max(seats[i].bet for i in alive)
        min_raise = bb
        # action order
        if street == 0:
            if len(alive) == 2:
                seq = [button, bb_seat]
            else:
                k = order.index(bb_seat)
                seq = order[k + 1:] + order[:k + 1]
        else:
            if len(alive) == 2:
                seq = [bb_seat, button]
            else:
                seq = order
        need = set(i for i in live if not seats[i].allin)
        idx = 0
        guard = 0
        while need and guard < 400:
            guard += 1
            i = seq[idx % len(seq)]; idx += 1
            s = seats[i]
            if i not in need or s.folded or s.allin:
                continue
            pot = sum(seats[j].total for j in alive)
            obs = _obs(seats, i, button, sb, bb, street, board, pot, cur_bet, min_raise,
                       history, hand_no, points, alive)
            a, amt = _ask(s.bot, obs, time_limit, timing)
            need.discard(i)
            if a == "fold":
                s.folded = True
                history.append((i, street, "fold", 0))
                if sum(1 for j in alive if not seats[j].folded) == 1:
                    need.clear()
            elif a in ("check", "call"):
                if a == "call":
                    post(i, cur_bet - s.bet)
                history.append((i, street, "allin" if s.allin else a, s.bet))
            else:
                amt = min(amt, s.bet + s.stack)
                raise_by = amt - cur_bet
                post(i, amt - s.bet)
                if raise_by >= min_raise:
                    min_raise = raise_by
                if s.bet > cur_bet:
                    cur_bet = s.bet
                    need = set(j for j in alive if not seats[j].folded and not seats[j].allin and j != i)
                history.append((i, street, "allin" if s.allin else "raise", s.bet))
    # showdown / award with side pots
    live = [i for i in alive if not seats[i].folded]
    shown = {}
    if len(live) == 1:
        seats[live[0]].stack += sum(seats[j].total for j in alive)
    else:
        while len(board) < sum(streets):
            board.append(d.pop())
        vals = {i: evaluate(seats[i].hole, board) for i in live}
        shown = {i: list(seats[i].hole) for i in live}
        contrib = {j: seats[j].total for j in alive}
        levels = sorted(set(contrib[i] for i in live))
        prev = 0
        for lv in levels:
            pot = sum(min(contrib[j], lv) - min(contrib[j], prev) for j in alive)
            elig = [i for i in live if contrib[i] >= lv]
            best = max(vals[i] for i in elig)
            winners = [i for i in elig if vals[i] == best]
            share, rem = divmod(pot, len(winners))
            for w in winners:
                seats[w].stack += share
            if rem:
                seats[winners[0]].stack += rem
            prev = lv
        # uncalled excess above the top live contribution returns to its owner
        top = levels[-1]
        for j in alive:
            if contrib[j] > top:
                seats[j].stack += contrib[j] - top
    info = {"history": history, "shown": shown, "board": list(board),
            "stacks_start": stacks_start, "seats_in_hand": list(alive), "bb": bb,
            "stacks_end": {i: seats[i].stack for i in alive},
            "contrib": {i: seats[i].total for i in alive}}
    for i in alive:
        b = seats[i].bot
        if hasattr(b, "on_hand_end"):
            try:
                b.on_hand_end(info)
            except Exception:
                pass
