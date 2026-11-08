# No Limit pokerbot kit

Python 3.10+, numpy (only for the offline solvers). The bot itself is pure Python.

| File | What it does |
|---|---|
| `cards.py` | Deck + hand evaluator. `VARIANT` dict covers hole count, board pattern, short deck, Omaha "must use 2". |
| `features.py` | Hand abstraction: preflop class, postflop bucket (made hand + draws + board texture). |
| `calibrate.py` | Monte Carlo -> `tables.pkl` (preflop/postflop equities) + `matrix.npy` (169x169). `--fast` ~1 min on 2 cores. |
| `pushfold_cfr.py` | Heads-up push/fold equilibrium by CFR+ -> `pushfold.pkl`. Prints exploitability. |
| `kuhn_cfr.py` | Vanilla CFR on Kuhn poker, sanity check from the Neller & Lanctot tutorial. |
| `icm.py` | Malmuth-Harville ICM (bitmask DP). |
| `opponent.py` | Per-seat stats with priors: VPIP, PFR, shove rate, aggression, fold-to-bet, shown ranges. |
| `agent.py` | The bot. Knobs at the top of the file. |
| `engine.py` | 8-seat SNG simulator matching the published format (2,000 chips, 10/20, +level every 25 hands, 200-hand cap, ICM split). |
| `bots.py` | Stand-in opponents: random, calling station, maniac, chart bots, nit, short-stack shover. |
| `bench.py` | Points per table + decision timing. `--field mixed|charts|self`, `--hero agent|chart`. |
| `ablate.py` | Switch layers off / sweep knobs: `python ablate.py 1600 full,no_icm,set:PUSH_FOLD_BB=10`. |
| `play.py` | Play the bot yourself in the terminal (`--seats 2`, `--hint`, `--watch`, `--opponents mixed`). |
| `build_submission.py` | Bundles everything + compressed tables into one `submission.py` (~270 KB). |

## Event-day order
1. Edit `cards.VARIANT` to the announced variant. Edit `engine.BLINDS` / `engine.POINTS` from the starter kit.
2. `python calibrate.py --fast` then `python pushfold_cfr.py` (hold'em-style 2-card variants only).
3. `python bench.py --tables 400` to confirm it runs and timing stays far under 1 ms.
4. Write the adapter at the bottom of `submission.py` that converts the kit's observation into the dict documented there and converts `(action, amount)` back.
5. `python build_submission.py`, test in the kit's simulator, submit. Re-submit improvements every 5 minutes.
