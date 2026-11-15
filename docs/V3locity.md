# V3locity — More than a Hack 2027, Phase 1 (Ball Possession)
## Technical document — scenario S2: XGBoost

**Submission file:** `V3locity.csv` (SHA-256 `3fca39c4b618e106adebec913e90551e3c03cbe80eec793509e75f5ecca5022f`)
**Code:** single Jupyter notebook `MTAH2027_Possession_Scenarios_v3.ipynb` in our public repository (URL: _to be filled_)

## 1. Approach
We treat possession as a sequence-labelling problem with three states (H, A, D) and decode it globally with a
Viterbi algorithm, because the ranking metric (`spellF1`) scores whole possessions rather than frames. Frame-level
evidence comes from football rules (IFAB Laws 8, 9, 13–17), from self-supervised models trained on the tracking data
itself, and from weak labels derived from the provided Smart Tagging. No possession label was ever produced by hand,
and the public 5-minute ground truth was used only to check that the pipeline is correct, never to choose parameters.
In this scenario the learned components use a XGBoost classifier.

### Stage A — Ball pre-processing
Ball samples farther than 8.0 m outside the pitch, isolated jumps faster than 40.0 m/s, and detections shorter
than 3 frames are removed; continuous runs are smoothed with a Savitzky–Golay filter (window 7, order 2).

### Stage B — Who is closest to the ball when it is missing (model M1, self-supervised)
On frames where the ball is visible we cut artificial gaps whose lengths follow the real gap distribution, and train the
model to predict which team has the player closest to the ball (within 2.0 m) from the ball positions at both ends
of the gap and the configuration of all tracked players (team centroids, spreads, fastest players, densest cluster,
distances to the interpolated ball). Gaps of up to 10 frames are linearly interpolated. Validation uses
4-fold GroupKFold over 2-minute blocks of the whole match. Out-of-fold: accuracy 0.7936, AUC 0.8621,
log-loss 0.5061 (n = 96487).

### Stage C — Contact and control
A contact is a frame where the nearest player is within 1.0 m of a reliably located ball; a duel is when players of
both teams are within 1.0 m (possession then stays with the previous team). Consecutive contacts form episodes.
An episode counts as **control** if it lasts at least 8 frames (0.32 s, "dwell"), or if the next team touch
within 100 frames (4.0 s) is by a team-mate ("chain"). Unconfirmed touches (deflections) carry weak evidence.

### Stage D — Ball out of play (model M2, weakly supervised)
Dead-ball features follow the Laws of the Game: ball beyond the lines, collective slow-down of players (with a
1.0-s lead, since players react after the whistle and after the restart kick), ball not tracked for long,
throw-in takers beyond the touchline, players at corner arcs, kick-off formation. Weak labels: the first 2.5 s of
every `SET PIECES` tag are "dead", the interior of possession phases is "alive", the time between a `GOALS` tag and the
following kick-off tag is "dead", and frames 110,808–113,741 (before the second-half kick-off, published) are dead.
Additional features describe the ball being stationary before a restart and the distance of the last ball position
to restart spots (corner arcs, goal area, touchlines, centre spot). Out-of-fold AUC against the weak labels 0.8003 (log-loss 0.4768).

### Stage E — Decoding
Team log-odds = control evidence + 0.1 × M1 log-odds inside long gaps; dead log-odds = 0.25 × logit(M2).
Transition penalties: H↔A 10.0, alive→dead 8.0, dead→alive 8.0. A dead-ball stretch must last at least 2.0 s
(a restart needs the ball to be placed), enforced exactly with a semi-Markov Viterbi in which the dead state is a
chain of 50 frames. Because Viterbi sees the whole half, possession boundaries are placed at the first touch of the
new team rather than when control is confirmed.

## 2. Assets used and dropped
| Asset | Use |
|---|---|
| Tactical ATD (players + ball) | all stages |
| Smart Tagging | weak labels for M2; independent consistency check of team labels |
| Homography (tactical) | error analysis only |
| Tactical video | not used in this scenario |
| Broadcast video / ATD | **dropped** — not synchronised to the tactical timeline |

## 3. Possession rules implemented
- A change of possession requires **control** by the other team (dwell or chain); single touches, deflections and
  duels keep the previous possessor.
- Dead ball (`D`) is never a possession; after a restart, the team in possession is decided by the next controlled
  touch in the tracking, not copied from the Smart Tagging.
- Loose and contested balls stay with the last controlling team.
- Ungraded frames (before kick-off, half-time, after the final whistle) are written as `X`.

## 4. Results available before submission
Public 5-minute window (sanity check only; 17 possessions, differences < 0.04 are noise): spellF1 0.4000
(90% bootstrap 0.2444–0.5444), macroF1 0.6228, 19 predicted vs 17 true possessions.
Full-match proxies: 326 possessions (3.43 per minute of live ball, median 11.0 s), team agreement with
Smart Tagging phases 0.6442, set-piece restarts recovered 0.6949, dead-ball share 0.1018.

## 5. Limitations
- M2 learns from automatically generated tags, so systematic tagging errors propagate to dead-ball detection.
- Very long ball gaps (> 10 s) remain hard; M1 accuracy drops with gap length.
- Control thresholds were set from football reasoning and literature, not tuned on hidden labels; they may not match
  the exact ownership rule of the corrected tracking used for the ground truth.

- With more time: event-type-specific restart models (throw-in, corner, goal kick with last-touch rules) and a
  sequence model learned end-to-end on top of the current features.

## 6. Reproduction
1. Put the official data bundle in `data/tactical/` (or set `data_dir`).
2. Install the pinned versions (`requirements_used.txt` in the run folder); Python 3.10.20.
3. Run all cells of the notebook with `scenarios` containing `S2`; the file is written to
   `<work_dir>/runs_v3/<RUN_ID>/scenarios/S2_XGBoost/V3locity.csv`.

Seed 2027, `n_threads` = 8, GBDT device `cpu`, official repository commit `3de6b60aeabd2475d95879a651d38b72bc4fba00`.
Library versions: numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, LightGBM 4.6.0, XGBoost 3.2.0,
CatBoost 1.2.10. Re-running with identical versions and thread count reproduces the same file
(checked by SHA-256 within our runs).

## References
- IFAB, *Laws of the Game 2025/26*, Laws 8 and 9 (The FA edition).
- Daniel Link and Martin Hoernig, "Individual ball possession in soccer," *PLoS ONE*, vol. 12, no. 7, e0179953, 2017,
  doi:10.1371/journal.pone.0179953.
- Ferran Vidal-Codina, Nicolas Evans, Bahaeddine El Fakir and Johsan Billingham, "Automatic event detection in football
  using tracking data," *Sports Engineering*, vol. 25, no. 1, art. 18, 2022, doi:10.1007/s12283-022-00381-6.
