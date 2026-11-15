# V3locity: Ball Possession from Tracking Data

More Than a Hack 2027, Phase 1. Technical document for the submitted file.

Team V3locity: Muhammad Nur Fuad (team lead), Asrul, Raihan Nur Rizqillah, Wa Ode Namida Heiwa, Wa Ode Resky Aulia A.
Code: https://github.com/muhammadfuad249/V3locity_More-Than-a-Hack-2027, notebook MTAH2027_Possession_Scenarios_v3.ipynb, scenario S2 (XGBoost)
Submission file: V3locity.csv
SHA-256: 3fca39c4b618e106adebec913e90551e3c03cbe80eec793509e75f5ecca5022f

## 1. Approach

We label every frame of the match as H (Barcelona in possession), A (Athletic in possession) or D (ball out of play) and decode each half in one pass with a Viterbi algorithm. A global decoder suits the ranking metric. spellF1 scores whole possessions, so a frame classifier that flickers between teams produces many short spells and loses most of its score even when its frame accuracy looks good. The decoder combines three kinds of evidence: contacts and control derived from the tracking and the Laws of the Game (IFAB Laws 8, 9 and 13 to 17), a self-supervised model that estimates which team is near the ball while the ball is missing (M1), and a weakly supervised dead-ball model trained on labels derived from the Smart Tagging (M2). We never labelled a frame by hand.

The hardest part of the data is the ball. The automatic tracking loses it on about 41% of graded frames, often for long stretches (median gap 10.7 s). Stages A and B below deal with this.

### Stage A: ball cleaning

We remove ball samples more than 8 m outside the pitch, isolated jumps faster than 40 m/s, and detections shorter than 3 frames. Each continuous run of ball positions is then smoothed with a Savitzky-Golay filter (window 7, order 2).

### Stage B: who is near the ball when it is missing (M1, self-supervised)

We interpolate gaps of up to 10 frames linearly. For longer gaps we train a classifier on frames where the ball is visible: we hide the ball over an artificial gap whose length follows the real gap distribution, and the model predicts which team has the player closest to the ball (within 2.0 m). Its inputs are the ball positions at both ends of the gap and the layout of all tracked players: team centroids and spreads, the fastest players, the densest cluster, and distances to the interpolated ball. We validate with 4-fold GroupKFold over 2-minute blocks of the whole match. Out-of-fold accuracy is 0.794, AUC 0.862 and log-loss 0.506 (n = 96,487).

### Stage C: contact and control

A contact is a frame where the nearest player is within 1.0 m of a reliably located ball. A duel is a frame where players of both teams are within 1.0 m; possession then stays with the previous team. Consecutive contacts form an episode. An episode counts as control if it lasts at least 8 frames (0.32 s, the dwell path) or if the next team touch within 100 frames (4.0 s) comes from a team-mate (the chain path). Touches that fail both tests, such as deflections, add only weak evidence.

### Stage D: ball out of play (M2, weakly supervised)

The dead-ball features follow the Laws of the Game: the ball beyond the lines, a collective slow-down of the players (with a 1.0 s lead, because players react after the whistle and after the restart kick), the ball untracked for a long time, a throw-in taker beyond the touchline, players at the corner arcs, and the kick-off formation. Further features describe a stationary ball before a restart and the distance from the last ball position to restart spots (corner arcs, goal area, touchlines, centre spot).

The weak labels come from the Smart Tagging. The first 2.5 s of every SET PIECES tag count as dead, the interior of possession phases counts as alive, the time between a GOALS tag and the next kick-off counts as dead, and so do frames 110,808 to 113,741, the published stretch before the second-half kick-off. Out-of-fold AUC against these weak labels is 0.800 (log-loss 0.477).

### Stage E: decoding

Team log-odds are the control evidence plus 0.1 times the M1 log-odds inside long gaps. Dead-ball log-odds are 0.25 times logit(M2). Switching between H and A costs 10.0, and entering or leaving D costs 8.0. A dead-ball stretch must last at least 2.0 s, since the ball has to be placed before a restart. We enforce this exactly with a semi-Markov Viterbi in which the dead state is a chain of 50 frames. Because the decoder sees the whole half, it places each change of possession at the first touch of the new team rather than at the moment control is confirmed.

## 2. Assets used and dropped

| Asset | How we used it |
|---|---|
| Tactical ATD (players and ball) | All stages |
| Smart Tagging | Weak labels for M2, and an independent check of team labels |
| Homography (tactical) | Error analysis in this scenario; pixel projection in the video experiment (Section 5) |
| Tactical video | Tried for ball detection (Section 5), not used in the submitted file |
| Broadcast video and broadcast ATD | Dropped, because they are not synchronised with the tactical timeline |

## 3. Possession rules implemented

- A change of possession needs control by the other team (dwell or chain). Single touches, deflections and duels leave the ball with the previous possessor.
- Loose and contested balls stay with the last team in control.
- Dead ball (D) is never a possession. After a restart, the next controlled touch in the tracking decides the team in possession; we do not copy it from the Smart Tagging.
- Ungraded frames (before kick-off, half-time, after the final whistle) are written as X.

## 4. How we chose the submitted scenario

The notebook builds six scenarios that share stages A to E and differ in the learner used for M1 and M2 (S1 to S5) or in the ball source (S6 adds ball detections from the video). The table compares them.

| Scenario | Public spellF1 (90% CI) | macroF1 | Restarts recovered | Median latency (frames) | Team agreement | M2 AUC |
|---|---|---|---|---|---|---|
| S1 LightGBM | 0.411 [0.278, 0.544] | 0.618 | 0.695 | -13.5 | 0.637 | 0.716 |
| S2 XGBoost (submitted) | 0.400 [0.244, 0.544] | 0.623 | 0.695 | -5 | 0.644 | 0.800 |
| S3 CatBoost | 0.400 [0.256, 0.544] | 0.630 | 0.492 | -12 | 0.649 | 0.829 |
| S4 Random forest | 0.354 [0.206, 0.491] | 0.502 | 0.237 | -7 | 0.633 | 0.800 |
| S5 Ensemble | 0.378 [0.222, 0.533] | 0.636 | 0.542 | -6 | 0.634 | 0.837 |
| S6 Video ball + ensemble | 0.337 [0.189, 0.484] | 0.545 | 0.559 | -21 | 0.648 | 0.812 |
| Official baseline smooth50 | 0.326 | 0.621 | 0.763 | n/a | 0.643 | n/a |

The public window holds 17 possessions, and the intervals of S1, S2, S3 and S5 overlap almost entirely, so the public score cannot separate them. We therefore chose on full-match diagnostics that relate to possession boundaries. S2 recovers the largest share of set-piece restarts (tied with S1), has the smallest median transition latency, and has stronger M1 and M2 models than S1. Restarts recovered is the share of SET PIECES tags with a predicted restart (a change from D to H or A) inside the tag or within 2 s after it. Team agreement is the share of predicted H and A frames inside team-tagged Smart Tagging phases that match the tagged team. Latency is the median offset between matched predicted and true transitions on the public window; a negative value means early.

## 5. Ball detection from video: tried and dropped

Since the tracking loses the ball so often, we tested whether a detector on the tactical video could fill the gaps (scenario S6). We fine-tuned a single-class YOLO detector (Ultralytics, yolo26s) on 960 by 960 crops at native resolution. The labels came from tracked ball positions projected to pixels with the homography, taken only from frames where the tracked ball moves slowly and is therefore likely on the ground.

The first detector reached mAP50 0.15 and often boxed grass, feet or part of the ball. We traced this to the labels. Once a first detector had learned to find the ball, the visible ball centre sat a median 7.8 px (90th percentile 20 px) from the projected tracking position. That is about one ball diameter, since the ball appears 6 to 13 px wide depending on its distance from the camera. In the second version we used the first detector as a teacher. It searched for the ball within 1.5 m of the tracked position, we refined each centre with a Difference-of-Gaussians blob detector, and we discarded crops without a confirmed ball (49.5% kept). We sized each box from the apparent ball diameter (minimum 15 px), added 356 hard negatives by inpainting the ball out of accepted crops, and linked detections into physically plausible tracks. The new detector reached mAP50 0.99 on its own validation crops.

Better detection did not change the possession score. On 40 held-out 2-second segments where the tracking sees the ball, only 69% to 71% of the selected detections lay within 1.5 m of the tracked ball for every confidence threshold from 0.05 to 0.70, and S6 scored public spellF1 0.337 in both versions. Because the errors persist at high confidence, we suspect ball-like objects, lofted balls whose ground projection is displaced, or moments when the tracking itself is wrong; we could not tell these apart without labelled video. More importantly, every scenario has a transition F1 at plus or minus 1 s of 0.10 or less. The limit lies in the timing of possession changes and in the ownership rule of the ground truth, not in where the ball is.

## 6. Results available before submission

On the public 5-minute window, S2 scores spellF1 0.400 (90% bootstrap interval 0.244 to 0.544) and macroF1 0.623, with 19 predicted possessions against 17 true ones. The four official baselines score 0.038, 0.287, 0.326 and 0.424 on this window; the organisers show that the last one, smooth125, is weaker than smooth50 on the full match. Over the whole match, S2 predicts 326 possessions (3.43 per minute of live ball, median duration 11.0 s), agrees with the Smart Tagging team on 64.4% of predicted possession frames inside tagged phases, recovers 69.5% of set-piece restarts, and marks 10.2% of graded frames as dead ball.

## 7. Limitations

- M2 learns from automatically generated tags, so systematic tagging errors carry over into dead-ball detection.
- Very long ball gaps (over 10 s) stay hard, and M1 accuracy drops as gaps get longer.
- We set the control thresholds from football reasoning and the literature, not from hidden labels. They may not match the ownership rule of the corrected tracking behind the ground truth.
- Transitions are often placed a few frames early (median -5 frames on the public window).

With more time we would build restart models for each event type (throw-in, corner, goal kick) with last-touch rules, and train a sequence model end to end on top of the current features to place transitions more precisely.

## 8. Reproduction

1. Put the official data bundle in data/tactical/, or point data_dir in the CONFIG cell at it.
2. Install the pinned versions from requirements_used.txt (Python 3.10.20).
3. Keep only S2 in CONFIG["scenarios"] (this also skips the video stages) and run all cells. The notebook writes the file to <work_dir>/runs_v3/<RUN_ID>/scenarios/S2_XGBoost/V3locity.csv and runs the official validator on it.

We used seed 2027, n_threads = 8, GBDT device cpu, and the official repository at commit 3de6b60aeabd2475d95879a651d38b72bc4fba00. Library versions: numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, LightGBM 4.6.0, XGBoost 3.2.0, CatBoost 1.2.10. With the same versions and thread count, our reruns reproduce the file byte for byte (same SHA-256).

## References

1. IFAB, Laws of the Game 2025/26, Laws 8, 9 and 13 to 17. The International Football Association Board, 2025.
2. Daniel Link and Martin Hoernig, "Individual ball possession in soccer," PLoS ONE, vol. 12, no. 7, e0179953, 2017. doi:10.1371/journal.pone.0179953.
3. Ferran Vidal-Codina, Nicolas Evans, Bahaeddine El Fakir and Johsan Billingham, "Automatic event detection in football using tracking data," Sports Engineering, vol. 25, no. 1, article 18, 2022. doi:10.1007/s12283-022-00381-6.
4. Andrew J. Viterbi, "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm," IEEE Transactions on Information Theory, vol. 13, no. 2, pp. 260-269, 1967. doi:10.1109/TIT.1967.1054010.
5. Abraham Savitzky and Marcel J. E. Golay, "Smoothing and differentiation of data by simplified least squares procedures," Analytical Chemistry, vol. 36, no. 8, pp. 1627-1639, 1964. doi:10.1021/ac60214a047.
6. Tianqi Chen and Carlos Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 785-794. doi:10.1145/2939672.2939785.
