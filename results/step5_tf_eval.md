# Evaluation of `step5_tf`

From the per-clip predictions in `results/step5_tf.json` (2026-11-03, seed 0). Every prediction is on a held-out signer. 495 clips.

## CER by word length

| Letters | Clips | CER | Word acc | Letter recall |
|---|---|---|---|---|
| 2 | 27 | 44.4% | 44.4% | 83.3% |
| 3 | 135 | 30.4% | 36.3% | 77.3% |
| 4 | 126 | 31.0% | 23.8% | 77.4% |
| 5 | 117 | 34.5% | 12.8% | 74.2% |
| 6 | 72 | 29.6% | 13.9% | 74.8% |
| 7 | 9 | 30.2% | 0.0% | 71.4% |
| 8 | 9 | 25.0% | 11.1% | 75.0% |

## CER by signing speed

Speed = letters in the word / clip length (s), including the resting hand at the start and end. Clips are split into three equal groups.

| Group | Letters per second | Clips | CER | Word acc | Clips per signer |
|---|---|---|---|---|---|
| slow | 0.45–0.94 | 165 | 33.6% | 22.4% | anna 70, saaya 95 |
| medium | 0.94–1.22 | 165 | 27.5% | 29.1% | anna 88, saaya 64, thurushi 13 |
| fast | 1.22–2.73 | 165 | 34.2% | 19.4% | anna 7, saaya 6, thurushi 152 |

| Signer | Average letters per second |
|---|---|
| anna | 0.96 |
| saaya | 0.93 |
| thurushi | 1.65 |

## Letter confusions

![Letter confusion matrix](step5_tf_confusion.png)

Most common mistakes (true letter → what it became):

| True | Became | Times |
|---|---|---|
| O | (dropped) | 33 |
| E | (dropped) | 27 |
| A | (dropped) | 23 |
| L | (dropped) | 17 |
| Q | (dropped) | 17 |
| R | (dropped) | 17 |
| D | (dropped) | 15 |
| U | (dropped) | 15 |
| Z | (dropped) | 15 |
| K | (dropped) | 14 |
| P | (dropped) | 14 |
| T | (dropped) | 14 |
| F | (dropped) | 13 |
| S | (dropped) | 11 |
| B | (dropped) | 10 |

Extra letters added (not in the word): F×55, E×13, L×11, I×11, B×9, K×8, A×7, O×7, H×6, T×5.

## Inference time

One clip at a time on this computer's CPU, after the clip has ended (495 clips):

| Mean | Median | 95th percentile | Slowest | Real-time factor |
|---|---|---|---|---|
| 0.9 ms | 0.9 ms | 1.2 ms | 1.7 ms | 0.0002 |

Real-time factor = compute time / video length; below 1 means faster than real time. Both models read the whole clip, so they need the whole word before it can answer: in the live app the delay is the time to the end of the word plus this inference time.
