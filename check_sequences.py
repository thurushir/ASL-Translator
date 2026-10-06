"""
Summarizes the fingerspelling clips in data_seq/ (made by collect_sequences.py).
Run: python check_sequences.py
Flags clips where a hand was found in fewer than 80% of frames.
"""
import os
import numpy as np

DIR = "./data_seq"
LOW_HAND_PCT = 80

if not os.path.isdir(DIR):
    raise SystemExit(f"No {DIR} folder yet. Record clips with collect_sequences.py first.")

total = 0
flagged = []
print(f"{'Word':<12}{'Clips':>6}{'Avg frames':>12}{'Avg secs':>10}{'Avg fps':>9}{'Hand %':>8}")
for word in sorted(os.listdir(DIR)):
    folder = os.path.join(DIR, word)
    if not os.path.isdir(folder):
        continue
    stats = []
    for name in sorted(f for f in os.listdir(folder) if f.endswith(".npz")):
        clip = np.load(os.path.join(folder, name))
        lm, ts = clip["landmarks"], clip["timestamps"]
        secs = float(ts[-1]) if len(ts) else 0.0
        hand_pct = 100 * (~np.isnan(lm[:, 0, 0])).mean()
        stats.append((len(lm), secs, len(lm) / secs if secs else 0.0, hand_pct))
        if hand_pct < LOW_HAND_PCT:
            flagged.append(f"{word}/{name} (hand in {hand_pct:.0f}% of frames)")
    if not stats:
        continue
    total += len(stats)
    avg = np.mean(stats, axis=0)
    print(f"{word:<12}{len(stats):>6}{avg[0]:>12.0f}{avg[1]:>10.1f}{avg[2]:>9.0f}{avg[3]:>8.0f}")

print(f"\nTotal clips: {total}")
if flagged:
    print(f"\nClips with a hand in < {LOW_HAND_PCT}% of frames (consider re-recording):")
    for f in flagged:
        print("  " + f)
