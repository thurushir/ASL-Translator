"""
Summarizes the fingerspelling clips in data_seq/ (made by collect_sequences.py).

Run: python check_sequences.py [--list word_list.txt] [--takes 3]
Shows: clips per word, clips per signer (and list progress with --list),
how often each letter appears, and clips where the hand was often lost.
"""
import argparse
import os
import re
from collections import Counter, defaultdict
import numpy as np

DIR = "./data_seq"
LOW_HAND_PCT = 80      # flag clips with a hand in fewer than this % of frames
LOW_LETTER_COUNT = 15  # flag letters that appear fewer times than this across all clips


def clip_signer(clip, filename):
    """Signer stored in the clip, else the <signer>_<n>.npz file-name prefix, else 'unknown'."""
    if "signer" in clip.files:
        return str(clip["signer"])
    m = re.match(r"^(.+)_\d+\.npz$", filename)
    return m.group(1) if m else "unknown"


def load_clips(root=DIR):
    """List of dicts: word, signer, name, frames, secs, hand_pct for every clip."""
    clips = []
    for word in sorted(os.listdir(root)):
        folder = os.path.join(root, word)
        if not os.path.isdir(folder):
            continue
        for name in sorted(f for f in os.listdir(folder) if f.endswith(".npz")):
            clip = np.load(os.path.join(folder, name))
            lm, ts = clip["landmarks"], clip["timestamps"]
            clips.append({
                "word": word, "signer": clip_signer(clip, name), "name": name,
                "frames": len(lm), "secs": float(ts[-1]) if len(ts) else 0.0,
                "hand_pct": 100 * (~np.isnan(lm[:, 0, 0])).mean() if len(lm) else 0.0,
            })
    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", help="word list to report progress against (e.g. word_list.txt)")
    parser.add_argument("--takes", type=int, default=3, help="target clips per word per signer (default 3)")
    args = parser.parse_args()

    if not os.path.isdir(DIR):
        raise SystemExit(f"No {DIR} folder yet. Record clips with collect_sequences.py first.")
    clips = load_clips()
    if not clips:
        raise SystemExit(f"No clips in {DIR} yet.")

    # per word
    by_word = defaultdict(list)
    for c in clips:
        by_word[c["word"]].append(c)
    print(f"{'Word':<12}{'Clips':>6}{'Avg frames':>12}{'Avg secs':>10}{'Avg fps':>9}{'Hand %':>8}")
    for word, cs in by_word.items():
        frames = np.mean([c["frames"] for c in cs])
        secs = np.mean([c["secs"] for c in cs])
        fps = np.mean([c["frames"] / c["secs"] for c in cs if c["secs"]] or [0])
        hand = np.mean([c["hand_pct"] for c in cs])
        print(f"{word:<12}{len(cs):>6}{frames:>12.0f}{secs:>10.1f}{fps:>9.0f}{hand:>8.0f}")
    print(f"\nTotal clips: {len(clips)} across {len(by_word)} words")

    # per signer (+ progress against the word list)
    words = None
    if args.list:
        with open(args.list) as f:
            words = [l.strip().upper() for l in f if l.strip() and not l.startswith("#")]
    print("\nClips per signer:")
    for signer, n in sorted(Counter(c["signer"] for c in clips).items()):
        line = f"  {signer:<12}{n:>5} clips"
        if words:
            have = Counter(c["word"] for c in clips if c["signer"] == signer)
            done = sum(have[w] >= args.takes for w in words)
            line += f"   {done}/{len(words)} words with {args.takes}+ takes"
        print(line)

    # letter coverage: each clip counts every letter in its word
    letters = Counter("".join(c["word"] for c in clips))
    print("\nLetter coverage (times each letter appears across all clips):")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    print("  " + "  ".join(f"{L}:{letters[L]}" for L in alphabet[:13]))
    print("  " + "  ".join(f"{L}:{letters[L]}" for L in alphabet[13:]))
    low = [L for L in alphabet if letters[L] < LOW_LETTER_COUNT]
    if low:
        print(f"  Under {LOW_LETTER_COUNT}: {', '.join(low)} (record more words with these)")

    flagged = [c for c in clips if c["hand_pct"] < LOW_HAND_PCT]
    if flagged:
        print(f"\nClips with a hand in < {LOW_HAND_PCT}% of frames (consider re-recording):")
        for c in flagged:
            print(f"  {c['word']}/{c['name']} (hand in {c['hand_pct']:.0f}% of frames)")


if __name__ == "__main__":
    main()
