"""
Loads the fingerspelling clips in data_seq/ (made by collect_sequences.py) for training and evaluation.
Every model and evaluation script should read clips through here so they all see the same data.

  load_clips()        -> list of Clip: per-frame features (T, 64), hand mask, letter labels, signer, fps
  signer_splits(clips) -> one Fold per signer: train on the other signers, test on that one
  encode / decode     -> "HELLO" <-> [8, 5, 12, 12, 15]  (A=1 ... Z=26; 0 is the CTC blank)

Run: python dataset.py   (prints a summary of the real data and the folds)
"""
import os
from dataclasses import dataclass, field
import numpy as np
import features as F

DIR = "./data_seq"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = 0                       # CTC "no letter" output; letters are 1..26
NUM_CLASSES = len(ALPHABET) + 1  # 27


def encode(word):
    """'HELLO' -> [8, 5, 12, 12, 15]. Raises ValueError on anything that isn't A-Z."""
    word = word.upper()
    bad = [ch for ch in word if ch not in ALPHABET]
    if bad:
        raise ValueError(f"{word!r} has characters outside A-Z: {bad}")
    return [ALPHABET.index(ch) + 1 for ch in word]


def decode(labels):
    """[8, 5, 12, 12, 15] -> 'HELLO'. Blanks (0) are skipped."""
    return "".join(ALPHABET[i - 1] for i in labels if i != BLANK)


@dataclass
class Clip:
    path: str
    word: str
    signer: str
    labels: list            # encode(word)
    features: np.ndarray    # (T, 64); zeros where no hand was found
    mask: np.ndarray        # (T,) True where a hand was found
    timestamps: np.ndarray  # (T,) seconds since the clip started
    fps: float

    def hand_only(self):
        """Features for just the frames where a hand was found."""
        return self.features[self.mask]


@dataclass
class Fold:
    test_signer: str
    train: list = field(default_factory=list)
    test: list = field(default_factory=list)


def load_clip(path):
    data = np.load(path)
    word = str(data["word"]).upper()
    feats, mask = F.sequence_features(data["landmarks"])
    ts = data["timestamps"]
    fps = (len(ts) - 1) / ts[-1] if len(ts) > 1 and ts[-1] > 0 else 0.0
    return Clip(path=path, word=word, signer=str(data["signer"]), labels=encode(word),
                features=feats, mask=mask, timestamps=ts, fps=float(fps))


def load_clips(root=DIR):
    """Every clip under root/<WORD>/*.npz, sorted by word then file name."""
    clips = []
    for word in sorted(os.listdir(root)):
        folder = os.path.join(root, word)
        if os.path.isdir(folder):
            clips += [load_clip(os.path.join(folder, f)) for f in sorted(os.listdir(folder)) if f.endswith(".npz")]
    return clips


def signers(clips):
    return sorted({c.signer for c in clips})


def signer_splits(clips):
    """Leave-one-signer-out: for each signer, test on their clips and train on everyone else's."""
    return [Fold(s, [c for c in clips if c.signer != s], [c for c in clips if c.signer == s])
            for s in signers(clips)]


if __name__ == "__main__":
    clips = load_clips()
    print(f"Loaded {len(clips)} clips, {len(signers(clips))} signers, {len({c.word for c in clips})} words")
    for s in signers(clips):
        mine = [c for c in clips if c.signer == s]
        print(f"  {s:<10}{len(mine):>4} clips   avg {np.mean([len(c.mask) for c in mine]):.0f} frames"
              f"   {np.mean([c.fps for c in mine]):.1f} fps   hand in {100 * np.mean([c.mask.mean() for c in mine]):.0f}% of frames")
    print("\nFolds (train on two signers, test on the third):")
    for f in signer_splits(clips):
        print(f"  test={f.test_signer:<10} train={len(f.train)}  test={len(f.test)}")
    mismatched = [c.path for c in clips if os.path.basename(os.path.dirname(c.path)) != c.word
                  or decode(c.labels) != c.word]
    print(f"\nLabel check: {'all labels match their folder names' if not mismatched else f'MISMATCH in {mismatched}'}")
    short = [(c.path, int(c.mask.sum()), len(c.word)) for c in clips if c.mask.sum() < len(c.word)]
    print(f"Clips with fewer hand frames than letters (CTC can't learn from these): {short or 'none'}")
