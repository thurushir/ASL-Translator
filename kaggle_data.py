"""
Turns the Google ASL Fingerspelling data from Kaggle (Step 7) into the same Clip format as our own clips,
so the training code can pretrain on many more signers and words.

Download from https://www.kaggle.com/competitions/asl-fingerspelling/data into data_kaggle/ (ignored by git):
train.csv (which phrase each recording is, and who signed it) and one or more train_landmarks/<file_id>.parquet
(the recordings: MediaPipe points for every frame, about 1,000 recordings per file).

  clean_phrase("sheree henson")  -> "SHEREEHENSON"; None if the phrase has digits or symbols
  sequence_to_clip(...)          -> one recording as a dataset.Clip, or None if it can't be used
  load_file(path, labels)        -> every usable recording in one .parquet file
  load_kaggle()                  -> every usable recording in every downloaded file

Conversion rules, fixed before any training results (see the Step 7 feasibility check):
  - phrases with letters and spaces only; spaces dropped (our model has no space output)
  - the hand seen in more frames is the signing hand; right hands are mirrored to match our mirrored clips
  - y is stretched by ASPECT_STRETCH: Kaggle was filmed on portrait phones and MediaPipe measures x and y as
    fractions of the image width and height, so their hands come out squashed sideways compared with our
    landscape webcams. 3.16 = (16/9)^2 for 16:9 cameras; it cut the feature gap with our clips from 1.90 to 0.43
  - no timestamps in the files: frames are assumed to be KAGGLE_FPS apart
  - signers are named kaggle_<participant_id>, so they never mix with ours (or become a test signer)

Run: python kaggle_data.py   (summary of the downloaded data)      Tests: python test_kaggle_data.py
"""
import os
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import dataset as D
import features as F

KAGGLE_DIR = "./data_kaggle"
ASPECT_STRETCH = 3.16  # (16/9)^2: portrait phone video vs our 16:9 landscape webcams
KAGGLE_FPS = 30.0      # assumed; speed augmentation covers the uncertainty
HAND_COLUMNS = [f"{axis}_{h}_hand_{i}" for h in ("right", "left") for axis in "xyz" for i in range(F.NUM_LANDMARKS)]


def clean_phrase(phrase):
    """'sheree henson' -> 'SHEREEHENSON'. None unless the phrase is letters and spaces only."""
    if not re.fullmatch(r"[A-Za-z ]*[A-Za-z][A-Za-z ]*", str(phrase)):
        return None
    return phrase.replace(" ", "").upper()


def frames_needed(word):
    """Fewest frames CTC can spell the word in: one per letter, plus a blank between each double letter."""
    return len(word) + sum(a == b for a, b in zip(word, word[1:]))


def _present(points):
    return ~np.isnan(points).any(axis=(1, 2))


def sequence_to_clip(frames, right, left, phrase, participant_id, path):
    """
    One Kaggle recording -> dataset.Clip, converted by the rules at the top of this file.
    frames: (T,) frame numbers; right, left: (T, 21, 3) points, NaN where that hand wasn't found.
    None if the phrase isn't usable or the hand isn't seen in enough frames to spell it.
    """
    word = clean_phrase(phrase)
    if word is None:
        return None
    use_right = _present(right).sum() >= _present(left).sum()
    points = np.array(right if use_right else left, dtype=np.float64)  # copy: never change the caller's arrays
    if use_right:
        points[:, :, 0] = 1 - points[:, :, 0]
    points[:, :, 1] *= ASPECT_STRETCH
    feats, mask = F.sequence_features(points)
    if mask.sum() < frames_needed(word):
        return None
    frames = np.asarray(frames, dtype=np.float64)
    ts = (frames - frames[0]) / KAGGLE_FPS
    fps = (len(ts) - 1) / ts[-1] if len(ts) > 1 and ts[-1] > 0 else 0.0
    return D.Clip(path=path, word=word, signer=f"kaggle_{participant_id}", labels=D.encode(word),
                  features=feats, mask=mask, timestamps=ts, fps=float(fps), landmarks=points)


def load_file(path, labels):
    """Every usable recording in one .parquet file. labels: train.csv as a DataFrame."""
    file_id = int(os.path.splitext(os.path.basename(path))[0])
    mine = labels[labels.file_id == file_id].set_index("sequence_id")
    # sequence_id is stored as the table's index; it must be asked for by name, or pyarrow silently drops it
    table = pq.read_table(path, columns=["sequence_id", "frame"] + HAND_COLUMNS).to_pandas()
    if "sequence_id" not in table.columns:
        table = table.reset_index()
    clips = []
    for sid, df in table.groupby("sequence_id", sort=False):
        if sid not in mine.index or clean_phrase(mine.at[sid, "phrase"]) is None:
            continue
        df = df.sort_values("frame")
        hands = {h: np.stack([df[[f"{axis}_{h}_hand_{i}" for i in range(F.NUM_LANDMARKS)]].to_numpy()
                              for axis in "xyz"], axis=-1) for h in ("right", "left")}
        clip = sequence_to_clip(df.frame.to_numpy(), hands["right"], hands["left"], mine.at[sid, "phrase"],
                                int(mine.at[sid, "participant_id"]), f"kaggle/{file_id}/{sid}")
        if clip is not None:
            clips.append(clip)
    return clips


def load_kaggle(root=KAGGLE_DIR):
    """Every usable recording in every downloaded .parquet file under root (or root/train_landmarks/)."""
    labels = pd.read_csv(os.path.join(root, "train.csv"))
    files = sorted(os.path.join(d, f) for d in (root, os.path.join(root, "train_landmarks")) if os.path.isdir(d)
                   for f in os.listdir(d) if f.endswith(".parquet"))
    return [c for path in files for c in load_file(path, labels)]


if __name__ == "__main__":
    clips = load_kaggle()
    ours = D.load_clips()
    print(f"Kaggle: {len(clips)} usable recordings, {len(D.signers(clips))} signers, "
          f"{len({c.word for c in clips})} different phrases")
    print(f"letters per phrase: median {np.median([len(c.word) for c in clips]):.0f}; "
          f"hand in {100 * np.mean([c.mask.mean() for c in clips]):.0f}% of frames (ours: "
          f"{100 * np.mean([c.mask.mean() for c in ours]):.0f}%)")
    ours_f = np.concatenate([c.hand_only() for c in ours])
    mu, sd = ours_f.mean(0), ours_f.std(0) + 1e-6
    gap = lambda f: float(np.mean(np.abs(f.mean(0) - mu) / sd))
    print(f"feature gap from our clips: Kaggle {gap(np.concatenate([c.hand_only() for c in clips])):.2f} "
          f"(our own signers: " + ", ".join(f"{s} {gap(np.concatenate([c.hand_only() for c in ours if c.signer == s])):.2f}"
                                            for s in D.signers(ours)) + ")")
    frames15 = sum(len(D.resample(c, 15).mask) for c in clips)
    print(f"{frames15:,} frames at 15 fps (ours: {sum(len(D.resample(c, 15).mask) for c in ours):,})")
    print("examples:", ", ".join(c.word for c in clips[:8]))
