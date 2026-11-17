"""
Tests for kaggle_data.py using a tiny fake Kaggle file written to a temp folder (no real data needed).
Run: python test_kaggle_data.py
"""
import os
import tempfile
import numpy as np
import pandas as pd
import dataset as D
import features as F
import kaggle_data as K


def check(name, got, want):
    print(f"{'ok  ' if got == want else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return got == want


results = []

# --- phrases: letters and spaces only; spaces dropped (our model has no space output) ---
results.append(check("letters only", K.clean_phrase("coffee"), "COFFEE"))
results.append(check("spaces dropped", K.clean_phrase("sheree henson"), "SHEREEHENSON"))
results.append(check("digits rejected", K.clean_phrase("1995 dalmar"), None))
results.append(check("symbols rejected", K.clean_phrase("wap.meinv8.org/x"), None))
results.append(check("hyphen rejected", K.clean_phrase("jean-luc"), None))
results.append(check("only spaces rejected", K.clean_phrase("   "), None))


def hand(rng, frames, present):
    """(frames, 21, 3) random hand points, NaN on frames where present is False."""
    a = rng.random((frames, 21, 3)) * 0.2 + 0.4
    a[~np.asarray(present)] = np.nan
    return a


rng = np.random.default_rng(0)
T = 10
right = hand(rng, T, [True] * 8 + [False] * 2)
left = hand(rng, T, [False] * 7 + [True] * 3)

# --- one sequence -> one Clip, built like dataset.load_clip() ---
clip = K.sequence_to_clip(np.arange(T), right, left, "hi", participant_id=7, path="kaggle/1/42")
results.append(check("word and labels", (clip.word, clip.labels), ("HI", [8, 9])))
results.append(check("signer is marked as Kaggle", clip.signer, "kaggle_7"))
results.append(check("features are (frames, 64)", clip.features.shape, (T, F.FEATURE_SIZE)))
results.append(check("uses the hand that appears in more frames (right, 8 vs 3)", int(clip.mask.sum()), 8))
results.append(check("timestamps 1/30 s apart from frame numbers", bool(np.allclose(np.diff(clip.timestamps), 1 / 30)), True))
results.append(check("starts at 0 s", float(clip.timestamps[0]), 0.0))
results.append(check("fps is 30", round(clip.fps), 30))

# right hand is mirrored (x -> 1 - x) and y is stretched
want = right[0].copy()
want[:, 0] = 1 - want[:, 0]
want[:, 1] *= K.ASPECT_STRETCH
results.append(check("right hand mirrored, y stretched", bool(np.allclose(clip.landmarks[0], want)), True))
results.append(check("features come from the converted points",
                     bool(np.allclose(clip.features[0], F.landmarks_to_features(want))), True))
results.append(check("no-hand frames stay NaN in landmarks and zero in features",
                     (bool(np.isnan(clip.landmarks[9]).all()), bool(clip.features[9].any())), (True, False)))

# a left-handed signer: not mirrored, still stretched
lefty = K.sequence_to_clip(np.arange(T), hand(rng, T, [False] * T), right, "hi", participant_id=8, path="p")
want = right[0].copy()
want[:, 1] *= K.ASPECT_STRETCH
results.append(check("left hand not mirrored, y stretched", bool(np.allclose(lefty.landmarks[0], want)), True))

# gaps in the frame numbers keep their real spacing
gappy = K.sequence_to_clip(np.array([0, 1, 2, 5, 6, 7, 8, 9, 10, 11]), right, left, "hi", participant_id=7, path="p")
results.append(check("a jump in frame numbers keeps its time gap", round(float(gappy.timestamps[3]), 4), round(5 / 30, 4)))

# the clip works with the rest of the pipeline
res = D.resample(clip, 15)
results.append(check("resamples to 15 fps (about half the frames)", len(res.mask), 5))

# sequences that can't be used are skipped
results.append(check("unusable phrase -> None",
                     K.sequence_to_clip(np.arange(T), right, left, "123 main", participant_id=7, path="p"), None))
few = hand(rng, T, [True] * 3 + [False] * 7)
results.append(check("fewer hand frames than CTC needs -> None (HELLO needs 6)",
                     K.sequence_to_clip(np.arange(T), few, hand(rng, T, [False] * T), "hello", participant_id=7, path="p"),
                     None))


# --- reading a whole file, in Kaggle's layout (sequence_id is the table's index) ---
def write_fake_file(path, sequences):
    """sequences: {sequence_id: (right, left)}. Same wide layout as Kaggle, plus a face column we must ignore."""
    rows = []
    for sid, (r, l) in sequences.items():
        for t in range(len(r)):
            row = {"sequence_id": sid, "frame": t, "x_face_0": 0.5}
            for name, a in (("right", r), ("left", l)):
                for i in range(21):
                    for k, axis in enumerate("xyz"):
                        row[f"{axis}_{name}_hand_{i}"] = a[t, i, k]
            rows.append(row)
    pd.DataFrame(rows).set_index("sequence_id").to_parquet(path)


with tempfile.TemporaryDirectory() as root:
    nohand = hand(rng, T, [False] * T)
    write_fake_file(os.path.join(root, "111.parquet"), {1: (right, left), 2: (right, left), 3: (right, nohand)})
    write_fake_file(os.path.join(root, "222.parquet"), {4: (right, left)})
    pd.DataFrame({"path": ["train_landmarks/111.parquet"] * 3 + ["train_landmarks/222.parquet", "train_landmarks/333.parquet"],
                  "file_id": [111, 111, 111, 222, 333], "sequence_id": [1, 2, 3, 4, 5],
                  "participant_id": [10, 11, 10, 12, 13],
                  "phrase": ["hi there", "+1-555", "no", "yes", "cat"]}).to_csv(os.path.join(root, "train.csv"), index=False)

    clips = K.load_file(os.path.join(root, "111.parquet"), pd.read_csv(os.path.join(root, "train.csv")))
    results.append(check("file: usable sequences only, in order", [c.word for c in clips], ["HITHERE", "NO"]))
    results.append(check("file: signers come from train.csv", [c.signer for c in clips], ["kaggle_10", "kaggle_10"]))
    results.append(check("file: path names the file and sequence", clips[0].path, "kaggle/111/1"))

    every = K.load_kaggle(root)
    results.append(check("folder: reads every downloaded file, skips listed files not downloaded",
                         [c.word for c in every], ["HITHERE", "NO", "YES"]))
    results.append(check("folder: no Kaggle signer looks like one of ours",
                         any(not c.signer.startswith("kaggle_") for c in every), False))

print("\nPASS: all kaggle_data tests passed" if all(results) else "\nFAIL: some kaggle_data tests failed")
