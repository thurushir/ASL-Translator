"""
Tests for dataset.py using small fake clips written to a temp folder (no real data needed).
Run: python test_dataset.py
"""
import os
import tempfile
import numpy as np
import dataset as D


def check(name, got, want):
    print(f"{'ok  ' if got == want else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return got == want


def fake_hand(rng):
    return rng.random((21, 3))


def write_clip(root, word, signer, n, frames, missing=(), secs=1.0):
    """Saves a fake clip in the same format collect_sequences.py uses."""
    rng = np.random.default_rng(n)
    lm = np.array([fake_hand(rng) for _ in range(frames)], dtype=np.float32)
    lm[list(missing)] = np.nan
    os.makedirs(os.path.join(root, word), exist_ok=True)
    np.savez_compressed(os.path.join(root, word, f"{signer}_{n}.npz"), landmarks=lm,
                        timestamps=np.linspace(0, secs, frames), word=word, signer=signer, mirrored=True)


results = []

# --- letters <-> numbers (0 is kept free for the CTC blank) ---
results.append(check("encode HELLO", D.encode("HELLO"), [8, 5, 12, 12, 15]))
results.append(check("decode back to HELLO", D.decode([8, 5, 12, 12, 15]), "HELLO"))
results.append(check("A is 1 and Z is 26", (D.encode("A"), D.encode("Z")), ([1], [26])))
results.append(check("lowercase is uppercased", D.encode("hi"), [8, 9]))
results.append(check("decode skips the blank (0)", D.decode([0, 8, 0, 9, 0]), "HI"))
try:
    D.encode("HI!")
    results.append(check("non-letter raises an error", "no error", "ValueError"))
except ValueError:
    results.append(check("non-letter raises an error", "ValueError", "ValueError"))

with tempfile.TemporaryDirectory() as root:
    # --- one clip ---
    write_clip(root, "CAT", "anna", 0, frames=5, missing=[1, 3], secs=0.2)
    clip = D.load_clip(os.path.join(root, "CAT", "anna_0.npz"))
    results.append(check("word and signer", (clip.word, clip.signer), ("CAT", "anna")))
    results.append(check("labels match the word", clip.labels, [3, 1, 20]))
    results.append(check("features are (frames, 64)", clip.features.shape, (5, 64)))
    results.append(check("mask marks frames with a hand", clip.mask.tolist(), [True, False, True, False, True]))
    results.append(check("missing frames are all zeros", bool(clip.features[[1, 3]].any()), False))
    results.append(check("fps from timestamps (5 frames over 0.2s)", round(clip.fps), 20))
    results.append(check("hand_only() drops missing frames", clip.hand_only().shape, (3, 64)))

    # --- a folder of clips: 3 signers x 2 words x 2 takes ---
    for signer in ["anna", "saaya", "thurushi"]:
        for word in ["CAT", "DOG"]:
            for n in range(2):
                if not (signer == "anna" and word == "CAT" and n == 0):  # already written above
                    write_clip(root, word, signer, n, frames=6)
    clips = D.load_clips(root)
    results.append(check("loads every clip", len(clips), 12))
    results.append(check("signers found", D.signers(clips), ["anna", "saaya", "thurushi"]))

    # --- held-out-signer splits ---
    folds = D.signer_splits(clips)
    results.append(check("one fold per signer", [f.test_signer for f in folds], ["anna", "saaya", "thurushi"]))
    results.append(check("test signer never in training",
                         all(f.test_signer not in {c.signer for c in f.train} for f in folds), True))
    results.append(check("test set is only the test signer",
                         all({c.signer for c in f.test} == {f.test_signer} for f in folds), True))
    results.append(check("train + test = all clips in every fold",
                         all(len(f.train) + len(f.test) == len(clips) for f in folds), True))
    test_paths = sorted(c.path for f in folds for c in f.test)
    results.append(check("every clip is test data exactly once",
                         test_paths == sorted(c.path for c in clips), True))

    # --- resampling to one frame rate (anna records at ~29 fps, the others at ~15) ---
    write_clip(root, "HI", "anna", 0, frames=61, secs=2.0)        # 30 fps for 2 s
    fast = D.load_clip(os.path.join(root, "HI", "anna_0.npz"))
    slow = D.resample(fast, fps=15)
    results.append(check("30 fps clip resampled to 15 fps has about half the frames", len(slow.mask), 31))
    results.append(check("resampled fps is 15", slow.fps, 15))
    results.append(check("resampled timestamps are 1/15 s apart",
                         bool(np.allclose(np.diff(slow.timestamps), 1 / 15)), True))
    results.append(check("each resampled frame is a real original frame (nearest, not averaged)",
                         all(any(np.array_equal(f, g) for g in fast.features) for f in slow.features), True))
    results.append(check("word, signer and labels unchanged", (slow.word, slow.signer, slow.labels),
                         (fast.word, fast.signer, fast.labels)))
    results.append(check("original clip not changed", len(fast.mask), 61))

    write_clip(root, "HI", "saaya", 0, frames=31, secs=2.0)       # already 15 fps
    same = D.load_clip(os.path.join(root, "HI", "saaya_0.npz"))
    results.append(check("15 fps clip is unchanged by resampling to 15 fps",
                         bool(np.array_equal(D.resample(same, fps=15).features, same.features)), True))

    write_clip(root, "HI", "thurushi", 0, frames=61, missing=range(20, 40), secs=2.0)
    gappy = D.resample(D.load_clip(os.path.join(root, "HI", "thurushi_0.npz")), fps=15)
    results.append(check("mask follows the frames: hand missing from 0.67 s to 1.3 s",
                         [round(t, 2) for t in gappy.timestamps[~gappy.mask]],
                         [0.67, 0.73, 0.8, 0.87, 0.93, 1.0, 1.07, 1.13, 1.2, 1.27]))
    results.append(check("missing frames are still all zeros after resampling",
                         bool(gappy.features[~gappy.mask].any()), False))

print("\nPASS: all dataset tests passed" if all(results) else "\nFAIL: some dataset tests failed")
