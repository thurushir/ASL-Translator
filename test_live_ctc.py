"""
Tests for live_ctc.py: cutting a stream of webcam frames into words, and predicting them exactly the way
the recorded clips are predicted offline. No webcam, real data or trained model needed.
Run: python test_live_ctc.py
"""
import os
import tempfile
import numpy as np
import torch
import dataset as D
import features as F
import model_ctc as MC
import train_ctc as T
import live_ctc as L


def check(name, got, want):
    print(f"{'ok  ' if got == want else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return got == want


rng = np.random.default_rng(0)


def hand():
    return rng.random((21, 3))


def feed(builder, frames, fps=30, start=100.0):
    """frames: list of 'H' (hand) / '.' (no hand). Returns the clips the builder finished along the way."""
    done = []
    for i, f in enumerate(frames):
        clip = builder.add(hand() if f == "H" else None, start + i / fps)
        if clip is not None:
            done.append(clip)
    return done


results = []

# --- WordBuilder: cutting the stream into words ---
b = L.WordBuilder()
results.append(check("no hand ever: no words", feed(b, "." * 90), []))
results.append(check("no hand ever: nothing pending", b.finish(), None))

b = L.WordBuilder()
words = feed(b, "H" * 20 + "." * 31)  # 31 empty frames at 30 fps = just over 1 s
results.append(check("1 s without a hand ends the word", len(words), 1))
results.append(check("trailing no-hand frames are trimmed", len(words[0].mask), 20))
results.append(check("timestamps start at 0", float(words[0].timestamps[0]), 0.0))
results.append(check("fps from the real frame times", round(words[0].fps), 30))

b = L.WordBuilder()
results.append(check("a 0.5 s gap does not end the word", feed(b, "H" * 20 + "." * 15 + "H" * 20), []))
clip = b.finish()
results.append(check("finish() returns the pending word", clip is not None, True))
results.append(check("the gap is kept inside the word, like recorded clips",
                     clip.mask.tolist() == [True] * 20 + [False] * 15 + [True] * 20, True))
results.append(check("no-hand frames are NaN landmarks and zero features",
                     (bool(np.isnan(clip.landmarks[~clip.mask]).all()), bool(clip.features[~clip.mask].any())),
                     (True, False)))
results.append(check("features come from features.py (same as training)",
                     bool(np.allclose(clip.features, F.sequence_features(clip.landmarks)[0])), True))
results.append(check("after finish() nothing is pending", b.finish(), None))

b = L.WordBuilder()
words = feed(b, "H" * 20 + "." * 40 + "H" * 25 + "." * 40)
results.append(check("two words separated by a gap", [len(c.mask) for c in words], [20, 25]))

b = L.WordBuilder()
results.append(check("a hand seen for only 3 frames is ignored", feed(b, "." * 10 + "H" * 3 + "." * 40), []))

b = L.WordBuilder()
feed(b, "...HHHHHHHHHH")
results.append(check("leading no-hand frames are not part of the word", len(b.finish().mask), 10))

b = L.WordBuilder()
feed(b, "H" * 12)
results.append(check("in_word is True while a word is being built", b.in_word, True))
b.finish()
results.append(check("in_word is False after finish()", b.in_word, False))

# --- Recognizer + replay: a recorded clip fed frame by frame gives exactly the offline prediction ---
torch.manual_seed(0)
model = MC.make_model("lstm", MC.INPUT_SIZE, hidden=16, layers=2, dropout=0.0).eval()  # untrained is fine:
mean, std = rng.random(64), rng.random(64) + 0.5                                       # we compare outputs exactly
rec = L.Recognizer(model, (mean, std), fps=15)

with tempfile.TemporaryDirectory() as root:
    n = 88  # ~3 s at ~29 fps with uneven frame times, like anna's camera
    ts = np.cumsum(np.r_[0, rng.uniform(0.030, 0.040, n - 1)])
    lm = rng.random((n, 21, 3))
    lm[[30, 31, 32, 60]] = np.nan  # the hand briefly lost mid-word
    os.makedirs(os.path.join(root, "HELLO"))
    path = os.path.join(root, "HELLO", "anna_0.npz")
    np.savez_compressed(path, landmarks=lm, timestamps=ts, word="HELLO", signer="anna", mirrored=True)

    offline_clip = D.resample(D.load_clip(path), 15)
    offline_word = T.predict(model, [offline_clip], (mean, std))[0]
    with torch.no_grad():
        x = torch.from_numpy(MC.clip_inputs(offline_clip, mean, std)).unsqueeze(0)
        offline_lp = model(x, torch.tensor([x.shape[1]]))[0].numpy()

    b = L.WordBuilder()
    for t, points in zip(ts, lm):
        assert b.add(None if np.isnan(points).any() else points, 5000.0 + t) is None
    live_clip = b.finish()
    live_lp = rec.log_probs(live_clip)
    results.append(check("replay: same number of frames as offline", live_lp.shape, offline_lp.shape))
    results.append(check("replay: same per-frame outputs as offline", bool(np.allclose(live_lp, offline_lp, atol=1e-6)), True))
    results.append(check("replay: same word as train_ctc.predict", rec.greedy(live_clip), offline_word))

results.append(check("log_probs are (frames, 27)", rec.log_probs(live_clip).shape[1], D.NUM_CLASSES))

print("\nPASS: all live_ctc tests passed" if all(results) else "\nFAIL: some live_ctc tests failed")
