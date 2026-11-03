"""
Tests for the helpers in train_ctc.py (batching, validation split, junk-letter check). No real data needed.
The end-to-end check is on real data: python train_ctc.py --overfit 10   (should memorize the 10 clips)
Run: python test_train_ctc.py
"""
import numpy as np
import torch
import dataset as D
import model_ctc as MC
import train_ctc as T


def check(name, got, want):
    print(f"{'ok  ' if got == want else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return got == want


def fake_clip(word, signer, frames, n=0):
    rng = np.random.default_rng(frames + n)
    return D.Clip(path=f"{word}/{signer}_{n}.npz", word=word, signer=signer, labels=D.encode(word),
                  features=rng.random((frames, 64)), mask=np.ones(frames, bool),
                  timestamps=np.arange(frames) / 15, fps=15.0)


results = []

# --- batching ---
clips = [fake_clip("HELLO", "anna", 5), fake_clip("HI", "saaya", 3)]
mean, std = MC.feature_stats(clips)
x, lengths, targets, target_lengths = T.make_batch(clips, mean, std)
results.append(check("batch is padded to the longest clip", tuple(x.shape), (2, 5, MC.INPUT_SIZE)))
results.append(check("lengths are each clip's frames", lengths.tolist(), [5, 3]))
results.append(check("padding frames are zeros", bool(x[1, 3:].any()), False))
results.append(check("targets are the words' letters joined", targets.tolist(), D.encode("HELLO") + D.encode("HI")))
results.append(check("target lengths are the word lengths", target_lengths.tolist(), [5, 2]))

# --- validation split (taken from the training signers' clips only) ---
pool = [fake_clip(w, s, 4, n) for w in ["CAT", "DOG", "HI", "NO"] for s in ["anna", "saaya"] for n in range(3)]
fit, val = T.split_validation(pool, frac=0.15, seed=0)
results.append(check("validation is about 15% of the clips", len(val), round(0.15 * len(pool))))
results.append(check("fit and validation don't overlap", bool({c.path for c in fit} & {c.path for c in val}), False))
results.append(check("fit + validation = all clips", sorted(c.path for c in fit + val), sorted(c.path for c in pool)))
fit2, val2 = T.split_validation(pool, frac=0.15, seed=0)
results.append(check("same seed gives the same split", [c.path for c in val2], [c.path for c in val]))

# --- held-out words: removed from training entirely, for the "new words" check ---
kept, held = T.split_words(pool, {"CAT", "HI"})
results.append(check("held-out words are gone from training", sorted({c.word for c in kept}), ["DOG", "NO"]))
results.append(check("held-out clips are returned separately", sorted({c.word for c in held}), ["CAT", "HI"]))
results.append(check("nothing lost", len(kept) + len(held), len(pool)))

# --- junk letter at the start of a prediction (anna's resting hand read as B in the baseline) ---
results.append(check("BHI for HI has a junk first letter", T.junk_start("HI", "BHI"), True))
results.append(check("HI for HI has none", T.junk_start("HI", "HI"), False))
results.append(check("a dropped letter isn't junk", T.junk_start("HI", "I"), False))
results.append(check("a wrong first letter isn't junk (it's a swap)", T.junk_start("HI", "BI"), False))

# --- one training step lowers the loss on a tiny batch (the loop is wired correctly) ---
torch.manual_seed(0)
model = MC.CTCModel()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
first = T.train_step(model, opt, x, lengths, targets, target_lengths)
for _ in range(20):
    last = T.train_step(model, opt, x, lengths, targets, target_lengths)
results.append(check("CTC loss goes down after 20 steps on one batch", last < first, True))

# --- Step 5: per-clip timing and records (what evaluate.py reads) ---
stats = (mean, std)
words, ms = T.time_inference(model, clips, stats)
results.append(check("one timing per clip", len(ms), len(clips)))
results.append(check("timings are positive milliseconds", all(m > 0 for m in ms), True))
results.append(check("timed predictions match batch predictions", words, T.predict(model, clips, stats)))
records = T.E.prediction_records(clips, words, ms)
results.append(check("records carry word, signer and prediction",
                     [(r["word"], r["signer"], r["pred"]) for r in records],
                     [(c.word, c.signer, w) for c, w in zip(clips, words)]))
results.append(check("--save is off by default", T.parse_args([]).save, False))
results.append(check("--save can be turned on", T.parse_args(["--save", "--name", "x"]).save, True))
results.append(check("--motion is off by default", T.parse_args([]).motion, False))
results.append(check("--motion can be turned on", T.parse_args(["--motion"]).motion, True))

# --- Step 5: speed augmentation, training clips only ---
cfg = T.parse_args(["--speed-aug"])
results.append(check("--speed-aug on, default range 0.7-1.4", (cfg.speed_aug, cfg.speed_min, cfg.speed_max),
                     (True, 0.7, 1.4)))
results.append(check("--speed-aug is off by default", T.parse_args([]).speed_aug, False))
long_clips = [fake_clip("HELLO", "anna", 60, n) for n in range(20)]  # 4 s each
rng = np.random.default_rng(0)
aug = T.augment(long_clips, rng, 0.7, 1.4)
results.append(check("augmenting keeps every clip and its word", [c.word for c in aug], [c.word for c in long_clips]))
results.append(check("lengths land between 60/1.4 and 60/0.7 frames",
                     all(round(59 / 1.4) + 1 <= len(c.mask) <= round(59 / 0.7) + 1 for c in aug), True))
results.append(check("speeds actually vary", len({len(c.mask) for c in aug}) > 3, True))
results.append(check("same seed, same speeds",
                     [len(c.mask) for c in T.augment(long_clips, np.random.default_rng(0), 0.7, 1.4)],
                     [len(c.mask) for c in aug]))
results.append(check("originals untouched", {len(c.mask) for c in long_clips}, {60}))

print("\nPASS: all train_ctc tests passed" if all(results) else "\nFAIL: some train_ctc tests failed")
