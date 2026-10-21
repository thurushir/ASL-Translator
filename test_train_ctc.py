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

print("\nPASS: all train_ctc tests passed" if all(results) else "\nFAIL: some train_ctc tests failed")
