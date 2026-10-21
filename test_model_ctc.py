"""
Tests for model_ctc.py: input preparation, the LSTM's output shape, and the greedy CTC decoder.
No real data or webcam needed. Run: python test_model_ctc.py
"""
import numpy as np
import torch
import dataset as D
import model_ctc as MC


def check(name, got, want):
    print(f"{'ok  ' if got == want else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return got == want


def frames_for(letters):
    """Hand-made model output: one frame per entry, all the probability on that entry (0 = blank)."""
    out = torch.full((len(letters), D.NUM_CLASSES), -1e9)
    for t, k in enumerate(letters):
        out[t, k] = 0.0  # log(1)
    return out


def L(ch):
    return D.encode(ch)[0]


results = []

# --- inputs: 64 standardized features + 1 "hand present" flag per frame ---
rng = np.random.default_rng(0)
feats = rng.random((6, 64))
mask = np.array([True, True, False, True, False, True])
feats[~mask] = 0
clip = D.Clip(path="x", word="HI", signer="anna", labels=[8, 9], features=feats, mask=mask,
              timestamps=np.arange(6) / 15, fps=15.0)
mean, std = MC.feature_stats([clip])
x = MC.clip_inputs(clip, mean, std)
results.append(check("inputs are (frames, 65)", x.shape, (6, 65)))
results.append(check("last column is the hand-present flag", x[:, -1].tolist(), [1, 1, 0, 1, 0, 1]))
results.append(check("no-hand frames are all zeros", bool(x[~mask].any()), False))
# float32 inputs: compare with a float32-sized tolerance (the default 1e-8 is below float32 rounding)
results.append(check("hand frames are standardized (mean 0)",
                     bool(np.allclose(x[mask, :64].mean(0), 0, atol=1e-6)), True))
results.append(check("hand frames are standardized (std 1)",
                     bool(np.allclose(x[mask, :64].std(0), 1, atol=1e-5)), True))
const = D.Clip(path="y", word="HI", signer="anna", labels=[8, 9], features=np.ones((3, 64)),
               mask=np.ones(3, bool), timestamps=np.arange(3) / 15, fps=15.0)
_, const_std = MC.feature_stats([const])
results.append(check("a constant feature doesn't cause division by zero", bool((const_std > 0).all()), True))

# --- model output ---
torch.manual_seed(0)
model = MC.CTCModel().eval()  # eval: dropout off, so outputs are repeatable
batch = torch.randn(3, 10, MC.INPUT_SIZE)
lengths = torch.tensor([10, 7, 4])
out = model(batch, lengths)
results.append(check("output is (batch, frames, 27)", tuple(out.shape), (3, 10, 27)))
results.append(check("each frame's outputs are log-probabilities (sum to 1)",
                     bool(torch.allclose(out.exp().sum(-1), torch.ones(3, 10), atol=1e-5)), True))
alone = model(batch[2:3, :4], torch.tensor([4]))
results.append(check("padding doesn't change a short clip's outputs",
                     bool(torch.allclose(out[2, :4], alone[0], atol=1e-5)), True))

# --- greedy decoder: best output per frame -> merge repeats -> drop blanks ---
H, I, LL, O, B = L("H"), L("I"), L("L"), L("O"), D.BLANK
results.append(check("H H blank I -> HI", MC.greedy_decode(frames_for([H, H, B, I])), "HI"))
results.append(check("repeats merge: H H H I I -> HI", MC.greedy_decode(frames_for([H, H, H, I, I])), "HI"))
results.append(check("double letter needs a blank: L blank L -> LL", MC.greedy_decode(frames_for([LL, B, LL])), "LL"))
results.append(check("without the blank it's one letter: L L -> L", MC.greedy_decode(frames_for([LL, LL])), "L"))
results.append(check("leading/trailing blanks ignored", MC.greedy_decode(frames_for([B, B, H, I, B])), "HI"))
results.append(check("all blanks -> empty word", MC.greedy_decode(frames_for([B, B, B])), ""))
hello = [B, H, E := L("E"), B, LL, LL, B, LL, O, O, B]
results.append(check("HELLO with a blank between the L's", MC.greedy_decode(frames_for(hello)), "HELLO"))
padded = torch.stack([frames_for([H, I, B, B]), frames_for([O, B, LL, LL])])
results.append(check("batch decode stops at each clip's length",
                     MC.decode_batch(padded, torch.tensor([2, 4])), ["HI", "OL"]))

print("\nPASS: all model_ctc tests passed" if all(results) else "\nFAIL: some model_ctc tests failed")
