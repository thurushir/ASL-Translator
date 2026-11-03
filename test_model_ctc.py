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

# --- save and load (Step 5): a saved model must give exactly the same outputs after loading ---
import os
import tempfile
torch.manual_seed(0)
small = MC.CTCModel(hidden=16, layers=2, dropout=0.3).eval()
stats = (rng.random(64), rng.random(64) + 0.5)
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "m.pt")
    MC.save_model(path, small, stats, {"hidden": 16, "layers": 2, "dropout": 0.3, "fps": 15.0},
                  test_signer="anna", seed=3)
    loaded, loaded_stats, info = MC.load_model(path)
xin = torch.from_numpy(MC.clip_inputs(clip, *stats)).unsqueeze(0)
n = torch.tensor([xin.shape[1]])
with torch.no_grad():
    same = torch.equal(small(xin, n), loaded(xin, n))
results.append(check("loaded model gives identical outputs", same, True))
results.append(check("loaded model is ready to predict (eval mode)", loaded.training, False))
results.append(check("standardization numbers come back",
                     bool(np.allclose(loaded_stats[0], stats[0]) and np.allclose(loaded_stats[1], stats[1])), True))
results.append(check("settings come back", (info["config"]["hidden"], info["test_signer"], info["seed"]),
                     (16, "anna", 3)))

# --- motion features as extra inputs (Step 5, --motion) ---
import features as F
import motion as MO
lm = rng.random((8, 21, 3))
lm[[2, 5]] = np.nan  # two frames without a hand
feats8, mask8 = F.sequence_features(lm)
mclip = D.Clip(path="m", word="ZOO", signer="anna", labels=D.encode("ZOO"), features=feats8, mask=mask8,
               timestamps=np.arange(8) / 15, fps=15.0, landmarks=lm)
plain_stats = MC.feature_stats([mclip])
motion_stats = MC.feature_stats([mclip], motion=True)
results.append(check("motion stats cover 64 + 14 numbers", len(motion_stats[0]), F.FEATURE_SIZE + MO.NUM_MOTION))
xm = MC.clip_inputs(mclip, *motion_stats)
xp = MC.clip_inputs(mclip, *plain_stats)
results.append(check("motion inputs are (frames, 79)", xm.shape, (8, F.FEATURE_SIZE + MO.NUM_MOTION + 1)))
results.append(check("without --motion inputs are still (frames, 65)", xp.shape, (8, MC.INPUT_SIZE)))
results.append(check("the 64 shape columns are the same with or without motion",
                     bool(np.allclose(xm[:, :F.FEATURE_SIZE], xp[:, :F.FEATURE_SIZE])), True))
results.append(check("hand flag is still the last column", xm[:, -1].tolist(), mask8.astype(float).tolist()))
results.append(check("no-hand frames are all zeros", bool(xm[~mask8].any()), False))
mmodel = MC.CTCModel(input_size=xm.shape[1], hidden=16).eval()
with torch.no_grad():
    out = mmodel(torch.from_numpy(xm).unsqueeze(0), torch.tensor([8]))
results.append(check("motion model outputs (1, frames, 27)", tuple(out.shape), (1, 8, D.NUM_CLASSES)))
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "m.pt")
    MC.save_model(path, mmodel, motion_stats, {"hidden": 16, "layers": 2, "dropout": 0.3, "fps": 15.0,
                                               "input_size": xm.shape[1], "motion": True})
    mloaded, mstats, _ = MC.load_model(path)
with torch.no_grad():
    same = torch.equal(mmodel(torch.from_numpy(xm).unsqueeze(0), torch.tensor([8])),
                       mloaded(torch.from_numpy(MC.clip_inputs(mclip, *mstats)).unsqueeze(0), torch.tensor([8])))
results.append(check("saved motion model loads with the right input size and same outputs", same, True))
# --- Transformer encoder (Step 5): same inputs, outputs and decoder as the LSTM ---
torch.manual_seed(0)
tf = MC.make_model("transformer", MC.INPUT_SIZE, hidden=32, layers=2, dropout=0.1).eval()
xb = torch.randn(2, 12, MC.INPUT_SIZE)
with torch.no_grad():
    out = tf(xb, torch.tensor([12, 7]))
results.append(check("transformer outputs (batch, frames, 27)", tuple(out.shape), (2, 12, D.NUM_CLASSES)))
results.append(check("transformer outputs are log-probabilities",
                     bool(torch.allclose(out.exp().sum(-1), torch.ones(2, 12), atol=1e-5)), True))
with torch.no_grad():
    alone = tf(xb[1:, :7], torch.tensor([7]))
results.append(check("transformer ignores padding (clip alone = clip padded in a batch)",
                     bool(torch.allclose(alone[0], out[1, :7], atol=1e-5)), True))
results.append(check("make_model('lstm') is the LSTM", type(MC.make_model("lstm", MC.INPUT_SIZE)).__name__, "CTCModel"))
try:
    MC.make_model("gru", MC.INPUT_SIZE)
    results.append(check("unknown model name refused", "no error", "ValueError"))
except ValueError:
    results.append(check("unknown model name refused", "ValueError", "ValueError"))
n_lstm = MC.count_params(MC.make_model("lstm", MC.INPUT_SIZE, hidden=128, layers=2))
n_tf = MC.count_params(MC.make_model("transformer", MC.INPUT_SIZE, hidden=128, layers=3))
results.append(check(f"default sizes are comparable (LSTM {n_lstm:,} vs Transformer {n_tf:,} parameters)",
                     0.5 < n_tf / n_lstm < 1.5, True))
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "tf.pt")
    MC.save_model(path, tf, stats, {"model": "transformer", "hidden": 32, "layers": 2, "dropout": 0.1, "fps": 15.0})
    tf_loaded, _, _ = MC.load_model(path)
with torch.no_grad():
    same = torch.equal(tf(xb, torch.tensor([12, 7])), tf_loaded(xb, torch.tensor([12, 7])))
results.append(check("saved transformer loads with identical outputs", same, True))

try:
    MC.clip_inputs(D.Clip(path="old", word="HI", signer="anna", labels=[8, 9], features=feats8, mask=mask8,
                          timestamps=np.arange(8) / 15, fps=15.0), *motion_stats)
    results.append(check("motion without raw landmarks gives a clear error", "no error", "ValueError"))
except ValueError:
    results.append(check("motion without raw landmarks gives a clear error", "ValueError", "ValueError"))

print("\nPASS: all model_ctc tests passed" if all(results) else "\nFAIL: some model_ctc tests failed")
