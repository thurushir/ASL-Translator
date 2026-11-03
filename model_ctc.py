"""
The temporal model: reads a whole clip and outputs, for every frame, how likely each letter (or "blank") is.
Trained with CTC loss (train_ctc.py), so it learns where letters start and end from (clip, word) pairs alone.

  feature_stats(clips)        -> mean, std of the 64 features over hand frames (training clips only)
  clip_inputs(clip, mean, std) -> (T, 65): standardized features + a "hand present" flag (0 where no hand)
                                 (T, 79) with motion stats: + 14 motion features (motion.py, Step 5)
  CTCModel                    -> 2-layer bidirectional LSTM -> 27 log-probabilities per frame
  greedy_decode / decode_batch -> best output per frame -> merge repeats -> drop blanks -> word
  save_model / load_model     -> weights + standardization numbers + settings in one .pt file (Step 5)

Run the tests: python test_model_ctc.py
"""
import numpy as np
import torch
from torch import nn
import dataset as D
import features as F

INPUT_SIZE = F.FEATURE_SIZE + 1  # 64 features + hand-present flag (65); with --motion: + 14 = 79


def frame_numbers(clip, motion=False):
    """(T, 64) shape features, or (T, 78) with the 14 motion features (motion.py) after them."""
    if not motion:
        return clip.features
    import motion as MO
    return np.concatenate([clip.features, MO.clip_motion(clip)], axis=1)


def feature_stats(clips, motion=False):
    """Mean and std of each input number over every hand frame in clips. Constant ones get std 1."""
    hand = np.concatenate([frame_numbers(c, motion)[c.mask] for c in clips])
    std = hand.std(axis=0)
    return hand.mean(axis=0), np.where(std < 1e-6, 1.0, std)


def clip_inputs(clip, mean, std):
    """
    (T, 65) float32: standardized features on hand frames, zeros on no-hand frames, then the mask.
    With motion stats (78 numbers, from feature_stats(..., motion=True)) it is (T, 79) instead.
    """
    nums = frame_numbers(clip, motion=len(mean) > F.FEATURE_SIZE)
    x = np.zeros((len(clip.mask), len(mean) + 1), dtype=np.float32)
    x[clip.mask, :-1] = (nums[clip.mask] - mean) / std
    x[:, -1] = clip.mask
    return x


class CTCModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(2 * hidden, D.NUM_CLASSES)

    def forward(self, x, lengths):
        """x (batch, frames, 65) padded with zeros, lengths (batch,) -> log-probs (batch, frames, 27)."""
        # Packing makes the LSTM skip padding, so a clip's outputs don't depend on what it was batched with.
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        return self.out(h).log_softmax(dim=-1)


def greedy_decode(log_probs):
    """(frames, 27) -> word. Most likely output per frame, merge runs of the same output, drop blanks."""
    best = log_probs.argmax(dim=-1).tolist()
    merged = [k for i, k in enumerate(best) if i == 0 or k != best[i - 1]]
    return D.decode(merged)  # decode() skips blanks


def decode_batch(log_probs, lengths):
    """(batch, frames, 27) padded -> list of words, each decoded from its own frames only."""
    return [greedy_decode(lp[:n]) for lp, n in zip(log_probs, lengths.tolist())]


# ---------- saving and loading (Step 5) ----------

def save_model(path, model, stats, config, **extra):
    """
    Saves everything needed to use the model later (evaluation, the live app): the weights, the
    standardization numbers it was trained with (mean, std), and its settings (hidden, layers, dropout, fps).
    extra: anything else worth keeping, e.g. test_signer="anna", seed=0.
    """
    mean, std = stats
    torch.save({"state_dict": model.state_dict(),
                "mean": torch.as_tensor(np.asarray(mean), dtype=torch.float64),
                "std": torch.as_tensor(np.asarray(std), dtype=torch.float64),
                "config": dict(config), **extra}, path)


def load_model(path):
    """-> (model in eval mode, (mean, std) as numpy arrays, the saved dict without the weights)."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    cfg = saved["config"]
    model = CTCModel(input_size=cfg.get("input_size", INPUT_SIZE), hidden=cfg["hidden"], layers=cfg["layers"],
                     dropout=cfg["dropout"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    info = {k: v for k, v in saved.items() if k not in ("state_dict", "mean", "std")}
    return model, (saved["mean"].numpy(), saved["std"].numpy()), info
