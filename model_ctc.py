"""
The temporal model: reads a whole clip and outputs, for every frame, how likely each letter (or "blank") is.
Trained with CTC loss (train_ctc.py), so it learns where letters start and end from (clip, word) pairs alone.

  feature_stats(clips)        -> mean, std of the 64 features over hand frames (training clips only)
  clip_inputs(clip, mean, std) -> (T, 65): standardized features + a "hand present" flag (0 where no hand)
                                 (T, 79) with motion stats: + 14 motion features (motion.py, Step 5)
  CTCModel                    -> 2-layer bidirectional LSTM -> 27 log-probabilities per frame
  TransformerCTCModel         -> small Transformer encoder, same inputs and outputs (Step 5 comparison)
  make_model(kind, ...)       -> 'lstm' or 'transformer'
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


class TransformerCTCModel(nn.Module):
    """
    Step 5 comparison model: a small Transformer encoder in place of the LSTM, with the same inputs, the same
    27 log-probabilities per frame and the same decoder. Every frame attends to every other frame of the clip
    (like the bidirectional LSTM, it sees the whole word). Frame order comes from sine/cosine position codes.
    """

    def __init__(self, input_size=INPUT_SIZE, hidden=128, layers=3, dropout=0.1, heads=4, max_frames=2000):
        super().__init__()
        self.inp = nn.Linear(input_size, hidden)
        pos = torch.arange(max_frames).unsqueeze(1)
        freq = torch.exp(torch.arange(0, hidden, 2) * (-np.log(10000.0) / hidden))
        pe = torch.zeros(max_frames, hidden)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        self.register_buffer("pe", pe, persistent=False)  # fixed, so not stored in saved models
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden, dropout=dropout,
                                           batch_first=True, norm_first=True)  # norm first: steadier training
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, D.NUM_CLASSES)

    def forward(self, x, lengths):
        """x (batch, frames, inputs) padded with zeros, lengths (batch,) -> log-probs (batch, frames, 27)."""
        frames = x.shape[1]
        padding = torch.arange(frames, device=x.device)[None, :] >= lengths.to(x.device)[:, None]  # True = padding
        h = self.drop(self.inp(x) + self.pe[:frames])
        h = self.encoder(h, src_key_padding_mask=padding)  # real frames never attend to padding
        return self.out(self.norm(h)).log_softmax(dim=-1)


def make_model(kind, input_size, hidden=128, layers=2, dropout=0.3):
    """'lstm' -> CTCModel, 'transformer' -> TransformerCTCModel."""
    if kind == "lstm":
        return CTCModel(input_size=input_size, hidden=hidden, layers=layers, dropout=dropout)
    if kind == "transformer":
        return TransformerCTCModel(input_size=input_size, hidden=hidden, layers=layers, dropout=dropout)
    raise ValueError(f"unknown model {kind!r}: use 'lstm' or 'transformer'")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


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
    model = make_model(cfg.get("model", "lstm"), cfg.get("input_size", INPUT_SIZE), hidden=cfg["hidden"],
                       layers=cfg["layers"], dropout=cfg["dropout"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    info = {k: v for k, v in saved.items() if k not in ("state_dict", "mean", "std")}
    return model, (saved["mean"].numpy(), saved["std"].numpy()), info
