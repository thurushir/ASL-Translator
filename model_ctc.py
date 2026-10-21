"""
The temporal model: reads a whole clip and outputs, for every frame, how likely each letter (or "blank") is.
Trained with CTC loss (train_ctc.py), so it learns where letters start and end from (clip, word) pairs alone.

  feature_stats(clips)        -> mean, std of the 64 features over hand frames (training clips only)
  clip_inputs(clip, mean, std) -> (T, 65): standardized features + a "hand present" flag (0 where no hand)
  CTCModel                    -> 2-layer bidirectional LSTM -> 27 log-probabilities per frame
  greedy_decode / decode_batch -> best output per frame -> merge repeats -> drop blanks -> word

Run the tests: python test_model_ctc.py
"""
import numpy as np
import torch
from torch import nn
import dataset as D
import features as F

INPUT_SIZE = F.FEATURE_SIZE + 1  # 64 features + hand-present flag


def feature_stats(clips):
    """Mean and std of each feature over every hand frame in clips. Constant features get std 1."""
    hand = np.concatenate([c.hand_only() for c in clips])
    std = hand.std(axis=0)
    return hand.mean(axis=0), np.where(std < 1e-6, 1.0, std)


def clip_inputs(clip, mean, std):
    """(T, 65) float32: standardized features on hand frames, zeros on no-hand frames, then the mask."""
    x = np.zeros((len(clip.mask), INPUT_SIZE), dtype=np.float32)
    x[clip.mask, :-1] = (clip.features[clip.mask] - mean) / std
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
