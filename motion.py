"""
Motion features (Step 5): how the hand moves, which the 64 shape features in features.py throw away
(they measure every point relative to the hand's own bounding box). The 64 stay unchanged, so the static
model and baseline B stay valid; these 14 are added after them when train_ctc.py runs with --motion.

Per frame (all lengths in hand-widths, so camera distance doesn't matter; speeds per second, so frame rate doesn't):
  0-1    wrist position (x, y) relative to where the wrist was in the clip's first frame with a hand
  2-3    wrist velocity (x, y): the whole hand moving, e.g. tracing a Z
  4-13   velocity (x, y) of each fingertip (thumb, index, middle, ring, pinky) relative to the wrist:
         fingers moving within the hand, e.g. the pinky's hook in J
Frames without a hand are all zeros (the model's hand-present flag marks them). A velocity needs a hand in
both this frame and the one before; right after a gap it is 0, so a hand coming back doesn't look like a jump.
"Hand-width" = the clip's median hand size (larger side of the 2D bounding box of the 21 points).

Tests: python test_motion.py
"""
import numpy as np

WRIST = 0
TIPS = [4, 8, 12, 16, 20]  # MediaPipe fingertips: thumb, index, middle, ring, pinky
NUM_MOTION = 2 + 2 + 2 * len(TIPS)  # 14


def motion_features(landmarks, mask, timestamps):
    """(T, 21, 3) raw landmarks, (T,) hand mask, (T,) seconds -> (T, 14) motion features."""
    lm = np.asarray(landmarks, dtype=np.float64)[:, :, :2]  # x, y only: MediaPipe's z is noisy
    mask = np.asarray(mask, dtype=bool)
    ts = np.asarray(timestamps, dtype=np.float64)
    out = np.zeros((len(mask), NUM_MOTION))
    if not mask.any():
        return out
    hand = lm[mask]
    scale = np.median((hand.max(axis=1) - hand.min(axis=1)).max(axis=1))  # median hand size in the clip
    if scale <= 0:
        return out
    wrist = lm[:, WRIST] / scale                          # (T, 2), in hand-widths
    tips = (lm[:, TIPS] - lm[:, [WRIST]]) / scale         # (T, 5, 2), relative to the wrist
    first = int(np.flatnonzero(mask)[0])
    out[mask, 0:2] = wrist[mask] - wrist[first]
    for t in range(1, len(mask)):
        dt = ts[t] - ts[t - 1]
        if mask[t] and mask[t - 1] and dt > 0:
            out[t, 2:4] = (wrist[t] - wrist[t - 1]) / dt
            out[t, 4:] = ((tips[t] - tips[t - 1]) / dt).reshape(-1)
    return out


def clip_motion(clip):
    """Motion features for a dataset.Clip (after resampling, so speeds use the steady 15 fps timestamps)."""
    if clip.landmarks is None:
        raise ValueError(f"{clip.path} has no raw landmarks; load it with dataset.load_clip()")
    return motion_features(clip.landmarks, clip.mask, clip.timestamps)
