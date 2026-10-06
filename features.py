"""
Shared hand-feature extraction used by training, data collection and live prediction.
Keeping it in one place guarantees every script computes features the same way.

Feature vector (64 values per frame):
  - 21 landmarks x (x, y, z), normalized so the hand fits in a 1x1 box
    (x, y relative to the top-left of the hand, z relative to the wrist)
  - thumb openness: distance from thumb tip (4) to thumb joint (2), divided by hand scale
    (helps tell apart closed-fist letters like M, N, S)
"""
import numpy as np

NUM_LANDMARKS = 21
FEATURE_SIZE = NUM_LANDMARKS * 3 + 1  # 64


def raw_landmarks(results):
    """MediaPipe results -> (21, 3) array of raw x, y, z for the first hand, or None if no hand."""
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float64)


def landmarks_to_features(points):
    """(21, 3) raw landmarks -> (64,) feature vector, or None if the hand has zero size."""
    points = np.asarray(points, dtype=np.float64)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    min_x, min_y, min_z = x.min(), y.min(), z[0]  # z is relative to the wrist (landmark 0)
    scale = max(x.max() - min_x, y.max() - min_y)  # hand width or height, whichever is larger
    if scale == 0:
        return None

    norm = (points - [min_x, min_y, min_z]) / scale  # (21, 3) -> flattened as x0,y0,z0,x1,...
    thumb_dist = np.sqrt(((points[4] - points[2]) ** 2).sum()) / scale
    return np.append(norm.reshape(-1), thumb_dist)


def frame_features(results):
    """MediaPipe results -> (64,) feature vector for one frame, or None if no usable hand."""
    points = raw_landmarks(results)
    return None if points is None else landmarks_to_features(points)


def sequence_features(landmark_seq):
    """
    (T, 21, 3) raw landmarks for T frames (NaN rows = no hand that frame)
    -> features (T, 64) with zeros for missing frames, and mask (T,) True where a hand was found.
    """
    landmark_seq = np.asarray(landmark_seq, dtype=np.float64)
    feats = np.zeros((len(landmark_seq), FEATURE_SIZE))
    mask = np.zeros(len(landmark_seq), dtype=bool)
    for t, points in enumerate(landmark_seq):
        if np.isnan(points).any():
            continue
        f = landmarks_to_features(points)
        if f is not None:
            feats[t], mask[t] = f, True
    return feats, mask
