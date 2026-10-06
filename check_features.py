"""
Checks that features.py produces exactly the same numbers as the original feature code.
Run: python check_features.py   (uses images from ./data if present, plus synthetic hands)
"""
import os
import random
import numpy as np
import features as F


def original_features(h_landmark):
    """Copy of the original per-frame feature code from process_data.py / predict_live.py (reference only)."""
    x = [lm.x for lm in h_landmark.landmark]
    y = [lm.y for lm in h_landmark.landmark]
    min_x, min_y, min_z = min(x), min(y), h_landmark.landmark[0].z
    max_x, max_y = max(x), max(y)
    scale = max(max_x - min_x, max_y - min_y)
    if scale == 0:
        return None
    norm_hand = []
    for lm in h_landmark.landmark:
        norm_hand.extend([(lm.x - min_x) / scale, (lm.y - min_y) / scale, (lm.z - min_z) / scale])
    tip, joint = h_landmark.landmark[4], h_landmark.landmark[2]
    norm_hand.append(((tip.x - joint.x)**2 + (tip.y - joint.y)**2 + (tip.z - joint.z)**2) ** 0.5 / scale)
    return np.array(norm_hand)


class _Point:  # minimal stand-ins for MediaPipe result objects
    def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z

class _Hand:
    def __init__(self, pts): self.landmark = [_Point(*p) for p in pts]

class _Results:
    def __init__(self, hands): self.multi_hand_landmarks = hands


def compare(hands, label):
    bad = 0
    for hand in hands:
        old = original_features(hand)
        new = F.frame_features(_Results([hand]))
        if (old is None) != (new is None) or (old is not None and not np.allclose(old, new, atol=1e-9)):
            bad += 1
    print(f"{label}: {len(hands) - bad}/{len(hands)} match")
    return bad == 0


def synthetic_hands(n=500):
    rng = np.random.default_rng(0)
    return [_Hand(rng.random((21, 3)) * [1, 1, 0.2]) for _ in range(n)]


def real_hands(per_letter=5):
    if not os.path.isdir("./data"):
        print("No ./data folder, skipping real-image check.")
        return []
    import cv2
    import hand_utils as h
    hands = h.init_hands(static_mode=True)
    found = []
    for letter in sorted(os.listdir("./data")):
        folder = os.path.join("./data", letter)
        if not os.path.isdir(folder):
            continue
        imgs = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
        for name in random.sample(imgs, min(per_letter, len(imgs))):
            img = cv2.imread(os.path.join(folder, name))
            if img is None:
                continue
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                found.append(res.multi_hand_landmarks[0])
    return found


def check_sequence():
    """sequence_features() must equal frame-by-frame features, with missing frames masked out."""
    rng = np.random.default_rng(1)
    seq = rng.random((10, 21, 3))
    seq[[2, 7]] = np.nan  # frames with no hand
    feats, mask = F.sequence_features(seq)
    ok = mask.tolist() == [i not in (2, 7) for i in range(10)]
    ok &= all(np.allclose(feats[t], F.landmarks_to_features(seq[t])) for t in range(10) if mask[t])
    ok &= not feats[[2, 7]].any()
    print(f"sequence_features: {'ok' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    results = [compare(synthetic_hands(), "Synthetic hands"), check_sequence()]
    real = real_hands()
    if real:
        results.append(compare(real, "Real hands from ./data"))
    print("PASS: features match" if all(results) else "FAIL: features differ")
