"""
Tests for motion.py using hand-made moving hands (no real data, model or webcam needed).
Run: python test_motion.py
"""
import numpy as np
import motion as MO


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return ok


def close(name, got, want, tol=1e-6):
    ok = bool(np.allclose(got, want, atol=tol))
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {np.round(got, 4).tolist()!r}, want {np.round(want, 4).tolist()!r}")
    return ok


# A hand template inside a 1x1 box. Points 1 and 2 sit on opposite corners, so the hand's size (its
# bounding box) stays exactly 1 x size even when a fingertip moves inside the box.
TEMPLATE = np.random.default_rng(0).uniform(0.2, 0.8, (21, 3)) * [1, 1, 0]
TEMPLATE[1] = [0, 0, 0]
TEMPLATE[2] = [1, 1, 0]
TEMPLATE[0] = [0.5, 0.9, 0]  # wrist


def clip_of(n, fps=15.0, size=0.1, start=(0.3, 0.3), step=(0.0, 0.0), tip_step=None, missing=()):
    """n frames of the template hand moving `step` (screen units) per 1/15 s, optionally one fingertip too."""
    ts = np.arange(n) / fps
    lm = np.zeros((n, 21, 3))
    for t in range(n):
        hand = TEMPLATE.copy()
        if tip_step is not None:
            tip, (dx, dy) = tip_step
            hand[tip, :2] += np.array([dx, dy]) * ts[t] * 15 / size  # moves inside the box
        lm[t] = hand * size
        lm[t, :, :2] += np.array(start) + np.array(step) * ts[t] * 15
    mask = np.ones(n, bool)
    mask[list(missing)] = False
    lm[~mask] = np.nan
    return lm, mask, ts


results = []

# --- shape ---
lm, mask, ts = clip_of(10)
m = MO.motion_features(lm, mask, ts)
results.append(check("14 motion numbers per frame", (m.shape, MO.NUM_MOTION), ((10, 14), 14)))
results.append(check("a hand that doesn't move gives all zeros", bool(np.allclose(m, 0)), True))

# --- the whole hand moving right at a steady speed ---
# 0.01 screen units per frame at 15 fps, hand size 0.1 -> 0.1 hand-widths per frame = 1.5 hand-widths per second
lm, mask, ts = clip_of(10, step=(0.01, 0.0))
m = MO.motion_features(lm, mask, ts)
results.append(close("wrist position grows 0.1 hand-widths per frame", m[:, 0], 0.1 * np.arange(10)))
results.append(close("wrist up/down position stays 0", m[:, 1], 0))
results.append(close("wrist speed: 0 on the first frame, then 1.5 per second", m[:, 2], [0] + [1.5] * 9))
results.append(close("no up/down speed", m[:, 3], 0))
results.append(close("fingers don't move relative to the wrist", m[:, 4:], 0))

# --- the same movement looks the same however big the hand is on screen, and wherever it is ---
lm2, mask2, ts2 = clip_of(10, size=0.2, start=(0.1, 0.5), step=(0.02, 0.0))  # twice as close to the camera
results.append(close("same features at twice the size and somewhere else on screen",
                     MO.motion_features(lm2, mask2, ts2), m))

# --- the same movement at 30 fps gives the same speeds (speeds are per second, not per frame) ---
lm3, mask3, ts3 = clip_of(19, fps=30.0, step=(0.01, 0.0))  # same speed in screen units per second
m3 = MO.motion_features(lm3, mask3, ts3)
results.append(close("same wrist speed at 30 fps", m3[1:, 2], 1.5))

# --- a frame without a hand ---
lm, mask, ts = clip_of(10, step=(0.01, 0.0), missing=[5])
m = MO.motion_features(lm, mask, ts)
results.append(close("no-hand frame is all zeros", m[5], 0))
results.append(close("no speed right after a gap (the hand didn't jump)", m[6, 2:4], 0))
results.append(close("position after the gap is still measured from the start", m[6, 0], 0.6))

# --- only the index fingertip moves (like J's pinky twist, but on finger 8) ---
lm, mask, ts = clip_of(6, tip_step=(8, (0.0, 0.002)))
m = MO.motion_features(lm, mask, ts)
col = 4 + 2 * MO.TIPS.index(8)
results.append(close("wrist stays still", m[:, :4], 0))
results.append(close("index tip moves down at 0.02 hand-widths per frame = 0.3 per second",
                     m[1:, col + 1], 0.3))
others = [c for c in range(4, 14) if c not in (col, col + 1)]
results.append(close("the other fingertips don't move", m[:, others], 0))

# --- edge cases ---
lm, mask, ts = clip_of(4, missing=[0, 1, 2, 3])
results.append(close("clip with no hand at all gives zeros", MO.motion_features(lm, mask, ts), 0))
lm, mask, ts = clip_of(1)
results.append(check("one-frame clip works", MO.motion_features(lm, mask, ts).shape, (1, 14)))

print("\nPASS: all motion tests passed" if all(results) else "\nFAIL: some motion tests failed")
