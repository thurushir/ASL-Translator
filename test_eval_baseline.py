"""
Tests for eval_baseline.py using hand-made per-frame predictions (no model or real data needed).
Run: python test_eval_baseline.py
"""
import numpy as np
import eval_baseline as E


def check(name, got, want):
    print(f"{'ok  ' if got == want else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return got == want


def hold(letter, n, conf=90):
    return [(letter, conf)] * n


def times(frames, fps):
    return [i / fps for i in range(len(frames))]


results = []

# --- turning the model's output into per-frame (letter, confidence %) ---
class FakeModel:
    """Predicts 'A' (80%) when the first feature is < 0.5, else 'B' (60%)."""
    classes_ = np.array(["A", "B"])

    def predict_proba(self, X):
        return np.array([[0.8, 0.2] if row[0] < 0.5 else [0.4, 0.6] for row in X])


feats = np.zeros((4, 64))
feats[2, 0] = 0.9
mask = np.array([True, False, True, True])
results.append(check("frame predictions: letter + confidence, None where no hand",
                     E.frame_predictions(FakeModel(), feats, mask), [("A", 80.0), (None, 0.0), ("B", 60.0), ("A", 80.0)]))

# --- A. as-is: the live app's rules (hold 8 frames at >= 50%) ---
slow = hold("C", 10) + hold("A", 10) + hold("T", 10)
results.append(check("A: letters held 10 frames are spelled", E.decode_as_is(slow, times(slow, 15)), "CAT"))
fast = hold("C", 5) + hold("A", 5) + hold("T", 5)
results.append(check("A: letters held only 5 frames are missed", E.decode_as_is(fast, times(fast, 15)), ""))
gap = hold("H", 10) + hold("I", 10) + hold(None, 20) + hold("O", 10)
results.append(check("A: a no-hand gap inside a clip doesn't leave a space", E.decode_as_is(gap, times(gap, 15)), "HIO"))

# --- B. tuned timing: hold and release set in seconds, so it works the same at any fps ---
at15 = hold("C", 4) + hold("A", 4) + hold("T", 4)    # each letter ~0.27s at 15 fps
at30 = hold("C", 8) + hold("A", 8) + hold("T", 8)    # each letter ~0.27s at 30 fps
results.append(check("B: 0.2s hold spells CAT at 15 fps",
                     E.decode_timed(at15, times(at15, 15), fps=15, hold_secs=0.2, min_conf=50, release_secs=0.2), "CAT"))
results.append(check("B: 0.2s hold spells CAT at 30 fps",
                     E.decode_timed(at30, times(at30, 30), fps=30, hold_secs=0.2, min_conf=50, release_secs=0.2), "CAT"))
results.append(check("B: 0.5s hold is too long for these letters",
                     E.decode_timed(at15, times(at15, 15), fps=15, hold_secs=0.5, min_conf=50, release_secs=0.2), ""))
low = hold("C", 4, conf=40) + hold("A", 4) + hold("T", 4)
results.append(check("B: letters below min_conf are skipped",
                     E.decode_timed(low, times(low, 15), fps=15, hold_secs=0.2, min_conf=50, release_secs=0.2), "AT"))

# --- C. recognition only: merge repeats, no hold rule, no confidence cutoff ---
results.append(check("C: S S S A A T T -> SAT", E.decode_collapse(hold("S", 3) + hold("A", 2) + hold("T", 2)), "SAT"))
results.append(check("C: one-frame flickers are kept", E.decode_collapse(hold("C", 3) + hold("X", 1) + hold("A", 3)), "CXA"))
results.append(check("C: a no-hand frame splits a double letter",
                     E.decode_collapse(hold("L", 3) + hold(None, 1) + hold("L", 3)), "LL"))
results.append(check("C: low confidence still counts", E.decode_collapse(hold("A", 2, conf=10)), "A"))

# --- picking B's settings using only the training signers ---
# totals[setting][signer] = (edits, letters)
totals = {
    "short": {"anna": (2, 10), "saaya": (2, 10), "thurushi": (9, 10)},
    "long":  {"anna": (5, 10), "saaya": (5, 10), "thurushi": (0, 10)},
}
results.append(check("best setting for anna+saaya ignores thurushi's results",
                     E.best_setting(totals, ["anna", "saaya"]), "short"))
results.append(check("best setting changes with the training signers",
                     E.best_setting(totals, ["saaya", "thurushi"]), "long"))

# --- J/Z split ---
results.append(check("words with J or Z are flagged", [E.has_motion_letter(w) for w in ["JAZZ", "ZOO", "CAT", "JOB"]],
                     [True, True, False, True]))

print("\nPASS: all eval_baseline tests passed" if all(results) else "\nFAIL: some eval_baseline tests failed")
