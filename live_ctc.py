"""
The live side of the temporal model (Step 6): turns a stream of webcam frames into words, and predicts each
word exactly the way recorded clips are predicted offline. No webcam code here, so it can be tested by replaying
recorded clips (test_live_ctc.py); predict_live_ctc.py connects it to the camera.

  WordBuilder   add(points or None, time) per frame -> a finished word clip once the hand has been gone
                END_AFTER seconds (or finish() on a key press). Same format as dataset.load_clip().
  Recognizer    clip -> resample to the model's fps -> standardized inputs -> per-frame log-probs -> word.
                Recognizer.load("models/<name>.pt") uses the standardization numbers saved with the model.

The model is bidirectional (every frame looks at the frames after it), so it can only answer once the word
has ended: the delay a user feels is END_AFTER, not the model (about 2 ms per word).
"""
import numpy as np
import torch
import dataset as D
import features as F
import model_ctc as MC

END_AFTER = 1.0       # seconds without a hand that end a word (same rule as transcript.py's word break)
MIN_HAND_FRAMES = 5   # fewer hand frames than this is a hand passing through, not a word


class WordBuilder:
    def __init__(self, end_after=END_AFTER, min_hand_frames=MIN_HAND_FRAMES):
        self.end_after = end_after
        self.min_hand_frames = min_hand_frames
        self._reset()

    def _reset(self):
        self._points, self._times, self._last_hand = [], [], None

    @property
    def in_word(self):
        return bool(self._points)

    def add(self, points, now):
        """
        Call once per webcam frame. points: (21, 3) raw landmarks (features.raw_landmarks()) or None if no hand.
        now: the frame's real time in seconds. Returns a finished word's Clip, or None.
        """
        if points is not None:
            self._points.append(np.asarray(points, dtype=np.float64))
            self._times.append(now)
            self._last_hand = now
            return None
        if not self._points:
            return None  # no word started: frames before the hand appears are not part of a word
        if now - self._last_hand >= self.end_after:
            return self.finish()
        self._points.append(np.full((F.NUM_LANDMARKS, 3), np.nan))
        self._times.append(now)
        return None

    def finish(self):
        """Ends the current word now (end of the gap, or a key press). Returns its Clip, or None."""
        if not self._points:
            return None
        lm, ts = np.array(self._points), np.array(self._times)
        hand = ~np.isnan(lm[:, 0, 0])
        self._reset()
        if hand.sum() < self.min_hand_frames:
            return None
        last = np.flatnonzero(hand)[-1]
        lm, ts = lm[:last + 1], ts[:last + 1] - ts[0]  # trim no-hand frames after the hand left
        return _make_clip(lm, ts)


def _make_clip(landmarks, timestamps):
    """A Clip built the same way dataset.load_clip() builds one from a recorded .npz file."""
    feats, mask = F.sequence_features(landmarks)
    fps = (len(timestamps) - 1) / timestamps[-1] if len(timestamps) > 1 and timestamps[-1] > 0 else 0.0
    return D.Clip(path="live", word="", signer="live", labels=[], features=feats, mask=mask,
                  timestamps=timestamps, fps=float(fps), landmarks=landmarks)


class Recognizer:
    def __init__(self, model, stats, fps=15.0):
        self.model, self.stats, self.fps = model.eval(), stats, fps

    @classmethod
    def load(cls, path):
        model, stats, info = MC.load_model(path)
        return cls(model, stats, fps=info["config"].get("fps", 15.0))

    @torch.no_grad()
    def log_probs(self, clip):
        """(frames, 27) numpy log-probabilities for the clip, after resampling to the model's fps."""
        clip = D.resample(clip, self.fps)
        x = torch.from_numpy(MC.clip_inputs(clip, *self.stats)).unsqueeze(0)
        return self.model(x, torch.tensor([x.shape[1]]))[0].numpy()

    def greedy(self, clip):
        return MC.greedy_decode(torch.from_numpy(self.log_probs(clip)))


def replay(path, builder, start=0.0):
    """Feeds a recorded clip's frames through builder as if from the webcam. Returns the word clips it made."""
    data = np.load(path)
    words = []
    for points, t in zip(data["landmarks"], data["timestamps"]):
        clip = builder.add(None if np.isnan(points).any() else points, start + float(t))
        if clip is not None:
            words.append(clip)
    last = builder.finish()
    return words + ([last] if last is not None else [])


def replay_check(run):
    """
    Replays every recorded clip through WordBuilder and the run's saved per-fold model for that signer
    (train_ctc.py --save --name <run>), and compares with the offline prediction stored in results/<run>.json.
    """
    import json
    res = json.load(open(f"results/{run}.json"))
    recs = {s: Recognizer.load(f"models/{run}_{s}.pt") for s in res["signers"]}
    same, diff = 0, []
    for p in res["predictions"]:
        words = replay(p["path"], WordBuilder())
        live = " ".join(recs[p["signer"]].greedy(c) for c in words)
        if live == p["pred"]:
            same += 1
        else:
            diff.append((p["path"], p["pred"], live, len(words)))
    print(f"{same}/{len(res['predictions'])} replayed clips give exactly the offline prediction")
    for path, offline, live, n in diff:
        print(f"  {path}: offline {offline!r}, live {live!r} ({n} word(s) cut)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Check the live path against offline predictions.")
    p.add_argument("--replay", required=True, help="a run saved with train_ctc.py --save --name <run>")
    replay_check(p.parse_args().replay)
