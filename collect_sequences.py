"""
Records continuous fingerspelling clips (one word per clip) as per-frame MediaPipe landmarks.

Run one word:      python collect_sequences.py --signer thurushi --word HELLO
Run the word list: python collect_sequences.py --signer thurushi --list word_list.txt
Options: --takes N (clips per word in list mode, default 3), --video (also save the .mp4)

Controls (video window):
  r = start recording (after a 3s countdown) / stop and save
  n = next word,  b = previous word,  d = delete your last take of this word,  q = quit
List mode resumes where you left off: it starts at the first word you have fewer than N takes of,
and moves to the next word automatically once a word has N takes.

Each clip is saved to data_seq/<WORD>/<signer>_<n>.npz containing:
  landmarks  (T, 21, 3)  raw x, y, z per frame; NaN rows where no hand was detected
  timestamps (T,)        seconds since recording started
  word, signer           labels, e.g. "HELLO", "thurushi"
  mirrored               True: frames are mirrored like the live view
With --video the raw (mirrored) webcam clip is also saved as <signer>_<n>.mp4.
"""
import argparse
import os
import re
import time
import numpy as np

DIR = "./data_seq"
COUNTDOWN = 3       # seconds before recording starts
MIN_FRAMES = 5      # clips shorter than this are discarded


# ---------- file helpers (no webcam needed) ----------

def load_word_list(path):
    """Words from a text file: one per line, blank lines and # comments ignored."""
    with open(path) as f:
        return [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]


def clip_numbers(word, signer, root=DIR):
    """Sorted take numbers this signer already has for this word (from <signer>_<n>.npz files)."""
    folder = os.path.join(root, word)
    if not os.path.isdir(folder):
        return []
    pattern = re.compile(rf"^{re.escape(signer)}_(\d+)\.npz$")
    return sorted(int(m.group(1)) for f in os.listdir(folder) if (m := pattern.match(f)))


def first_unfinished(words, signer, takes, root=DIR):
    """Index of the first word with fewer than `takes` clips (0 if all are done)."""
    for i, w in enumerate(words):
        if len(clip_numbers(w, signer, root)) < takes:
            return i
    return 0


def clip_path(word, signer, n, ext, root=DIR):
    return os.path.join(root, word, f"{signer}_{n}.{ext}")


# ---------- recording session ----------

def main():
    import cv2
    import hand_utils as h
    import features as F

    parser = argparse.ArgumentParser()
    parser.add_argument("--signer", required=True, help="your name, e.g. thurushi (goes in every file name)")
    parser.add_argument("--word", help="record a single word")
    parser.add_argument("--list", help="record every word in a word-list file")
    parser.add_argument("--takes", type=int, default=3, help="clips per word in list mode (default 3)")
    parser.add_argument("--video", action="store_true", help="also save the raw .mp4 clip")
    args = parser.parse_args()

    signer = re.sub(r"[^a-z0-9]", "", args.signer.lower())
    if not signer:
        raise SystemExit("--signer must contain letters or numbers, e.g. --signer thurushi")
    if args.list:
        words = load_word_list(args.list)
    else:
        words = [(args.word or input("Word to fingerspell: ")).strip().upper()]

    s = {  # session state
        "i": first_unfinished(words, signer, args.takes) if args.list else 0,
        "phase": "idle",   # idle -> countdown -> recording -> idle
        "countdown_end": 0.0, "start": 0.0, "n": None, "writer": None, "msg": "",
    }
    frames, times = [], []

    def word():
        return words[s["i"]]

    def takes_done():
        return len(clip_numbers(word(), signer))

    def all_done():
        return args.list and all(len(clip_numbers(w, signer)) >= args.takes for w in words)

    def remove_video():
        mp4 = clip_path(word(), signer, s["n"], "mp4")
        if os.path.exists(mp4):
            os.remove(mp4)

    def stop_and_save():
        if s["writer"]:
            s["writer"].release()
            s["writer"] = None
        if len(frames) < MIN_FRAMES:
            remove_video()
            s["msg"] = f"Too short ({len(frames)} frames), discarded"
            return
        landmarks = np.array(frames, dtype=np.float32)
        path = clip_path(word(), signer, s["n"], "npz")
        np.savez_compressed(path, landmarks=landmarks, timestamps=np.array(times),
                            word=word(), signer=signer, mirrored=True)
        hand_pct = 100 * (~np.isnan(landmarks[:, 0, 0])).mean()
        print(f"Saved {path}  ({len(frames)} frames, {times[-1]:.1f}s, hand in {hand_pct:.0f}% of frames)")
        s["msg"] = f"Saved take {takes_done()} ({hand_pct:.0f}% hand)"
        if args.list and takes_done() >= args.takes and s["i"] < len(words) - 1:
            s["i"] += 1  # word finished: move on automatically
            s["msg"] += f" - next word: {word()}"

    def delete_last():
        nums = clip_numbers(word(), signer)
        if not nums:
            s["msg"] = "Nothing to delete for this word"
            return
        for ext in ("npz", "mp4"):
            p = clip_path(word(), signer, nums[-1], ext)
            if os.path.exists(p):
                os.remove(p)
        print(f"Deleted {clip_path(word(), signer, nums[-1], 'npz')}")
        s["msg"] = f"Deleted last take of {word()}"

    def handle_frame(frame, results):
        now = time.time()
        if s["phase"] == "countdown" and now >= s["countdown_end"]:
            s["phase"], s["start"] = "recording", now
            frames.clear()
            times.clear()
            if args.video:
                height, width = frame.shape[:2]
                s["writer"] = cv2.VideoWriter(clip_path(word(), signer, s["n"], "mp4"),
                                              cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
        if s["phase"] == "recording":
            if s["writer"]:
                s["writer"].write(frame)  # save the clean frame before drawing on it
            points = F.raw_landmarks(results)
            frames.append(points if points is not None else np.full((F.NUM_LANDMARKS, 3), np.nan))
            times.append(now - s["start"])

        h.draw_hand_landmarks(frame, results)

        # overlay: progress, status, last message, key help
        font = cv2.FONT_HERSHEY_SIMPLEX
        progress = f"Word {s['i'] + 1}/{len(words)}: {word()}   takes done: {takes_done()}" + (f"/{args.takes}" if args.list else "")
        if s["phase"] == "idle":
            status, color = ("All words done! Press q to quit" if all_done() else "Press r to record"), (0, 255, 0)
        elif s["phase"] == "countdown":
            status, color = f"Starting in {int(s['countdown_end'] - now) + 1}...", (0, 165, 255)
        else:
            status, color = f"REC {now - s['start']:.1f}s  (r to stop)", (0, 0, 255)
        cv2.putText(frame, f"Signer: {signer}", (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, progress, (10, 60), font, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, status, (10, 95), font, 0.9, color, 2)
        cv2.putText(frame, s["msg"], (10, 125), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "r record/stop  n next  b back  d delete last  q quit",
                    (10, frame.shape[0] - 15), font, 0.55, (255, 255, 255), 1)

    def handle_key(key):
        if key == ord("r"):
            if s["phase"] == "idle":
                os.makedirs(os.path.join(DIR, word()), exist_ok=True)
                nums = clip_numbers(word(), signer)
                s["n"] = (nums[-1] + 1) if nums else 0  # never overwrite an existing take
                s["phase"], s["countdown_end"], s["msg"] = "countdown", time.time() + COUNTDOWN, ""
            elif s["phase"] == "countdown":
                s["phase"], s["msg"] = "idle", "Cancelled"
            else:
                s["phase"] = "idle"
                stop_and_save()
        elif s["phase"] == "idle":  # word navigation only while not recording
            if key == ord("n") and s["i"] < len(words) - 1:
                s["i"], s["msg"] = s["i"] + 1, ""
            elif key == ord("b") and s["i"] > 0:
                s["i"], s["msg"] = s["i"] - 1, ""
            elif key == ord("d"):
                delete_last()
        return False

    hands = h.init_hands(static_mode=False)  # video mode: tracks the hand between frames
    print(f"Signer '{signer}', {len(words)} word(s). Use the video window: r record/stop, n next, b back, d delete last, q quit.")
    h.start_video(hands, handle_frame, handle_key)

    if s["phase"] == "recording":  # quit mid-recording: don't keep a half-finished clip
        if s["writer"]:
            s["writer"].release()
        remove_video()
        print("Quit while recording; that clip was discarded.")


if __name__ == "__main__":
    main()
