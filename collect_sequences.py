"""
Records continuous fingerspelling clips (one word per clip) as per-frame MediaPipe landmarks.

Run:      python collect_sequences.py [--word HELLO] [--video]
Controls: r = start recording (after a 3s countdown) / stop and save,  q = quit

Each clip is saved to data_seq/<WORD>/<n>.npz containing:
  landmarks  (T, 21, 3)  raw x, y, z per frame; NaN rows where no hand was detected
  timestamps (T,)        seconds since recording started
  word                   the label, e.g. "HELLO"
  mirrored               True: frames are mirrored like the live view
With --video the raw (mirrored) webcam clip is also saved as <n>.mp4.
"""
import argparse
import os
import time
import cv2
import numpy as np
import hand_utils as h
import features as F

DIR = "./data_seq"
COUNTDOWN = 3       # seconds before recording starts
MIN_FRAMES = 5      # clips shorter than this are discarded

parser = argparse.ArgumentParser()
parser.add_argument("--word", help="word to fingerspell (asked for if omitted)")
parser.add_argument("--video", action="store_true", help="also save the raw .mp4 clip")
args = parser.parse_args()

word = (args.word or input("Word to fingerspell: ")).strip().upper()
save_dir = os.path.join(DIR, word)
os.makedirs(save_dir, exist_ok=True)

phase = "idle"      # idle -> countdown -> recording -> idle
countdown_end = 0.0
start_time = 0.0
frames, times = [], []
writer = None
clip_idx = None
saved_count = 0


def next_clip_index():
    """Next unused clip number in save_dir, so existing clips are never overwritten."""
    nums = [int(f.split(".")[0]) for f in os.listdir(save_dir) if f.split(".")[0].isdigit()]
    return max(nums, default=-1) + 1


def save_clip():
    global writer, saved_count
    if writer:
        writer.release()
        writer = None
    path = os.path.join(save_dir, f"{clip_idx}.npz")
    if len(frames) < MIN_FRAMES:
        print(f"Clip too short ({len(frames)} frames), discarded.")
        return discard_video()
    landmarks = np.array(frames, dtype=np.float32)
    np.savez_compressed(path, landmarks=landmarks, timestamps=np.array(times),
                        word=word, mirrored=True)
    saved_count += 1
    hand_pct = 100 * (~np.isnan(landmarks[:, 0, 0])).mean()
    print(f"Saved {path}  ({len(frames)} frames, {times[-1]:.1f}s, hand in {hand_pct:.0f}% of frames)")


def discard_video():
    mp4 = os.path.join(save_dir, f"{clip_idx}.mp4")
    if os.path.exists(mp4):
        os.remove(mp4)


def handle_frame(frame, results):
    global phase, start_time, writer
    now = time.time()

    if phase == "countdown" and now >= countdown_end:
        phase, start_time = "recording", now
        frames.clear()
        times.clear()
        if args.video:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(os.path.join(save_dir, f"{clip_idx}.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))

    if phase == "recording":
        if writer:
            writer.write(frame)  # save the clean frame before drawing on it
        points = F.raw_landmarks(results)
        frames.append(points if points is not None else np.full((F.NUM_LANDMARKS, 3), np.nan))
        times.append(now - start_time)

    h.draw_hand_landmarks(frame, results)

    # overlay
    if phase == "idle":
        status, color = "Press r to record", (0, 255, 0)
    elif phase == "countdown":
        status, color = f"Starting in {int(countdown_end - now) + 1}...", (0, 165, 255)
    else:
        status, color = f"REC {now - start_time:.1f}s  (r to stop)", (0, 0, 255)
    cv2.putText(frame, f"Word: {word}  | clips saved this session: {saved_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def handle_key(key):
    global phase, countdown_end, clip_idx
    if key != ord("r"):
        return False
    if phase == "idle":
        clip_idx = next_clip_index()
        phase, countdown_end = "countdown", time.time() + COUNTDOWN
    elif phase == "countdown":
        phase = "idle"  # cancel
    else:
        phase = "idle"
        save_clip()
    return False


hands = h.init_hands(static_mode=False)  # video mode: tracks the hand between frames
print(f"Recording clips for '{word}'. Press r in the video window to start/stop, q to quit.")
h.start_video(hands, handle_frame, handle_key)

if phase == "recording":  # quit mid-recording: don't keep a half-finished clip
    if writer:
        writer.release()
    discard_video()
    print("Quit while recording; that clip was discarded.")
