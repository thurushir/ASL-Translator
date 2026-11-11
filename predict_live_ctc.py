"""
Live fingerspelling with the temporal model (LSTM + CTC). predict_live.py (the static model) is unchanged.

Run: python predict_live_ctc.py [--model models/<name>.pt] [--word-list]
  Fingerspell a word, then drop your hand (or press space): after 1 s the word appears in the text bar.
  The model reads the whole word at once, so words appear when you finish them, not letter by letter.
  The overlay shows the word from beam search, its confidence and the next two guesses.
Keys: space = end the word now, backspace = delete last word, c = clear,
      w = show/hide the closest word from word_list.txt (off by default), h = show/hide the on-screen guide, q = quit.

The word-list line only ever picks one of our 55 practice words, so it can't spell a new word (a name, say);
the text bar always keeps the model's own spelling.
Default model: models/step6_final.pt (trained on all three of us). For an honest test of yourself, use a model
that never trained on you: --model models/step6_check_<yourname>.pt (trained on the other two signers).
"""
import argparse
import time
import decode as DE
from live_ctc import WordBuilder, Recognizer

DEFAULT_MODEL = "models/step6_final.pt"
WORD_LIST = "word_list.txt"
HOW_LINES = [  # top-right box of the on-screen guide (h hides both boxes)
    "1. Fingerspell a whole word",
    "2. Drop your hand: it appears in 1 s",
    "% = how sure the model is",
    "Other guesses = next most likely spellings",
]


def key_lines(session):
    """Bottom-right box: one key per line, with the practice-word list's current state."""
    return ["space       end the word now",
            "backspace   delete last word",
            "c           clear all",
            f"w           practice words: {'ON' if session.show_word_list else 'OFF'}",
            "h           hide this guide",
            "q           quit"]


class LiveSession:
    """Webcam frames in, text and overlay out. Free of camera code so it can be tested (test_predict_live_ctc.py)."""

    def __init__(self, recognizer, builder=None, on_word=None, words=None, show_word_list=False):
        self.recognizer = recognizer
        self.builder = builder or WordBuilder()
        self.on_word = on_word  # called with (word, frames) for every finished word, e.g. to log it
        self.words = list(words or [])
        self.show_word_list = show_word_list
        self.show_help = True
        self.text = ""
        self._clear_overlay()

    def _clear_overlay(self):
        self.last_word, self.confidence, self.next_guesses, self.known_word = "", 0.0, [], ""

    @property
    def signing(self):
        return self.builder.in_word

    def on_frame(self, points, now):
        self._add(self.builder.add(points, now))

    def end_word(self):
        self._add(self.builder.finish())

    def _add(self, clip):
        if clip is None:
            return
        lp = self.recognizer.log_probs(clip)
        ranked = DE.confidence(DE.beam_search(lp))
        word, conf = ranked[0]
        if word:
            self.last_word, self.confidence = word, conf
            self.next_guesses = [(w, c) for w, c in ranked[1:] if w][:2]
            self.known_word = DE.lexicon_decode(lp, self.words)[0][0] if self.words else ""
            self.text = f"{self.text} {word}".strip()
        if self.on_word:  # after the overlay is updated, so a logger can read confidence and known_word
            self.on_word(word, len(clip.mask))

    def toggle_word_list(self):
        self.show_word_list = not self.show_word_list

    def toggle_help(self):
        self.show_help = not self.show_help

    def delete_word(self):
        self.text = " ".join(self.text.split()[:-1])

    def clear(self):
        self.text = ""
        self._clear_overlay()


def main():
    import cv2
    import hand_utils as h
    import features as F
    from collect_sequences import load_word_list

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"saved model (default {DEFAULT_MODEL})")
    p.add_argument("--word-list", action="store_true", help="start with the closest-known-word line shown (w toggles)")
    args = p.parse_args()
    start = time.time()

    def log(word, frames):
        extra = (f"  conf {session.confidence:.0%}" if word else "") + \
                (f"  known word: {session.known_word}" if word and session.known_word else "")
        print(f"[{time.time() - start:6.1f}s] {word or '(nothing)':<12} ({frames} frames){extra}")

    session = LiveSession(Recognizer.load(args.model), on_word=log, words=load_word_list(WORD_LIST),
                          show_word_list=args.word_list)
    font = cv2.FONT_HERSHEY_TRIPLEX

    def conf_color(c):  # BGR: green when sure, orange, red when unsure
        return (0, 200, 0) if c >= 0.8 else (0, 165, 255) if c >= 0.5 else (0, 0, 255)

    def draw_box(frame, title, lines, top=None, bottom=None, columns=False, scale=0.7, pad=14):
        """
        A see-through box in the top- or bottom-right corner, sized to its text (the hand stays visible).
        columns=True: each line is "key   description", drawn as two aligned columns.
        """
        mono, line_h = cv2.FONT_HERSHEY_SIMPLEX, int(42 * scale)
        size = lambda t: cv2.getTextSize(t, mono, scale, 2)[0][0]
        rows = [line.split(None, 1) if columns else [line] for line in lines]
        key_w = max(size(r[0]) for r in rows) + 25 if columns else 0
        box_w = max([size(title)] + [key_w + size(r[-1]) if columns else size(r[0]) for r in rows]) + 2 * pad
        box_h = line_h * (len(lines) + 1) + pad
        y0 = top if top is not None else bottom - box_h
        x0 = frame.shape[1] - box_w - 15
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, title, (x0 + pad, y0 + line_h), mono, scale, (0, 215, 255), 2, cv2.LINE_AA)
        for i, row in enumerate(rows, start=2):
            color = (255, 200, 0) if "ON" in row[-1] else (255, 255, 255)
            y = y0 + i * line_h
            if columns:
                cv2.putText(frame, row[0], (x0 + pad, y), mono, scale, (0, 215, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, row[1], (x0 + pad + key_w, y), mono, scale, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, row[0], (x0 + pad, y), mono, scale, color, 2, cv2.LINE_AA)

    def process_frame(frame, results):
        h.draw_hand_landmarks(frame, results)
        session.on_frame(F.raw_landmarks(results), time.time())
        status, color = (("Signing...", (0, 0, 255)) if session.signing
                         else ("Fingerspell a word, then drop your hand", (0, 200, 0)))
        cv2.putText(frame, status, (20, 40), font, 0.8, color, 2, cv2.LINE_AA)
        if session.last_word:
            cv2.putText(frame, f"Model spelled: {session.last_word}  {session.confidence:.0%} sure", (20, 90), font,
                        1.1, conf_color(session.confidence), 2, cv2.LINE_AA)
            if session.next_guesses:
                guesses = "   ".join(f"{w} {c:.0%}" for w, c in session.next_guesses)
                cv2.putText(frame, f"Other guesses: {guesses}", (20, 125), font, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
            if session.show_word_list and session.known_word:
                cv2.putText(frame, f"Closest practice word: {session.known_word}", (20, 165), font, 0.9,
                            (255, 200, 0), 2, cv2.LINE_AA)
        height, width = frame.shape[:2]
        if session.show_help:
            draw_box(frame, "HOW IT WORKS", HOW_LINES, top=15)
            draw_box(frame, "KEYS", key_lines(session), bottom=height - 75, columns=True)
        cv2.rectangle(frame, (0, height - 60), (width, height), (0, 0, 0), -1)
        cv2.putText(frame, session.text[-30:] + "_", (20, height - 20), font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    def handle_key(key):
        if key == ord(" "):
            session.end_word()
        elif key in (8, 127):  # backspace: 8 on Windows, 127 on macOS
            session.delete_word()
        elif key == ord("c"):
            session.clear()
        elif key == ord("w"):
            session.toggle_word_list()
        elif key == ord("h"):
            session.toggle_help()
        return False

    print(f"Model: {args.model}\nKeys: space = end word, backspace = delete word, c = clear, "
          f"w = practice-word list on/off, h = guide on/off, q = quit.\n")
    h.start_video(h.init_hands(static_mode=False), process_frame, handle_key)
    session.end_word()
    if session.text:
        print(f"Final text: {session.text}")


if __name__ == "__main__":
    main()
