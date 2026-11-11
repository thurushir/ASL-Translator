"""
Tests for predict_live_ctc.py's LiveSession: webcam frames in, text bar and overlay out.
No webcam or trained model needed. Run: python test_predict_live_ctc.py
"""
import numpy as np
import dataset as D
from predict_live_ctc import LiveSession, HOW_LINES, key_lines


def check(name, got, want):
    print(f"{'ok  ' if got == want else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return got == want


def sharp_output(word, p=0.97):
    """Model output that clearly spells word: each letter on its own frame, blanks in between."""
    rows = []
    for ch in word:
        for k in (D.encode(ch)[0], D.BLANK):
            row = np.full(D.NUM_CLASSES, (1 - p) / (D.NUM_CLASSES - 1))
            row[k] = p
            rows.append(np.log(row))
    if not rows:  # an empty word: only blanks
        row = np.full(D.NUM_CLASSES, (1 - p) / (D.NUM_CLASSES - 1))
        row[D.BLANK] = p
        rows = [np.log(row)] * 3
    return np.array(rows)


class FakeRecognizer:
    """Spells word N as the Nth entry of `words`, so tests know what each finished word should become."""
    def __init__(self, words):
        self.words, self.seen = list(words), []

    def log_probs(self, clip):
        self.seen.append(len(clip.mask))
        return sharp_output(self.words.pop(0))


rng = np.random.default_rng(0)


def feed(session, frames, fps=30, start=0.0):
    for i, f in enumerate(frames):
        session.on_frame(rng.random((21, 3)) if f == "H" else None, start + i / fps)


results = []

s = LiveSession(FakeRecognizer(["HELLO"]))
feed(s, "H" * 20)
results.append(check("while signing, nothing is added yet", s.text, ""))
results.append(check("status says a word is being signed", s.signing, True))
feed(s, "." * 31, start=1.0)
results.append(check("after 1 s without a hand the word appears", s.text, "HELLO"))
results.append(check("status is back to waiting", s.signing, False))
results.append(check("last word is remembered for the overlay", s.last_word, "HELLO"))

s = LiveSession(FakeRecognizer(["HI", "THERE"]))
feed(s, "H" * 20 + "." * 40 + "H" * 20 + "." * 40)
results.append(check("words are separated by spaces", s.text, "HI THERE"))

s = LiveSession(FakeRecognizer(["CAT"]))
feed(s, "H" * 20)
s.end_word()
results.append(check("space key ends the word right away", s.text, "CAT"))

s = LiveSession(FakeRecognizer(["A", "B"]))
feed(s, "H" * 20 + "." * 40 + "H" * 20 + "." * 40)
s.delete_word()
results.append(check("backspace removes the last word", s.text, "A"))
s.clear()
results.append(check("c clears everything", (s.text, s.last_word), ("", "")))

s = LiveSession(FakeRecognizer([""]))
feed(s, "H" * 20 + "." * 40)
results.append(check("a word the model reads as empty adds nothing", s.text, ""))

fake = FakeRecognizer(["X"])
s = LiveSession(fake)
feed(s, "." * 5 + "H" * 3 + "." * 40)
results.append(check("a hand passing through briefly is not sent to the model", fake.seen, []))

heard = []
s = LiveSession(FakeRecognizer(["HI", "YO"]), on_word=lambda word, frames: heard.append((word, frames)))
feed(s, "H" * 20 + "." * 40 + "H" * 12 + "." * 40)
results.append(check("on_word is told each word and its frame count (for the terminal log)",
                     heard, [("HI", 20), ("YO", 12)]))

# --- Step 6 Part 6: beam search, confidence, next guesses, optional word list ---
s = LiveSession(FakeRecognizer(["CAT"]))
feed(s, "H" * 20 + "." * 40)
results.append(check("a clear word gets high confidence (> 0.9)", s.confidence > 0.9, True))
results.append(check("up to 2 next-best guesses, not including the word itself",
                     (len(s.next_guesses) <= 2, "CAT" not in [w for w, _ in s.next_guesses]), (True, True)))
results.append(check("guesses come with their confidence, best first",
                     [c for _, c in s.next_guesses] == sorted([c for _, c in s.next_guesses], reverse=True), True))

s = LiveSession(FakeRecognizer(["SCHOL"]), words=["SCHOOL", "CAT", "HELLO"])
results.append(check("word list is off by default", s.show_word_list, False))
feed(s, "H" * 20 + "." * 40)
results.append(check("closest known word is worked out for the overlay", s.known_word, "SCHOOL"))
results.append(check("the text bar keeps the model's own spelling", s.text, "SCHOL"))
s.toggle_word_list()
results.append(check("w turns the word list on", s.show_word_list, True))
s.toggle_word_list()
results.append(check("w again turns it off", s.show_word_list, False))

s = LiveSession(FakeRecognizer(["CAT"]), words=["CAT"], show_word_list=True)
results.append(check("--word-list starts with it on", s.show_word_list, True))

s = LiveSession(FakeRecognizer(["CAT"]))
feed(s, "H" * 20 + "." * 40)
results.append(check("without a word list there is no known word", s.known_word, ""))
s.clear()
results.append(check("c also clears the overlay", (s.confidence, s.next_guesses, s.known_word), (0.0, [], "")))

s = LiveSession(FakeRecognizer([]))
results.append(check("the on-screen guide is shown by default", s.show_help, True))
s.toggle_help()
results.append(check("h hides the guide", s.show_help, False))
results.append(check("'how it works' box is short (fits top right)", 2 <= len(HOW_LINES) <= 6, True))
keys = key_lines(LiveSession(FakeRecognizer([])))
results.append(check("keys box: one key per line, every key the app uses",
                     [line.split()[0] for line in keys], ["space", "backspace", "c", "w", "h", "q"]))
s = LiveSession(FakeRecognizer([]))
results.append(check("keys box shows the practice-word list is off", any("OFF" in line for line in key_lines(s)), True))
s.toggle_word_list()
results.append(check("keys box shows it is on after w", any("ON" in line for line in key_lines(s)), True))

print("\nPASS: all predict_live_ctc tests passed" if all(results) else "\nFAIL: some predict_live_ctc tests failed")
