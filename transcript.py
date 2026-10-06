"""
Turns a stream of per-frame letter predictions into persistent text.

Rules (tune the constants below if it feels too slow or too jumpy):
  - A letter is added once it is predicted for HOLD_FRAMES frames in a row at >= MIN_CONF % confidence.
  - The SAME letter can only be added again after a "release": RELEASE_FRAMES frames in a row
    where that letter isn't confidently seen (relax your hand briefly, e.g. between the L's in HELLO).
  - If no hand is seen for SPACE_AFTER seconds, a space is added (word break).
"""

HOLD_FRAMES = 8
MIN_CONF = 50.0
RELEASE_FRAMES = 4
SPACE_AFTER = 1.0


class Transcript:
    def __init__(self, hold_frames=HOLD_FRAMES, min_conf=MIN_CONF,
                 release_frames=RELEASE_FRAMES, space_after=SPACE_AFTER):
        self.hold_frames = hold_frames
        self.min_conf = min_conf
        self.release_frames = release_frames
        self.space_after = space_after
        self.text = ""
        self._candidate, self._count = None, 0  # letter currently being held, and for how many frames
        self._last = None                       # last letter added
        self._gap = 0                           # frames in a row without a confident self._last
        self._released = True                   # True once self._last may be added again
        self._no_hand_since = None

    def update(self, letter, conf, now):
        """Call once per frame. letter=None when no hand is detected. conf in %, now in seconds."""
        confident = letter is not None and conf >= self.min_conf

        # release tracking for repeated letters
        self._gap = 0 if (confident and letter == self._last) else self._gap + 1
        if self._gap >= self.release_frames:
            self._released = True

        # word break when the hand is gone long enough
        if letter is None:
            if self._no_hand_since is None:
                self._no_hand_since = now
            elif now - self._no_hand_since >= self.space_after:
                self.add_space()
        else:
            self._no_hand_since = None

        # hold tracking
        if not confident:
            self._candidate, self._count = None, 0
            return
        if letter == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = letter, 1

        if self._count >= self.hold_frames and (letter != self._last or self._released):
            self.text += letter
            self._last, self._released, self._gap = letter, False, 0
            self._candidate, self._count = None, 0

    def add_space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "
            self._last, self._released = None, True  # a new word can start with any letter

    def backspace(self):
        self.text = self.text[:-1]

    def clear(self):
        self.text = ""
        self._last, self._released = None, True
