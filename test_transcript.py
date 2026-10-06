"""
Tests for transcript.py using fake prediction streams (no webcam needed).
Run: python test_transcript.py
"""
from transcript import Transcript

FPS = 30


def run(frames, t=None):
    """frames: list of (letter or None, confidence). Returns the transcript text."""
    t = t or Transcript()
    for i, (letter, conf) in enumerate(frames):
        t.update(letter, conf, i / FPS)
    return t.text


def hold(letter, n=10, conf=90):
    return [(letter, conf)] * n


def check(name, got, want):
    print(f"{'ok  ' if got == want else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return got == want


results = [
    check("holding letters spells a word", run(hold("C") + hold("A") + hold("B")), "CAB"),
    check("letter held a long time is added once", run(hold("A", 90)), "A"),
    check("short flickers are ignored", run(hold("A") + hold("X", 3) + hold("B")), "AB"),
    check("low confidence is ignored", run(hold("A") + hold("X", 20, conf=30) + hold("B")), "AB"),
    check("double letter needs a release",
          run(hold("H") + hold("E") + hold("L") + hold(None, 5) + hold("L") + hold("O")), "HELLO"),
    check("no release means no double letter", run(hold("L") + hold("L")), "L"),
    check("hand gone 1s adds a space", run(hold("H") + hold("I") + hold(None, 35) + hold("O")), "HI O"),
    check("only one space for a long gap", run(hold("A") + hold(None, 200) + hold("B")), "A B"),
    check("no leading space", run(hold(None, 60) + hold("A")), "A"),
]

t = Transcript()
run(hold("A") + hold("B"), t)
t.backspace()
results.append(check("backspace", t.text, "A"))
t.clear()
results.append(check("clear", t.text, ""))

print("\nPASS: all transcript tests passed" if all(results) else "\nFAIL: some transcript tests failed")
