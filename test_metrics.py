"""
Tests for metrics.py with hand-checked examples.
Run: python test_metrics.py
Argument order everywhere is (truth, prediction).
"""
import metrics as M


def check(name, got, want):
    ok = abs(got - want) < 1e-9 if isinstance(want, float) else got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return ok


results = [
    # --- edit distance: fewest letter adds / removes / swaps ---
    check("same word = 0 edits", M.edit_distance("HELLO", "HELLO"), 0),
    check("one letter missing = 1 edit", M.edit_distance("HELLO", "HELO"), 1),
    check("one letter extra = 1 edit", M.edit_distance("CAT", "CAAT"), 1),
    check("one letter swapped = 1 edit", M.edit_distance("SAT", "AAT"), 1),
    check("empty prediction = word length", M.edit_distance("CAT", ""), 3),
    check("nothing in common", M.edit_distance("AB", "CD"), 2),

    # --- CER for one word = edits / letters in the true word ---
    check("CER perfect", M.cer("HELLO", "HELLO"), 0.0),
    check("CER HELO vs HELLO", M.cer("HELLO", "HELO"), 0.2),
    check("CER empty prediction", M.cer("CAT", ""), 1.0),
    check("CER can go above 1 (lots of extra letters)", M.cer("HI", "HIHIHI"), 2.0),

    # --- alignment: which letter turned into which ---
    check("alignment of SAT vs AAT", M.align("SAT", "AAT"), [("S", "A"), ("A", "A"), ("T", "T")]),
    check("alignment marks a missing letter with None",
          M.align("CAT", "CT"), [("C", "C"), ("A", None), ("T", "T")]),
    check("alignment marks an extra letter with None",
          M.align("CAT", "CAXT"), [("C", "C"), ("A", "A"), (None, "X"), ("T", "T")]),
    check("errors() lists only the mistakes", M.errors("SAT", "AAT"), [("S", "A")]),
]

# --- scoring a batch ---
truths = ["CAT", "DOG", "HELLO", "SAT"]
preds = ["CAT", "DOG", "HELO", "AAT"]
s = M.score(truths, preds)
results += [
    check("batch word accuracy (2 of 4 exactly right)", s["word_acc"], 0.5),
    check("batch CER = total edits / total letters (2 / 14)", s["cer"], 2 / 14),
    check("batch count", s["n"], 4),
    check("letter recall = true letters found / total (12 / 14)", s["letter_recall"], 12 / 14),
    check("extra letters don't lower letter recall",
          M.score(["CAT"], ["CXAXTX"])["letter_recall"], 1.0),
    check("confusion counts S read as A", s["confusion"][("S", "A")], 1),
    check("confusion counts a dropped L", s["confusion"][("L", None)], 1),
    check("confusion counts correct letters too", s["confusion"][("C", "C")], 1),
]

# --- per-letter report built from the confusion counts ---
rep = M.letter_report(s["confusion"])
results += [
    check("S was right 0 of 1 times", (rep["S"]["total"], rep["S"]["correct"]), (1, 0.0)),
    check("S was mistaken for A", rep["S"]["mistaken_for"], [("A", 1)]),
    check("L (twice in HELLO): 1 of 2 right, one dropped",
          (rep["L"]["total"], rep["L"]["correct"], rep["L"]["mistaken_for"]), (2, 0.5, [("(dropped)", 1)])),
]
# found on real data: a letter dropped as often as it was swapped (a tie) used to crash the sort
tie = M.letter_report(M.score(["SS"], ["A"])["confusion"])
results.append(check("tie between dropped and swapped doesn't crash",
                     sorted(tie["S"]["mistaken_for"]), [("(dropped)", 1), ("A", 1)]))
try:
    M.score(["A", "B"], ["A"])
    results.append(check("mismatched lengths raise an error", "no error", "ValueError"))
except ValueError:
    results.append(check("mismatched lengths raise an error", "ValueError", "ValueError"))

print("\nPASS: all metrics tests passed" if all(results) else "\nFAIL: some metrics tests failed")
