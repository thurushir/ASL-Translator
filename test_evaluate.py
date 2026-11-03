"""
Tests for evaluate.py using hand-made prediction records (no real data, model or webcam needed).
Run: python test_evaluate.py
"""
import json
import os
import tempfile
from types import SimpleNamespace
import numpy as np
import evaluate as E


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return ok


def close(name, got, want, tol=1e-9):
    ok = got is not None and abs(got - want) <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return ok


def rec(word, pred, secs=2.0, signer="anna", ms=None):
    return {"path": f"data_seq/{word}/{signer}_0.npz", "word": word, "signer": signer, "pred": pred,
            "secs": secs, "frames": int(secs * 15) + 1, "ms": ms}


results = []

# --- prediction_records: one record per clip, built from clips and predictions ---
clips = [SimpleNamespace(path="data_seq/HI/anna_0.npz", word="HI", signer="anna",
                         timestamps=np.linspace(0, 3, 46), mask=np.ones(46, bool)),
         SimpleNamespace(path="data_seq/NO/saaya_1.npz", word="NO", signer="saaya",
                         timestamps=np.linspace(0.5, 2.5, 31), mask=np.ones(31, bool))]
recs = E.prediction_records(clips, ["HI", "N"], [12.5, 8.0])
results += [
    check("one record per clip", len(recs), 2),
    check("record fields", recs[1], {"path": "data_seq/NO/saaya_1.npz", "word": "NO", "signer": "saaya",
                                     "pred": "N", "secs": 2.0, "frames": 31, "ms": 8.0}),
    check("timing is optional", E.prediction_records(clips, ["HI", "NO"])[0]["ms"], None),
]
try:
    E.prediction_records(clips, ["HI"])
    results.append(check("mismatched lengths raise", "no error", "ValueError"))
except ValueError:
    results.append(check("mismatched lengths raise", "ValueError", "ValueError"))

# --- CER by word length ---
groups = E.by_length([rec("HI", "HI"), rec("NO", "NA"), rec("HELLO", "HELO")])
results += [
    check("lengths found", sorted(groups), [2, 5]),
    check("length 2 clips", groups[2]["n"], 2),
    close("length 2 CER (1 edit / 4 letters)", groups[2]["cer"], 0.25),
    close("length 2 word acc", groups[2]["word_acc"], 0.5),
    close("length 5 CER", groups[5]["cer"], 0.2),
]

# --- signing speed ---
results += [
    close("speed = letters per second", E.speed(rec("HELLO", "", secs=2.5)), 2.0),
    check("zero-length clip has no speed", E.speed(rec("HI", "", secs=0.0)), None),
]
speedy = [rec("ABCDEF"[:n], "ABCDEF"[:n] if n % 2 else "X", secs=1.0, signer=f"s{n}") for n in range(1, 7)]
sg = E.speed_groups(speedy, n_groups=3)  # speeds 1..6 letters/s
results += [
    check("three speed groups", len(sg), 3),
    check("groups split evenly", [g["n"] for g in sg], [2, 2, 2]),
    check("slowest first", [(g["lo"], g["hi"]) for g in sg], [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
    check("group labels", [g["label"] for g in sg], ["slow", "medium", "fast"]),
    close("slow group CER: A right, AB->X (2 edits) = 2/3", sg[0]["cer"], 2 / 3),
    check("signers counted per group", sg[2]["signers"], {"s5": 1, "s6": 1}),
]

# --- letter confusion matrix ---
m, extra = E.letter_confusion([rec("HELLO", "HELO"), rec("SAT", "AAT"), rec("HI", "BHI")])
L, S, A, DROP = E.LETTERS.index("L"), E.LETTERS.index("S"), E.LETTERS.index("A"), len(E.LETTERS)
results += [
    check("matrix shape (26 true x 26 predicted + dropped)", m.shape, (26, 27)),
    check("one L right", int(m[L, L]), 1),
    check("one L dropped", int(m[L, DROP]), 1),
    check("S read as A", int(m[S, A]), 1),
    check("extra letters counted separately", dict(extra), {"B": 1}),
    check("every true letter counted once", int(m.sum()), len("HELLO") + len("SAT") + len("HI")),
]

# --- inference time ---
t = E.timing([rec("HI", "HI", secs=1, ms=10), rec("NO", "NO", secs=1, ms=20), rec("YES", "YES", secs=2, ms=30)])
results += [
    close("mean ms", t["mean_ms"], 20.0),
    close("median ms", t["median_ms"], 20.0),
    close("real-time factor = compute time / video time", t["realtime_factor"], 60 / 4000),
    check("no timings -> None", E.timing([rec("HI", "HI")]), None),
]

# --- whole run: results JSON in, markdown + image out ---
with tempfile.TemporaryDirectory() as tmp:
    fake = {"date": "2026-11-02", "config": {"seed": 0},
            "predictions": [rec("HELLO", "HELO", 3.0, "anna", 15.0), rec("HI", "HI", 1.0, "saaya", 9.0),
                            rec("SAT", "AAT", 2.0, "thurushi", 11.0), rec("BOOK", "BOOK", 1.5, "anna", 10.0)]}
    with open(os.path.join(tmp, "run.json"), "w") as f:
        json.dump(fake, f)
    out = E.run("run", results_dir=tmp)
    md = open(out["markdown"]).read()
    results += [
        check("markdown written", os.path.exists(out["markdown"]), True),
        check("confusion image written", os.path.getsize(out["image"]) > 1000, True),
        check("markdown has all four sections",
              all(s in md for s in ["## CER by word length", "## CER by signing speed",
                                    "## Letter confusions", "## Inference time"]), True),
    ]
    with open(os.path.join(tmp, "old.json"), "w") as f:
        json.dump({"date": "2026-10-21"}, f)  # a Step 4 results file: no per-clip predictions
    try:
        E.run("old", results_dir=tmp)
        results.append(check("old results file gives a clear error", "no error", "SystemExit"))
    except SystemExit as e:
        results.append(check("old results file gives a clear error", "--save" in str(e) or "train_ctc" in str(e), True))

print("\nPASS: all evaluate tests passed" if all(results) else "\nFAIL: some evaluate tests failed")
