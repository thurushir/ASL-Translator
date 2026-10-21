"""
Tests for compare_runs.py using small hand-made results (no real data or training needed).
Run: python test_compare_runs.py
"""
import compare_runs as C


def check(name, got, want):
    print(f"{'ok  ' if got == want else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return got == want


def fake_run(cers, baseline_cer=0.4):
    """A results dict shaped like train_ctc.py's JSON, with one CER per signer."""
    scores = {s: {"cer": c, "word_acc": 1 - c, "letter_recall": 1 - c / 2, "cer_no_jz": c}
              for s, c in cers.items()}
    return {"signers": [s for s in cers if s != "all"], "scores": scores,
            "baseline_b": {s: {"cer": baseline_cer, "word_acc": 0.2, "letter_recall": 0.7, "cer_no_jz": 0.3}
                           for s in cers}}


results = []
runs = [fake_run({"anna": 0.30, "saaya": 0.10, "all": 0.20}),
        fake_run({"anna": 0.20, "saaya": 0.20, "all": 0.20}),
        fake_run({"anna": 0.40, "saaya": 0.30, "all": 0.35})]

s = C.stats(runs, "anna", "cer")
results.append(check("values in run order", s["values"], [0.30, 0.20, 0.40]))
results.append(check("mean", round(s["mean"], 4), 0.30))
results.append(check("min and max", (s["min"], s["max"]), (0.20, 0.40)))
results.append(check("spread = max - min", round(s["spread"], 4), 0.20))

table = C.markdown(runs, ["seed0", "seed1", "seed2"])
results.append(check("table has a row per signer plus all", sum(line.startswith("| LSTM") for line in table.splitlines()), 3))
results.append(check("table shows the mean CER for anna", "| LSTM+CTC | anna | 30.0% / 20.0% / 40.0% | **30.0%** |" in table, True))
results.append(check("baseline B row is included", "| Baseline B | all |" in table, True))

try:
    C.stats([fake_run({"anna": 0.3, "all": 0.3}), fake_run({"saaya": 0.3, "all": 0.3})], "anna", "cer")
    results.append(check("runs with different signers are refused", "no error", "ValueError"))
except ValueError:
    results.append(check("runs with different signers are refused", "ValueError", "ValueError"))

print("\nPASS: all compare_runs tests passed" if all(results) else "\nFAIL: some compare_runs tests failed")
