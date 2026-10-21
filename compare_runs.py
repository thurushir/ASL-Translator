"""
Averages several train_ctc.py runs (e.g. the same settings with different --seed) into one table.
One run is not enough: the same model scored 25.3%, 27.8% and 30.7% CER with seeds 0, 1 and 2.
Compare changes by their mean over seeds, and only trust a gain bigger than the spread between seeds.

  python compare_runs.py lstm_ctc lstm_ctc_seed1 lstm_ctc_seed2 --out lstm_ctc_3seeds
      reads results/<name>.json for each run, writes results/<out>.md

Tests: python test_compare_runs.py
"""
import argparse
import json
import os
import statistics

RESULTS_DIR = "./results"


def stats(runs, signer, key):
    """values (one per run, in order), mean, min, max and spread (max - min) of one score."""
    missing = [i for i, r in enumerate(runs) if signer not in r["scores"]]
    if missing:
        raise ValueError(f"run(s) {missing} have no scores for {signer!r}; compare runs on the same signers")
    values = [r["scores"][signer][key] for r in runs]
    return {"values": values, "mean": statistics.mean(values), "min": min(values), "max": max(values),
            "spread": max(values) - min(values)}


def pct(x):
    return f"{100 * x:.1f}%"


def markdown(runs, names):
    signers = runs[0]["signers"] + ["all"]
    L = [f"# LSTM + CTC averaged over {len(runs)} runs", "",
         f"Runs: {', '.join(f'`results/{n}.json`' for n in names)}. Held-out signer scores; lower CER is better.",
         "Compare a change by its **mean** over seeds; a gain smaller than the CER range between seeds may be noise.", "",
         f"| Model | Signer | CER per run ({' / '.join(names)}) | CER mean | CER range | Word acc mean "
         "| Letter recall mean | CER (no J/Z) mean |",
         "|---|---|---|---|---|---|---|---|"]
    for s in signers:
        c = stats(runs, s, "cer")
        L.append(f"| LSTM+CTC | {s} | {' / '.join(pct(v) for v in c['values'])} | **{pct(c['mean'])}** "
                 f"| {pct(c['min'])}-{pct(c['max'])} | {pct(stats(runs, s, 'word_acc')['mean'])} "
                 f"| {pct(stats(runs, s, 'letter_recall')['mean'])} | {pct(stats(runs, s, 'cer_no_jz')['mean'])} |")
    for s in signers:
        b = runs[0]["baseline_b"][s]
        L.append(f"| Baseline B | {s} | {pct(b['cer'])} | {pct(b['cer'])} | - | {pct(b['word_acc'])} "
                 f"| {pct(b['letter_recall'])} | {pct(b['cer_no_jz'])} |")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="+", help="results names, e.g. lstm_ctc lstm_ctc_seed1 (reads results/<name>.json)")
    p.add_argument("--out", required=True, help="writes results/<out>.md")
    args = p.parse_args()
    runs = []
    for n in args.names:
        with open(os.path.join(RESULTS_DIR, f"{n}.json")) as f:
            runs.append(json.load(f))
    table = markdown(runs, args.names)
    with open(os.path.join(RESULTS_DIR, f"{args.out}.md"), "w") as f:
        f.write(table)
    print(table)
    print(f"Wrote results/{args.out}.md")


if __name__ == "__main__":
    main()
