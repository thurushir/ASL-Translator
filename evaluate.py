"""
The evaluation our proposal promised (Section 3), computed from a training run's saved per-clip predictions.
No retraining: train_ctc.py stores every held-out clip's prediction (and how long it took) in results/<name>.json.

  python evaluate.py <name>      e.g. python evaluate.py step5_lstm
      reads  results/<name>.json
      writes results/<name>_eval.md        CER by word length, by signing speed, letter confusions, inference time
             results/<name>_confusion.png  letter confusion matrix (rows = true letter, each row adds up to 100%)

Signing speed = letters in the word / clip length in seconds. The clip includes the resting hand at the start
and end, so this measures the whole clip, not just the signing; it still separates fast and slow signers.
Tests: python test_evaluate.py
"""
import argparse
import json
import os
from collections import Counter
import numpy as np
import metrics as M

RESULTS_DIR = "./results"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DROPPED = len(LETTERS)  # last column of the confusion matrix: the letter was left out


# ---------- per-clip records (built by train_ctc.py) ----------

def prediction_records(clips, preds, ms=None):
    """One dict per clip: path, word, signer, predicted word, clip length (s), frames, inference time (ms)."""
    if len(clips) != len(preds) or (ms is not None and len(ms) != len(clips)):
        raise ValueError(f"{len(clips)} clips but {len(preds)} predictions" + (f" and {len(ms)} timings" if ms else ""))
    out = []
    for i, (c, p) in enumerate(zip(clips, preds)):
        ts = c.timestamps
        out.append({"path": c.path, "word": c.word, "signer": c.signer, "pred": p,
                    "secs": round(float(ts[-1] - ts[0]), 3) if len(ts) else 0.0, "frames": int(len(c.mask)),
                    "ms": None if ms is None else round(float(ms[i]), 3)})
    return out


# ---------- analyses ----------

def _score(records):
    s = M.score([r["word"] for r in records], [r["pred"] for r in records])
    return {"n": s["n"], "cer": s["cer"], "word_acc": s["word_acc"], "letter_recall": s["letter_recall"]}


def by_length(records):
    """{word length: scores} for every length present."""
    lengths = sorted({len(r["word"]) for r in records})
    return {n: _score([r for r in records if len(r["word"]) == n]) for n in lengths}


def speed(record):
    """Letters per second over the whole clip, or None for a zero-length clip."""
    return len(record["word"]) / record["secs"] if record["secs"] > 0 else None


def speed_groups(records, n_groups=3):
    """Clips sorted by speed and cut into n_groups equal-sized groups, slowest first, each with its scores."""
    timed = sorted((r for r in records if speed(r) is not None), key=speed)
    labels = ["slow", "medium", "fast"] if n_groups == 3 else [f"group {i + 1}" for i in range(n_groups)]
    groups = []
    for label, idx in zip(labels, np.array_split(np.arange(len(timed)), n_groups)):
        part = [timed[i] for i in idx]
        if not part:
            continue
        groups.append({"label": label, "lo": round(speed(part[0]), 2), "hi": round(speed(part[-1]), 2),
                       **_score(part), "signers": Counter(r["signer"] for r in part)})
    return groups


def letter_confusion(records):
    """(26 x 27 counts: true letter x predicted letter or dropped, Counter of extra predicted letters)."""
    m = np.zeros((len(LETTERS), len(LETTERS) + 1), dtype=int)
    extra = Counter()
    for r in records:
        for t, p in M.align(r["word"], r["pred"]):
            if t is None:
                extra[p] += 1
            else:
                m[LETTERS.index(t), DROPPED if p is None else LETTERS.index(p)] += 1
    return m, extra


def timing(records):
    """Inference time per clip (ms) and the real-time factor (compute time / video time), or None if not timed."""
    timed = [r for r in records if r.get("ms") is not None]
    if not timed:
        return None
    ms = np.array([r["ms"] for r in timed], dtype=float)
    return {"n": len(timed), "mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
            "p95_ms": float(np.percentile(ms, 95)), "max_ms": float(ms.max()),
            "realtime_factor": float(ms.sum() / (1000 * sum(r["secs"] for r in timed)))}


# ---------- output ----------

def plot_confusion(m, path, title):
    import matplotlib
    matplotlib.use("Agg")  # no window needed
    import matplotlib.pyplot as plt
    rows = m.sum(axis=1, keepdims=True)
    share = np.divide(m, rows, out=np.zeros(m.shape), where=rows > 0)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.axvline(DROPPED - 0.5, color="gray", linewidth=1)  # "drop" is not a letter: set it apart
    ax.set_xticks(range(m.shape[1]), list(LETTERS) + ["drop"], rotation=90)
    ax.set_yticks(range(m.shape[0]), list(LETTERS))
    ax.set_xlabel("Predicted letter (drop = left out)")
    ax.set_ylabel("True letter")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Share of that true letter")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pct(x):
    return f"{100 * x:.1f}%"


def write_markdown(name, res, records, path, image_name):
    L = [f"# Evaluation of `{name}`", "",
         f"From the per-clip predictions in `results/{name}.json` ({res.get('date', '?')}, seed "
         f"{res.get('config', {}).get('seed', '?')}). Every prediction is on a held-out signer. "
         f"{len(records)} clips.", "",
         "## CER by word length", "",
         "| Letters | Clips | CER | Word acc | Letter recall |", "|---|---|---|---|---|"]
    for n, s in by_length(records).items():
        L.append(f"| {n} | {s['n']} | {pct(s['cer'])} | {pct(s['word_acc'])} | {pct(s['letter_recall'])} |")
    L += ["", "## CER by signing speed", "",
          "Speed = letters in the word / clip length (s), including the resting hand at the start and end. "
          "Clips are split into three equal groups.", "",
          "| Group | Letters per second | Clips | CER | Word acc | Clips per signer |", "|---|---|---|---|---|---|"]
    for g in speed_groups(records):
        who = ", ".join(f"{s} {n}" for s, n in sorted(g["signers"].items()))
        L.append(f"| {g['label']} | {g['lo']:.2f}–{g['hi']:.2f} | {g['n']} | {pct(g['cer'])} | {pct(g['word_acc'])} | {who} |")
    L += ["", "| Signer | Average letters per second |", "|---|---|"]
    for s in sorted({r["signer"] for r in records}):
        sp = [speed(r) for r in records if r["signer"] == s and speed(r) is not None]
        L.append(f"| {s} | {np.mean(sp):.2f} |")
    m, extra = letter_confusion(records)
    mistakes = sorted(((LETTERS[i], "(dropped)" if j == DROPPED else LETTERS[j], int(m[i, j]))
                       for i in range(m.shape[0]) for j in range(m.shape[1]) if j != i and m[i, j]),
                      key=lambda x: -x[2])
    L += ["", "## Letter confusions", "", f"![Letter confusion matrix]({image_name})", "",
          "Most common mistakes (true letter → what it became):", "",
          "| True | Became | Times |", "|---|---|---|"]
    L += [f"| {t} | {p} | {n} |" for t, p, n in mistakes[:15]]
    L += ["", "Extra letters added (not in the word): "
          + (", ".join(f"{k}×{v}" for k, v in extra.most_common(10)) or "none") + "."]
    t = timing(records)
    L += ["", "## Inference time", ""]
    if t is None:
        L.append("Not measured in this run.")
    else:
        L += [f"One clip at a time on this computer's CPU, after the clip has ended ({t['n']} clips):", "",
              "| Mean | Median | 95th percentile | Slowest | Real-time factor |", "|---|---|---|---|---|",
              f"| {t['mean_ms']:.1f} ms | {t['median_ms']:.1f} ms | {t['p95_ms']:.1f} ms | {t['max_ms']:.1f} ms "
              f"| {t['realtime_factor']:.4f} |", "",
              "Real-time factor = compute time / video length; below 1 means faster than real time. "
              "The LSTM is bidirectional, so it needs the whole word before it can answer: in the live app the "
              "delay is the time to the end of the word plus this inference time."]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def run(name, results_dir=RESULTS_DIR):
    with open(os.path.join(results_dir, f"{name}.json")) as f:
        res = json.load(f)
    records = res.get("predictions")
    if not records:
        raise SystemExit(f"results/{name}.json has no per-clip predictions. It was made before Step 5; "
                         f"re-run it with the current train_ctc.py, e.g. python train_ctc.py --save --name {name}")
    image = f"{name}_confusion.png"
    m, _ = letter_confusion(records)
    plot_confusion(m, os.path.join(results_dir, image), f"LSTM + CTC letter confusions ({name}, held-out signers)")
    md = os.path.join(results_dir, f"{name}_eval.md")
    write_markdown(name, res, records, md, image)
    return {"markdown": md, "image": os.path.join(results_dir, image)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="results name used with train_ctc.py --name")
    out = run(p.parse_args().name)
    print(open(out["markdown"]).read())
    print(f"Wrote {out['markdown']} and {out['image']}")


if __name__ == "__main__":
    main()
