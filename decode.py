"""
Better decoding for the CTC model's output (Step 6a). Input everywhere: (frames, 27) log-probabilities,
e.g. from live_ctc.Recognizer.log_probs(clip). Greedy decoding (model_ctc.greedy_decode) takes the single best
output per frame; these consider whole words instead.

  beam_search(lp, width)     -> [(word, log P(word))] best first. CTC prefix beam search: a word's probability
                                is the sum over every frame alignment that spells it (A A, A -, - A all spell A).
  lexicon_decode(lp, words)  -> [(word, log P(word))] for each word in a list, best first: the list word the clip
                                most likely spells (exact CTC probability). Flatters known words; can't spell new ones.
  confidence(candidates)     -> [(word, share)] each candidate's share of the candidates' total probability,
                                for the live overlay (top word's confidence + next guesses).
"""
import numpy as np
import torch
import dataset as D

BEAM_WIDTH = 10  # fixed before any results (a standard value), never tuned on a test signer


def _logsumexp(*xs):
    m = max(xs)
    return m if m == -np.inf else m + np.log(sum(np.exp(x - m) for x in xs))


def beam_search(log_probs, width=BEAM_WIDTH):
    """CTC prefix beam search. Returns up to `width` (word, log probability) pairs, best first."""
    lp = np.asarray(log_probs, dtype=np.float64)
    # each prefix (tuple of labels) -> (log P(ends in blank), log P(ends in its last letter))
    beams = {(): (0.0, -np.inf)}
    for t in range(len(lp)):
        nxt = {}

        def add(prefix, b=-np.inf, nb=-np.inf):
            ob, onb = nxt.get(prefix, (-np.inf, -np.inf))
            nxt[prefix] = (_logsumexp(ob, b), _logsumexp(onb, nb))

        for prefix, (pb, pnb) in beams.items():
            total = _logsumexp(pb, pnb)
            add(prefix, b=total + lp[t, D.BLANK])                 # blank: prefix unchanged
            if prefix:
                add(prefix, nb=pnb + lp[t, prefix[-1]])           # repeat of the last letter merges into it
            for k in range(1, D.NUM_CLASSES):
                if prefix and k == prefix[-1]:
                    add(prefix + (k,), nb=pb + lp[t, k])          # same letter again only after a blank (LL)
                else:
                    add(prefix + (k,), nb=total + lp[t, k])
        beams = dict(sorted(nxt.items(), key=lambda kv: -_logsumexp(*kv[1]))[:width])
    ranked = sorted(((D.decode(p), _logsumexp(*s)) for p, s in beams.items()), key=lambda ws: -ws[1])
    return ranked


def lexicon_decode(log_probs, words):
    """Each list word with log P(word | clip) under CTC, best first."""
    lp = torch.as_tensor(np.asarray(log_probs), dtype=torch.float64)[:, None, :]  # (frames, batch 1, 27)
    scored = []
    for w in words:
        if len(w) == 0:
            continue
        nll = torch.nn.functional.ctc_loss(lp, torch.tensor([D.encode(w)]), torch.tensor([lp.shape[0]]),
                                           torch.tensor([len(w)]), blank=D.BLANK, reduction="sum",
                                           zero_infinity=False)
        scored.append((w, -nll.item()))
    return sorted(scored, key=lambda ws: -ws[1])


def confidence(candidates):
    """[(word, log P)] -> [(word, share of the candidates' total probability)], best first."""
    if not candidates:
        return []
    logs = np.array([s for _, s in candidates])
    shares = np.exp(logs - logs.max())
    shares /= shares.sum()
    return [(w, float(c)) for (w, _), c in zip(candidates, shares)]


# ---------- comparison on the recorded clips ----------

def compare(run, word_list="word_list.txt"):
    """
    Greedy vs beam search vs word list, on every clip, each decoded by the run's per-fold model that never
    trained on that clip's signer (models/<run>_<signer>.pt from train_ctc.py --save). Same outputs, so the
    only difference is the decoder. Writes results/<run>_decoding.md and .json.
    """
    import json
    import time
    import metrics as M
    from live_ctc import Recognizer
    from collect_sequences import load_word_list
    from eval_baseline import pct

    clips = D.load_clips()
    signers = D.signers(clips)
    recs = {s: Recognizer.load(f"models/{run}_{s}.pt") for s in signers}
    words = load_word_list(word_list)
    methods = ["greedy", f"beam (width {BEAM_WIDTH})", "word list (55 words)"]
    preds = {m: [] for m in methods}
    start = time.time()
    for c in clips:
        lp = recs[c.signer].log_probs(c)
        preds[methods[0]].append(__import__("model_ctc").greedy_decode(torch.from_numpy(lp)))
        preds[methods[1]].append(beam_search(lp)[0][0])
        preds[methods[2]].append(lexicon_decode(lp, words)[0][0])
    res = {"run": run, "n_clips": len(clips), "signers": signers, "beam_width": BEAM_WIDTH, "scores": {}}
    for m in methods:
        res["scores"][m] = {}
        for s in signers + ["all"]:
            idx = [i for i, c in enumerate(clips) if s == "all" or c.signer == s]
            r = M.score([clips[i].word for i in idx], [preds[m][i] for i in idx])
            res["scores"][m][s] = {k: r[k] for k in ("cer", "word_acc", "letter_recall", "n")}
    changed = [(c.signer, c.word, g, b) for c, g, b in zip(clips, preds[methods[0]], preds[methods[1]]) if g != b]
    res["beam_changed"] = changed

    L = [f"# Decoding comparison: greedy vs beam search vs word list ({run})", "",
         f"{len(clips)} clips. Every clip is decoded by the model that never trained on its signer "
         f"(`models/{run}_<signer>.pt`), so all rows are held-out-signer scores. Same model outputs; only the decoder differs.", "",
         f"- **Beam search**: width {BEAM_WIDTH}, fixed before any results.",
         "- **Word list**: picks the list word the clip most likely spells. It can only output our 55 words, all of which "
         "the model trained on, so it **flatters** the model and can never spell a new word. Report it next to greedy/beam, "
         "never instead of them.", "",
         "| Decoder | Signer | CER | Word acc | Letter recall |", "|---|---|---|---|---|"]
    for m in methods:
        for s in signers + ["all"]:
            r = res["scores"][m][s]
            L.append(f"| {m} | {s} | {pct(r['cer'])} | {pct(r['word_acc'])} | {pct(r['letter_recall'])} |")
    L += ["", f"## Clips where beam search changed the greedy answer ({len(changed)})", "",
          "| Signer | Word | Greedy | Beam |", "|---|---|---|---|"]
    L += [f"| {s} | {w} | {g or '∅'} | {b or '∅'} |" for s, w, g, b in changed]
    with open(f"results/{run}_decoding.json", "w") as f:
        json.dump(res, f, indent=2)
    with open(f"results/{run}_decoding.md", "w") as f:
        f.write("\n".join(L) + "\n")
    for m in methods:
        r = res["scores"][m]["all"]
        print(f"{m:<22} CER {pct(r['cer']):>6}  word acc {pct(r['word_acc']):>6}  letter recall {pct(r['letter_recall']):>6}")
    print(f"Beam changed {len(changed)} of {len(clips)} greedy answers. "
          f"Wrote results/{run}_decoding.md ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Compare greedy, beam search and word-list decoding on held-out signers.")
    p.add_argument("--run", required=True, help="a run saved with train_ctc.py --save --name <run>")
    compare(p.parse_args().run)
