"""
Tests for decode.py (beam search, word-list decoding, confidence) using hand-made model outputs.
No trained model or real data needed. Run: python test_decode.py
"""
import numpy as np
import torch
import dataset as D
import model_ctc as MC
import decode as DE


def check(name, got, want):
    print(f"{'ok  ' if got == want else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return got == want


def frames(*rows):
    """Each row: {symbol: probability} for one frame ('-' = blank); the rest is spread evenly. -> (T, 27) log-probs."""
    out = []
    for row in rows:
        p = np.full(D.NUM_CLASSES, 0.0)
        for sym, prob in row.items():
            p[D.BLANK if sym == "-" else D.encode(sym)[0]] = prob
        rest = 1 - p.sum()
        p[p == 0] = rest / (p == 0).sum()
        out.append(np.log(np.maximum(p, 1e-12)))
    return np.array(out)


def sharp(seq):
    """One near-certain symbol per frame, e.g. 'HEL-LO'."""
    return frames(*[{ch: 0.97} for ch in seq])


def greedy(lp):
    return MC.greedy_decode(torch.from_numpy(lp))


results = []

# --- beam search ---
results.append(check("clear output: beam gives the obvious word", DE.beam_search(sharp("HHEEL-LLO"))[0][0], "HELLO"))
results.append(check("a blank between two Ls gives a double letter", DE.beam_search(sharp("L-L"))[0][0], "LL"))
results.append(check("two Ls in a row with no blank give one L", DE.beam_search(sharp("LL"))[0][0], "L"))
for seq in ["HHEEL-LLO", "C-AAT", "--JAZ-Z-", "MY"]:
    results.append(check(f"clear output {seq!r}: width 1 equals greedy", DE.beam_search(sharp(seq), width=1)[0][0], greedy(sharp(seq))))

# Greedy follows the best frame each time (blank, blank -> nothing). Beam adds up every way of spelling "A":
# A A, A -, - A = 0.4*0.4 + 0.4*0.6 + 0.6*0.4 = 0.64, more than blank blank = 0.36.
tricky = frames({"-": 0.6, "A": 0.4}, {"-": 0.6, "A": 0.4})
results.append(check("greedy picks the best single path", greedy(tricky), ""))
results.append(check("beam picks the most likely word", DE.beam_search(tricky)[0][0], "A"))
cands = DE.beam_search(tricky)
results.append(check("beam probability of A is the sum over its paths (0.64)", round(float(np.exp(cands[0][1])), 3), 0.64))
results.append(check("candidates come out best first", [s for _, s in cands] == sorted([s for _, s in cands], reverse=True), True))
results.append(check("at most `width` candidates", len(DE.beam_search(sharp("CAT"), width=3)) <= 3, True))
results.append(check("no candidate is repeated", len({w for w, _ in cands}) == len(cands), True))

# --- word-list decoding: the list word the clip most likely spells (CTC probability of each word) ---
words = ["HELLO", "HELP", "YELLOW", "CAT"]
results.append(check("word list: HELLO from a clear HELLO", DE.lexicon_decode(sharp("HHEEL-LLO"), words)[0][0], "HELLO"))
unsure_ll = frames({"H": .97}, {"E": .97}, {"L": .97}, {"L": .6, "-": .4}, {"L": .97}, {"O": .97})
results.append(check("an unsure blank between the Ls: greedy merges them", greedy(unsure_ll), "HELO"))
results.append(check("word list: the same output maps to HELLO (not HELP)",
                     DE.lexicon_decode(unsure_ll, words)[0][0], "HELLO"))
results.append(check("word list: ranked best first", [w for w, _ in DE.lexicon_decode(sharp("CAAT"), words)][0], "CAT"))
results.append(check("word list: every list word gets a score", len(DE.lexicon_decode(sharp("CAT"), words)), 4))
results.append(check("word list: matches torch's CTC loss for one word",
                     round(DE.lexicon_decode(tricky, ["A"])[0][1], 5),
                     round(-torch.nn.functional.ctc_loss(torch.from_numpy(tricky)[:, None, :], torch.tensor([[1]]),
                                                         torch.tensor([2]), torch.tensor([1]), reduction="sum").item(), 5)))

# --- confidence for the live overlay ---
sure = DE.confidence(DE.beam_search(sharp("CCAATT")))
unsure = DE.confidence(DE.beam_search(frames(*[{"C": 0.35, "O": 0.35}] * 3)))
results.append(check("a sharp output gives high confidence (> 0.9)", sure[0][1] > 0.9, True))
results.append(check("a flat output gives low confidence (< 0.6)", unsure[0][1] < 0.6, True))
results.append(check("confidences add up to at most 1", sum(c for _, c in unsure) <= 1 + 1e-9, True))
results.append(check("confidences come out best first", [c for _, c in unsure] == sorted([c for _, c in unsure], reverse=True), True))

print("\nPASS: all decode tests passed" if all(results) else "\nFAIL: some decode tests failed")
