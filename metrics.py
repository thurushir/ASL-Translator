"""
Scores predicted words against the true words. Every model is scored with this file, so results are comparable.
Argument order everywhere is (truth, prediction).

  edit_distance("HELLO", "HELO") -> 1     fewest letter adds / removes / swaps to fix the prediction
  cer("HELLO", "HELO")           -> 0.2   character error rate = edits / letters in the true word
  align("SAT", "AAT")            -> [("S","A"), ("A","A"), ("T","T")]   None marks a missing or extra letter
  score(truths, preds)           -> {"cer", "word_acc", "n", "confusion"}
"""
from collections import Counter


def _table(truth, pred):
    """Edit-distance table: d[i][j] = edits to turn pred[:j] into truth[:i]."""
    d = [[0] * (len(pred) + 1) for _ in range(len(truth) + 1)]
    for i in range(len(truth) + 1):
        d[i][0] = i
    for j in range(len(pred) + 1):
        d[0][j] = j
    for i in range(1, len(truth) + 1):
        for j in range(1, len(pred) + 1):
            swap = 0 if truth[i - 1] == pred[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + swap,  # match or swap
                          d[i - 1][j] + 1,         # letter missing from the prediction
                          d[i][j - 1] + 1)         # extra letter in the prediction
    return d


def edit_distance(truth, pred):
    return _table(truth, pred)[-1][-1]


def cer(truth, pred):
    return edit_distance(truth, pred) / max(len(truth), 1)


def align(truth, pred):
    """Pairs each true letter with the predicted letter it became (None = missing or extra letter)."""
    d = _table(truth, pred)
    i, j, pairs = len(truth), len(pred), []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (truth[i - 1] != pred[j - 1]):
            pairs.append((truth[i - 1], pred[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            pairs.append((truth[i - 1], None))
            i -= 1
        else:
            pairs.append((None, pred[j - 1]))
            j -= 1
    return pairs[::-1]


def errors(truth, pred):
    """Only the mistakes from align(): swaps, missing letters and extra letters."""
    return [(t, p) for t, p in align(truth, pred) if t != p]


def score(truths, preds):
    """
    Scores a batch of words.
      cer       total edits / total true letters (longer words count more, the standard way)
      word_acc  fraction of words predicted exactly right
      letter_recall  fraction of true letters that were predicted correctly (extra letters don't count against it)
      confusion Counter of (true letter, predicted letter) over all aligned pairs, correct ones included;
                (letter, None) = that letter was dropped, (None, letter) = an extra letter was added
    """
    if len(truths) != len(preds):
        raise ValueError(f"{len(truths)} true words but {len(preds)} predictions")
    confusion = Counter()
    for t, p in zip(truths, preds):
        confusion.update(align(t, p))
    total_letters = sum(len(t) for t in truths)
    return {
        "cer": sum(edit_distance(t, p) for t, p in zip(truths, preds)) / max(total_letters, 1),
        "word_acc": sum(t == p for t, p in zip(truths, preds)) / max(len(truths), 1),
        "letter_recall": sum(n for (t, p), n in confusion.items() if t is not None and t == p) / max(total_letters, 1),
        "n": len(truths),
        "confusion": confusion,
    }


def letter_report(confusion):
    """Per true letter: how often it was right, and what it was most often mistaken for."""
    rows = {}
    for letter in sorted({t for t, _ in confusion if t is not None}):
        seen = {p: n for (t, p), n in confusion.items() if t == letter}
        total = sum(seen.values())
        wrong = sorted(((p if p is not None else "(dropped)", n) for p, n in seen.items() if p != letter),
                       key=lambda pn: -pn[1])  # most common first; sort on the count only
        rows[letter] = {"total": total, "correct": seen.get(letter, 0) / total, "mistaken_for": wrong[:3]}
    return rows
