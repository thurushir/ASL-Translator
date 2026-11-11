"""
Trains the LSTM + CTC model (model_ctc.py) on the fingerspelling clips and scores it like the baseline.

For each held-out signer (dataset.signer_splits): train on the other two signers, stop early using a
validation set taken from those two signers only, then score the held-out signer with metrics.py
(through eval_baseline.summarize, so the columns match results/baseline.md exactly).

  python train_ctc.py               -> all 3 folds, writes results/lstm_ctc.md and results/lstm_ctc.json
  python train_ctc.py --overfit 10  -> first check: trains on 10 clips and should spell them almost perfectly
  python train_ctc.py --save --name step5_lstm
                                    -> also saves each fold's model to models/step5_lstm_<signer>.pt
  python train_ctc.py --motion --name step5_motion
                                    -> adds wrist and fingertip motion to the inputs (motion.py)
  python train_ctc.py --speed-aug --name step5_speed
                                    -> trains on each clip at a random 0.7x-1.4x speed every epoch
  python train_ctc.py --model transformer --speed-aug --name step5_tf
                                    -> a Transformer encoder instead of the LSTM (same inputs, CTC and decoder)
  python train_ctc.py --final --speed-aug --name step6
                                    -> one model trained on ALL signers -> models/step6_final.pt (Step 6: live app,
                                       outside-signer test). No held-out score: the outside signers are its test.
Every run stores each held-out clip's prediction and inference time in the results JSON ("predictions");
python evaluate.py <name> turns them into the proposal's evaluation (by word length, speed, confusions, timing).
Tests: python test_train_ctc.py
"""
import argparse
import copy
import datetime
import json
import os
import time
import numpy as np
import torch
from torch import nn
import dataset as D
import metrics as M
import model_ctc as MC
import evaluate as E
from eval_baseline import summarize, pct, has_motion_letter

RESULTS_DIR = "./results"
MODELS_DIR = "./models"  # ignored by git: model files are large and can be rebuilt with --save
CTC_LOSS = nn.CTCLoss(blank=D.BLANK, zero_infinity=True)  # zero_infinity: an impossible clip adds 0, not inf


# ---------- data ----------

def split_validation(clips, frac=0.15, seed=0):
    """(fit, val): a fixed random frac of clips for deciding when to stop. Pass training-signer clips only."""
    order = np.random.default_rng(seed).permutation(len(clips))
    n_val = int(round(frac * len(clips)))
    return [clips[i] for i in order[n_val:]], [clips[i] for i in order[:n_val]]


def final_split(clips, frac=0.15, seed=0):
    """(fit, val) for --final: frac of EACH signer's clips for validation, so every signer is in it."""
    fit, val = [], []
    for s in D.signers(clips):
        f, v = split_validation([c for c in clips if c.signer == s], frac, seed)
        fit, val = fit + f, val + v
    return fit, val


def split_words(clips, words):
    """(kept, held): held = clips of the given words. Used to leave words out of training entirely."""
    return [c for c in clips if c.word not in words], [c for c in clips if c.word in words]


def make_batch(clips, mean, std):
    """Clips -> padded inputs (B, T, 65), lengths (B,), joined letter targets, and word lengths."""
    xs = [torch.from_numpy(MC.clip_inputs(c, mean, std)) for c in clips]
    x = nn.utils.rnn.pad_sequence(xs, batch_first=True)  # pads with zeros
    lengths = torch.tensor([len(v) for v in xs])
    targets = torch.tensor([k for c in clips for k in c.labels])
    target_lengths = torch.tensor([len(c.labels) for c in clips])
    return x, lengths, targets, target_lengths


def augment(clips, rng, lo, hi):
    """Training clips at random speeds between lo and hi (speed augmentation, Step 5). Never used on val/test."""
    return [D.change_speed(c, rng.uniform(lo, hi)) for c in clips]


def junk_start(truth, pred):
    """True if the prediction has an extra letter before the word starts (e.g. anna's resting hand -> 'BHI')."""
    pairs = M.align(truth, pred)
    return bool(pairs) and pairs[0][0] is None


# ---------- training ----------

def ctc_loss(model, x, lengths, targets, target_lengths):
    log_probs = model(x, lengths).transpose(0, 1)  # CTCLoss wants (frames, batch, classes)
    return CTC_LOSS(log_probs, targets, lengths, target_lengths)


def train_step(model, opt, x, lengths, targets, target_lengths):
    model.train()
    opt.zero_grad()
    loss = ctc_loss(model, x, lengths, targets, target_lengths)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # LSTMs can have sudden huge gradients; cap them
    opt.step()
    return loss.item()


@torch.no_grad()
def predict(model, clips, stats, batch_size=64):
    """Words predicted for clips (greedy decoding)."""
    model.eval()
    words = []
    for i in range(0, len(clips), batch_size):
        x, lengths, _, _ = make_batch(clips[i:i + batch_size], *stats)
        words += MC.decode_batch(model(x, lengths), lengths)
    return words


@torch.no_grad()
def time_inference(model, clips, stats):
    """
    (words, ms): each clip predicted on its own, as the live app would, timing input preparation + model +
    decoding in milliseconds. The first clip is run once untimed first (PyTorch is slow on its very first call).
    """
    model.eval()
    if clips:
        predict(model, clips[:1], stats)  # warm-up
    words, ms = [], []
    for c in clips:
        start = time.perf_counter()
        words += predict(model, [c], stats)
        ms.append((time.perf_counter() - start) * 1000)
    return words, ms


@torch.no_grad()
def evaluate(model, clips, stats):
    """(CER, mean CTC loss) on clips."""
    model.eval()
    x, lengths, targets, target_lengths = make_batch(clips, *stats)
    loss = ctc_loss(model, x, lengths, targets, target_lengths).item()
    return M.score([c.word for c in clips], predict(model, clips, stats))["cer"], loss


def train(fit, val, cfg, log=print):
    """
    Trains on fit; after every epoch scores val and keeps the best model.
    Better = lower val CER, or the same CER with lower val loss (early on CTC outputs only blanks, so CER
    stays at 100% for a while even though the model is learning; the loss still shows that progress).
    Stops after cfg.patience epochs without improvement.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    aug_rng = np.random.default_rng(cfg.seed + 1)  # separate, so the clip order matches a run without augmentation
    speed_aug = getattr(cfg, "speed_aug", False)
    stats = MC.feature_stats(fit, motion=getattr(cfg, "motion", False))  # standardize with the fitting clips only
    model = MC.make_model(getattr(cfg, "model", "lstm"), len(stats[0]) + 1, hidden=cfg.hidden, layers=cfg.layers,
                          dropout=cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best, best_key, best_epoch, history = None, None, 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        order = rng.permutation(len(fit))
        batches = [[fit[i] for i in order[b:b + cfg.batch]] for b in range(0, len(fit), cfg.batch)]
        if speed_aug:
            batches = [augment(bc, aug_rng, cfg.speed_min, cfg.speed_max) for bc in batches]
        losses = [train_step(model, opt, *make_batch(bc, *stats)) for bc in batches]
        val_cer, val_loss = evaluate(model, val, stats)
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
                        "val_loss": round(val_loss, 4), "val_cer": round(val_cer, 4)})
        if best_key is None or (val_cer, val_loss) < best_key:
            best, best_key, best_epoch = copy.deepcopy(model.state_dict()), (val_cer, val_loss), epoch
        if epoch % 10 == 0 or epoch == 1:
            log(f"    epoch {epoch:>3}  train loss {np.mean(losses):.3f}  val loss {val_loss:.3f}  val CER {pct(val_cer)}")
        if epoch - best_epoch >= cfg.patience:
            break
    model.load_state_dict(best)
    return model, stats, {"params": MC.count_params(model), "best_epoch": best_epoch, "epochs_run": epoch, "best_val_cer": best_key[0],
                          "history": history}


# ---------- runs ----------

def overfit_check(clips, cfg):
    """Trains and scores on the same few clips, no dropout. A working loop should spell them ~perfectly."""
    picked = [clips[i] for i in np.linspace(0, len(clips) - 1, cfg.overfit).astype(int)]  # spread across words
    cfg = copy.copy(cfg)
    cfg.dropout, cfg.patience, cfg.speed_aug = 0.0, cfg.max_epochs, False
    print(f"Overfit check: training and scoring on the same {len(picked)} clips (no dropout)")
    model, stats, info = train(picked, picked, cfg)
    preds = predict(model, picked, stats)
    for c, p in zip(picked, preds):
        print(f"  {c.word:<10} -> {p:<10} {'ok' if p == c.word else 'WRONG'}   ({c.signer})")
    s = M.score([c.word for c in picked], preds)
    print(f"Overfit check: CER {pct(s['cer'])}, word accuracy {pct(s['word_acc'])} "
          f"(best epoch {info['best_epoch']} of {info['epochs_run']})")


def brief(clips, preds):
    """summarize() without the per-letter table."""
    return {k: v for k, v in summarize(clips, preds).items() if k != "letters"}


def run_folds(clips, cfg):
    signers = D.signers(clips)
    preds = [None] * len(clips)
    index = {c.path: i for i, c in enumerate(clips)}
    held_words = set(cfg.holdout_words)
    new_words = {"held_out_signer_new_words": ([], []), "held_out_signer_seen_words": ([], []),
                 "training_signers_new_words": ([], [])}  # group -> (clips, preds), all folds together
    folds, records = {}, []
    for fold in D.signer_splits(clips):
        start = time.time()
        train_pool, train_new = split_words(fold.train, held_words)  # held-out words never trained on
        fit, val = split_validation(train_pool, cfg.val_frac, cfg.seed)
        print(f"\nFold: test on {fold.test_signer} ({len(fold.test)} clips); "
              f"train on {len(fit)}, validate on {len(val)} from the other signers"
              + (f"; {len(held_words)} words left out of training" if held_words else ""))
        model, stats, info = train(fit, val, cfg)
        test_preds, test_ms = time_inference(model, fold.test, stats)  # same words as predict(), plus timing
        for c, p in zip(fold.test, test_preds):
            preds[index[c.path]] = p
        records += E.prediction_records(fold.test, test_preds, test_ms)
        if cfg.save:
            os.makedirs(MODELS_DIR, exist_ok=True)
            info["model_file"] = os.path.join(MODELS_DIR, f"{cfg.name}_{fold.test_signer}.pt")
            MC.save_model(info["model_file"], model, stats, model_config(cfg, stats),
                          test_signer=fold.test_signer, seed=cfg.seed,
                          train_signers=[s for s in signers if s != fold.test_signer])
        train_scores = summarize(train_pool, predict(model, train_pool, stats))
        test_scores = summarize(fold.test, test_preds)
        if held_words:
            by_path = dict(zip((c.path for c in fold.test), test_preds))
            seen, new = split_words(fold.test, held_words)
            groups = {"held_out_signer_new_words": (new, [by_path[c.path] for c in new]),
                      "held_out_signer_seen_words": (seen, [by_path[c.path] for c in seen]),
                      "training_signers_new_words": (train_new, predict(model, train_new, stats))}
            info["new_words"] = {g: brief(*cp) for g, cp in groups.items()}
            for g, (cs, ps) in groups.items():
                new_words[g][0].extend(cs)
                new_words[g][1].extend(ps)
            print("  new-words check: " + "   ".join(
                f"{g.replace('_', ' ')}: CER {pct(r['cer'])}, word acc {pct(r['word_acc'])}"
                for g, r in info["new_words"].items()))
        info.update(seconds=round(time.time() - start), train_signers=[s for s in signers if s != fold.test_signer],
                    train_scores={k: v for k, v in train_scores.items() if k != "letters"},
                    junk_start=sum(junk_start(c.word, p) for c, p in zip(fold.test, test_preds)) / len(fold.test))
        folds[fold.test_signer] = info
        print(f"  best epoch {info['best_epoch']} of {info['epochs_run']} ({info['seconds']}s)   "
              f"training signers: CER {pct(train_scores['cer'])}, word acc {pct(train_scores['word_acc'])}   "
              f"held-out {fold.test_signer}: CER {pct(test_scores['cer'])}, word acc {pct(test_scores['word_acc'])}")
    scores = {s: summarize([c for c in clips if c.signer == s], [p for c, p in zip(clips, preds) if c.signer == s])
              for s in signers}
    scores["all"] = summarize(clips, preds)
    new_words = {g: brief(cs, ps) for g, (cs, ps) in new_words.items()} if held_words else None
    return preds, scores, folds, new_words, records


def model_config(cfg, stats):
    """The settings saved with a model, so load_model() can rebuild it."""
    return {"model": cfg.model, "hidden": cfg.hidden, "layers": cfg.layers, "dropout": cfg.dropout, "fps": cfg.fps,
            "input_size": len(stats[0]) + 1, "motion": cfg.motion}


def run_final(clips, cfg, log=print):
    """
    --final (Step 6): one model trained on every signer, for the live app and the outside-signer test.
    There is no held-out signer, so no fair score here: validation (frac of each signer) only decides when to
    stop, and the training-clip scores below only show it learned. Outside signers are this model's test.
    Always saves to models/<name>_final.pt and writes results/<name>_final.json.
    """
    start = time.time()
    signers = D.signers(clips)
    fit, val = final_split(clips, cfg.val_frac, cfg.seed)
    log(f"Final model: train on {len(fit)} clips, validate on {len(val)} (all of {', '.join(signers)})")
    model, stats, info = train(fit, val, cfg, log=log)
    os.makedirs(MODELS_DIR, exist_ok=True)
    info["model_file"] = os.path.join(MODELS_DIR, f"{cfg.name}_final.pt")
    MC.save_model(info["model_file"], model, stats, model_config(cfg, stats),
                  test_signer=None, seed=cfg.seed, train_signers=signers)
    info["predictions"] = predict(model, clips, stats)
    fit_scores = M.score([c.word for c in fit], predict(model, fit, stats))
    info.update(seconds=round(time.time() - start), train_signers=signers,
                fit_cer=fit_scores["cer"], fit_word_acc=fit_scores["word_acc"])
    res = {"date": datetime.date.today().isoformat(), "n_clips": len(clips), "signers": signers,
           "config": vars(cfg), "note": "No held-out score: trained on every signer. Outside signers are its test.",
           "final": {k: v for k, v in info.items() if k not in ("history", "predictions")}, "history": info["history"]}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{cfg.name}_final.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    log(f"Saved {info['model_file']}  (best epoch {info['best_epoch']} of {info['epochs_run']}, "
        f"validation CER {pct(info['best_val_cer'])}, training-clip CER {pct(info['fit_cer'])}; {info['seconds']}s)")
    return info


def model_label(config):
    return "Transformer+CTC" if config.get("model") == "transformer" else "LSTM+CTC"


def model_description(config):
    if config.get("model") == "transformer":
        return (f"{config['layers']}-layer Transformer encoder ({config['hidden']} wide, 4 heads, "
                f"dropout {config['dropout']}, learning rate {config['lr']})")
    return f"{config['layers']}-layer bidirectional LSTM ({config['hidden']} units)"


def write_markdown(res, path):
    base = res["baseline_b"]
    label = model_label(res["config"])
    L = [f"# {label.replace('+', ' + ')} on fingerspelling clips", "",
         f"Generated by `train_ctc.py` on {res['date']}. {res['n_clips']} clips, signers: {', '.join(res['signers'])}.", "",
         "- Every row is a **held-out signer**: the model never trained on that person's clips.",
         "- Inputs per frame: 64 hand-shape features + hand-present flag"
         + (" + 14 motion features (wrist position and speed, fingertip speeds; motion.py)." if res["config"].get("motion") else "."),
         (f"- Speed augmentation: each training clip played at a random {res['config']['speed_min']}x to "
          f"{res['config']['speed_max']}x speed every epoch (never the validation or test clips)."
          if res["config"].get("speed_aug") else "- No speed augmentation."),
         f"- Clips resampled to {res['config']['fps']} fps. Model: {model_description(res['config'])}"
         + (f" ({n:,} parameters)" if (n := next(iter(res["folds"].values())).get("params")) else "")
         + ", greedy CTC decoding.",
         "- Early stopping on a validation set taken from the two training signers only.",
         "- *No J/Z* columns leave out words with J or Z (comparable with the baseline's columns).",
         "- Baseline B rows are copied from `results/baseline.json` (the number to beat).", "",
         "## Scores", "",
         "| Model | Signer | CER | Word acc | Letter recall | CER (no J/Z) | Word acc (no J/Z) |",
         "|---|---|---|---|---|---|---|"]
    for name, table in [("Baseline B", base), (label, res["scores"])]:
        for s in res["signers"] + ["all"]:
            r = table[s]
            L.append(f"| {name} | {s} | {pct(r['cer'])} | {pct(r['word_acc'])} | {pct(r['letter_recall'])} "
                     f"| {pct(r['cer_no_jz'])} | {pct(r['word_acc_no_jz'])} |")
    L += ["", "## Memorization check: training signers vs held-out signer", "",
          "A big gap between the two means the model memorizes instead of generalizing.", "",
          "| Held-out signer | Trained on | CER on training signers | CER on held-out | Word acc on training signers "
          "| Word acc on held-out | Best epoch / run | Junk first letter |",
          "|---|---|---|---|---|---|---|---|"]
    for s in res["signers"]:
        f, t = res["folds"][s], res["scores"][s]
        L.append(f"| {s} | {', '.join(f['train_signers'])} | {pct(f['train_scores']['cer'])} | {pct(t['cer'])} "
                 f"| {pct(f['train_scores']['word_acc'])} | {pct(t['word_acc'])} | {f['best_epoch']} / {f['epochs_run']} "
                 f"| {pct(f['junk_start'])} |")
    if res.get("new_words"):
        L += ["", "## New-words check", "",
              f"These words were left out of training entirely: {', '.join(res['config']['holdout_words'])}. "
              "Every letter in them still appears in the training words. Good scores on new signers but poor scores "
              "on new words means the model memorizes whole words instead of learning letters. "
              "(The Scores table above includes the left-out words.)", "",
              "| Group (all folds together) | Clips | CER | Word acc | Letter recall |", "|---|---|---|---|---|"]
        names = {"held_out_signer_seen_words": "Held-out signer, words seen in training",
                 "held_out_signer_new_words": "Held-out signer, new words",
                 "training_signers_new_words": "Training signers, new words"}
        for g, name in names.items():
            r = res["new_words"][g]
            L.append(f"| {name} | {r['n']} | {pct(r['cer'])} | {pct(r['word_acc'])} | {pct(r['letter_recall'])} |")
    L += ["", "## Per-letter results (held-out signer)", "",
          "How often each true letter came out correctly, and what it was most often turned into.", "",
          "| Letter | " + " | ".join(res["signers"]) + " |", "|---|" + "---|" * len(res["signers"])]
    for letter in D.ALPHABET:
        cells = []
        for s in res["signers"]:
            r = res["scores"][s]["letters"].get(letter)
            top = ", ".join(f"{p}×{n}" for p, n in r["mistaken_for"][:2]) if r else ""
            cells.append("-" if r is None else pct(r["correct"]) + (f" ({top})" if top else ""))
        L.append(f"| {letter} | " + " | ".join(cells) + " |")
    L += ["", "## Example predictions (first take of each word for each signer)", "",
          "| Word | " + " | ".join(res["signers"]) + " |", "|---|" + "---|" * len(res["signers"])]
    for word, row in res["examples"].items():
        L.append(f"| {word} | " + " | ".join(row.get(s, "") or "∅" for s in res["signers"]) + " |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fps", type=float, default=15)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--model", choices=["lstm", "transformer"], default="lstm",
                   help="lstm (Step 4) or transformer (Step 5 comparison)")
    p.add_argument("--layers", type=int, default=None, help="default: 2 for lstm, 3 for transformer")
    p.add_argument("--dropout", type=float, default=None, help="default: 0.3 for lstm, 0.1 for transformer")
    p.add_argument("--lr", type=float, default=None, help="default: 0.001 for lstm, 0.0005 for transformer")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overfit", type=int, default=0, help="train and score on this many clips (loop check)")
    p.add_argument("--holdout-words", default="",
                   help="comma-separated words to leave out of training (new-words check), e.g. CAT,DOG")
    p.add_argument("--name", default="lstm_ctc", help="results file name (results/<name>.md and .json)")
    p.add_argument("--speed-aug", action="store_true",
                   help="speed augmentation: play each training clip at a random speed every epoch (Step 5)")
    p.add_argument("--speed-min", type=float, default=0.7, help="slowest speed for --speed-aug (default 0.7x)")
    p.add_argument("--speed-max", type=float, default=1.4, help="fastest speed for --speed-aug (default 1.4x)")
    p.add_argument("--motion", action="store_true",
                   help="add the 14 motion features from motion.py to every frame (Step 5)")
    p.add_argument("--save", action="store_true",
                   help="also save each fold's trained model to models/<name>_<signer>.pt")
    p.add_argument("--final", action="store_true",
                   help="train one model on ALL signers and save models/<name>_final.pt (Step 6; no held-out score)")
    cfg = p.parse_args(argv)
    if cfg.final and (cfg.holdout_words or cfg.overfit):
        p.error("--final trains on everything: it can't be combined with --holdout-words or --overfit")
    # Each model's usual settings, fixed before any results (never tuned on the held-out signer).
    defaults = {"lstm": {"layers": 2, "dropout": 0.3, "lr": 1e-3},
                "transformer": {"layers": 3, "dropout": 0.1, "lr": 5e-4}}[cfg.model]
    for key, value in defaults.items():
        if getattr(cfg, key) is None:
            setattr(cfg, key, value)
    cfg.holdout_words = [w.strip().upper() for w in cfg.holdout_words.split(",") if w.strip()]
    return cfg


def main():
    cfg = parse_args()
    start = time.time()
    clips = [D.resample(c, cfg.fps) for c in D.load_clips()]
    print(f"Loaded {len(clips)} clips, resampled to {cfg.fps:g} fps")
    if cfg.overfit:
        overfit_check(clips, cfg)
        return
    if cfg.final:
        run_final(clips, cfg)
        return

    unknown = set(cfg.holdout_words) - {c.word for c in clips}
    if unknown:
        raise SystemExit(f"--holdout-words not in the data: {sorted(unknown)}")
    preds, scores, folds, new_words, records = run_folds(clips, cfg)
    with open(os.path.join(RESULTS_DIR, "baseline.json")) as f:
        baseline_b = json.load(f)["methods"]["B"]
    res = {"date": datetime.date.today().isoformat(), "n_clips": len(clips), "signers": D.signers(clips),
           "config": vars(cfg), "scores": scores, "folds": folds, "new_words": new_words,
           "baseline_b": {s: {k: v for k, v in r.items() if k != "letters"} for s, r in baseline_b.items()},
           "examples": {}, "predictions": records}
    for c, pr in zip(clips, preds):
        if c.path.endswith("_0.npz"):
            res["examples"].setdefault(c.word, {})[c.signer] = pr
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{cfg.name}.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    write_markdown(res, os.path.join(RESULTS_DIR, f"{cfg.name}.md"))

    print(f"\n{'Model':<12}{'Signer':<10}{'CER':>8}{'WordAcc':>9}{'Recall':>8}{'CER-noJZ':>10}")
    for name, table in [("Baseline B", res["baseline_b"]), (model_label(res["config"]), scores)]:
        for s in res["signers"] + ["all"]:
            r = table[s]
            print(f"{name:<12}{s:<10}{pct(r['cer']):>8}{pct(r['word_acc']):>9}{pct(r['letter_recall']):>8}{pct(r['cer_no_jz']):>10}")
    jz = [i for i, c in enumerate(clips) if has_motion_letter(c.word)]
    jz_score = M.score([clips[i].word for i in jz], [preds[i] for i in jz])
    print(f"\nWords with J/Z only: CER {pct(jz_score['cer'])}, word acc {pct(jz_score['word_acc'])}")
    print(f"Wrote results/{cfg.name}.md and .json ({time.time() - start:.0f}s)")
    if cfg.save:
        print(f"Saved models: {', '.join(f['model_file'] for f in folds.values())}")
    print(f"Next: python evaluate.py {cfg.name}")


if __name__ == "__main__":
    main()
