# Temporal (continuous fingerspelling) work — `temporalwip` branch

The `main` branch stays as the static-letter version. This branch moves toward recognizing continuous fingerspelling.

## Step 1: setup (done)

| File | What it does |
|---|---|
| `features.py` | **The only place features are computed.** `frame_features()` gives 64 numbers for one frame; `sequence_features()` does the same for a whole clip. |
| `check_features.py` | Confirms `features.py` matches the original feature code. |
| `hand_utils.py` | `init_hands()` now actually uses its settings; `start_video()` can pass key presses to a script. |
| `train_model.py` | Also saves `baseline_results.json` (the static model's accuracy, our reference). |
| `collect_sequences.py` | Records clips of fingerspelled words as per-frame landmarks to `data_seq/<WORD>/<signer>_<n>.npz`. |
| `check_sequences.py` | Summarizes recorded clips and flags bad ones. |
| `transcript.py` | Turns per-frame predictions into text that stays on screen (rules are at the top of the file). |
| `test_transcript.py` | Tests the transcript rules without a webcam. |

### Clip format (`.npz`)
- `landmarks`: (frames, 21, 3) raw x, y, z; rows are NaN when no hand was found
- `timestamps`: seconds since the clip started
- `word`, `signer`: labels; `mirrored`: True (frames are mirrored like the live view)

Raw landmarks are saved (not features) so the feature design can change later without re-recording.

### Live controls (`predict_live.py`)
Hold a letter steadily to add it. Relax your hand briefly before repeating a letter (the LL in HELLO).
Drop your hand for 1s for a space. Keys: `space` = space, `backspace` = delete, `c` = clear, `q` = quit.

## How to verify
```bash
python check_features.py        # PASS: features match
python test_transcript.py       # PASS: all transcript tests passed
python process_data.py && python train_model.py   # ~92-93% accuracy, writes baseline_results.json
python collect_sequences.py --signer yourname --word HELLO   # r = start/stop, q = quit
python check_sequences.py
python predict_live.py
```

## Step 2: recording the dataset

New: `word_list.txt` (55 words; every letter 5+ times, J/Z, 16 double-letter words),
session mode in `collect_sequences.py`, per-signer and letter coverage in `check_sequences.py`.
`predict_live.py` no longer needs `data.pickle` (just `model.pickle`).

### How to record (each person)
```bash
git pull
python collect_sequences.py --signer yourname --list word_list.txt
```
- Keys: `r` record/stop, `n` next word, `b` previous word, `d` delete your last take, `q` quit.
- Target: 3 takes per word (~165 clips, ~20 min). Quit anytime; it resumes at your first unfinished word.
- Use the same `--signer` name every time (lowercase, e.g. `thurushi`).

### Tips
- Start and end each clip with your hand in view and relaxed.
- Sign at natural speed; do double letters the way you normally would (bounce or slide).
- Keep your whole hand in frame, about arm's length from the camera.
- Vary room, lighting and distance a little across sittings.
- Messed up a take? Press `d` and redo it.

### After each sitting
```bash
python check_sequences.py --list word_list.txt
```
Re-record any flagged clips, then copy your clips into the shared `data_seq/` folder.
File names include the signer, so everyone's clips can live in the same folders.
Old test clips named `<n>.npz` show up as signer `unknown`; delete them.

## Step 3: measuring (done)

All three signers' clips are recorded: 495 clips (anna, saaya, thurushi × 55 words × 3 takes).
They live in the shared Drive folder, not in git. Copy them into `data_seq/` (merge, don't replace).

| File | What it does |
|---|---|
| `dataset.py` | **Every model reads clips through this.** `load_clips()` gives per-frame features (T, 64), a hand mask, letter labels (A=1 … Z=26, 0 = CTC blank), signer and fps. `signer_splits()` gives 3 folds: train on two signers, test on the third. |
| `metrics.py` | **Every model is scored with this.** CER, word accuracy, letter recall, letter alignment (which letter turned into which) and per-letter reports. Arguments are always (truth, prediction). |
| `eval_baseline.py` | Scores the static model on the clips three ways (A/B/C below) and writes `results/baseline.md` and `results/baseline.json`. |
| `test_dataset.py`, `test_metrics.py`, `test_eval_baseline.py` | Tests for the above (no webcam or real data needed). |

### How to verify
```bash
python test_dataset.py          # PASS: all dataset tests passed
python test_metrics.py          # PASS: all metrics tests passed
python test_eval_baseline.py    # PASS: all eval_baseline tests passed
python dataset.py               # 495 clips, 3 folds of 330 train / 165 test
python eval_baseline.py         # ~10s, rewrites results/ (numbers should match exactly)
```

### Rules for fair comparison
- Always test on a signer the model did not train on (`signer_splits()`). Never split clips at random.
- Pick any settings (hyperparameters, thresholds) using the training signers only.
- Report words with J or Z separately; the static model can never produce those letters.
- The static model was trained on thurushi's hands, so her baseline rows are not an unseen signer.

### Baseline results (full tables in `results/baseline.md`)

| Method | CER | Word acc | Letter recall |
|---|---|---|---|
| A. As-is: the live app's rules (hold 8 frames at ≥ 50%) | 63.5% | 2.4% | 36.6% |
| **B. Tuned timing: hold/release in seconds, picked on 2 signers, tested on the 3rd** | **36.6%** | **18.4%** | 68.8% |
| C. Recognition only: merge repeated per-frame letters, no hold rule | 272.7% | 0.4% | 84.9% |

**B is the number to beat** (CER 36.6%, word accuracy 18.4%, all three held-out folds together).

What the baseline shows:
- **Timing, not recognition, is the main problem.** Somewhere in the clip the model sees 85% of the letters (C), but the best hand-tuned timing rule keeps only 69% (B).
- **J and Z: 0%.** J is read as I and Z as L or X (their starting hand shapes).
- **Double letters get merged** (BOOK → BOK, HELLO → HELO). Most "dropped" letters are the second O, E or L.
- **S is the worst static letter, for different reasons per signer.** thurushi and anna: mostly dropped or lost to timing (48% and 71% correct in C, 0% in B). saaya: read as M (shape problem). A thumb-position feature may help saaya's case later; timing is the temporal model's job.
- **anna's relaxed hand at the start of clips reads as "B"** (BHI, BMY, BORY), and her W is often read as B.
- **Frame rates differ:** anna ~29 fps (avg 133 frames/clip), saaya ~15 fps (71), thurushi ~15 fps (41; she also signs ~1.7× faster).

## Step 4: first LSTM + CTC model (done, Anna)

| File | What it does |
|---|---|
| `dataset.py` | New `resample(clip, fps=15)`: nearest original frame on a 1/15 s grid (anna's ~29 fps clips -> 15). `load_clips()` is unchanged. |
| `model_ctc.py` | `clip_inputs()` (64 standardized features + hand-present flag = 65 per frame), `CTCModel` (2-layer bidirectional LSTM, 128 units, -> 27 log-probs), `greedy_decode()`. |
| `train_ctc.py` | Trains one model per held-out signer; early stopping on 15% of the training signers' clips; writes `results/<name>.md/.json` with baseline B's rows alongside. `--seed` = random seed, `--overfit 10` = loop check, `--holdout-words` = new-words check. |
| `compare_runs.py` | Averages several runs (e.g. 3 seeds) into one table: each run's CER, mean and range. Writes `results/<out>.md`. |
| `test_model_ctc.py`, `test_train_ctc.py`, `test_compare_runs.py` | Tests (no webcam or real data needed). `test_dataset.py` gained resampling tests. |

### How to verify
```bash
pip install -r requirements.txt   # now includes torch
python test_dataset.py            # PASS: all dataset tests passed
python test_model_ctc.py          # PASS: all model_ctc tests passed
python test_train_ctc.py          # PASS: all train_ctc tests passed
python test_compare_runs.py       # PASS: all compare_runs tests passed
python train_ctc.py --overfit 10  # under a minute: CER 0.0%, word accuracy 100.0%
python train_ctc.py                                  # ~20 min on CPU: all signers CER 25.3% (seed 0)
python train_ctc.py --seed 1 --name lstm_ctc_seed1   # all signers CER 27.8%
python train_ctc.py --seed 2 --name lstm_ctc_seed2   # all signers CER 30.7%
python compare_runs.py lstm_ctc lstm_ctc_seed1 lstm_ctc_seed2 --out lstm_ctc_3seeds   # mean 27.9%
python train_ctc.py --holdout-words CAT,DOG,MILK,WATER,BOX,LOVE,GOOD,ZERO --name lstm_ctc_new_words
```

### Results (held-out signer, all three folds; mean of three training runs, seeds 0-2)

Full table: `results/lstm_ctc_3seeds.md`. Per-letter tables for each run: `results/lstm_ctc.md`, `lstm_ctc_seed1.md`, `lstm_ctc_seed2.md`.

| Model | CER | Word acc | Letter recall | anna CER | saaya CER | thurushi CER |
|---|---|---|---|---|---|---|
| Baseline B | 36.6% | 18.4% | 68.8% | 42.0% | 39.9% | 27.9% |
| **LSTM + CTC, mean of 3 runs** | **27.9%** | **31.4%** | 79.7% | 32.1% | 14.2% | 37.5% |
| LSTM + CTC, range over runs | 25.3-30.7% | 29.3-35.4% | 77.1-81.7% | 29.5-36.9% | 10.1-17.0% | 36.0-38.3% |

- **Beats baseline B overall in every run**, by a lot for anna and saaya. thurushi is worse than her B row in every run (her B row was never an unseen signer); her letters are mostly *dropped*, consistent with her signing ~1.7x faster than the two training signers.
- **Runs vary:** overall CER spans 5.4 points across seeds, one signer up to 7.4 (anna). **Compare Step 5 changes by their 3-seed mean** (`compare_runs.py`); smaller gains may be noise.
- **Memorization gap:** CER 0.4-2.5% on the training signers vs 10-38% on the held-out one.
- **New-words check** (8 words never trained on; seed 0): held-out signer CER 27.9% on seen words vs 37.8% on new words; training signers 13.0% on new words vs ~1% on seen ones. The model partly memorizes whole words but still spells new ones (76-89% letter recall).
- **S fixed for anna and saaya** (B: 0% / 9.5% -> 90-100% in every run); thurushi's S 48-67% (B: 0%). **J and Z** 20-96% depending on signer and run (B: 0%), with no motion features yet; per-letter numbers swing between runs (15-45 instances per letter per signer).
- **anna's junk first letter** remains (about 40% of her words: 40% / 21% / 42% in the three runs, now usually F).

**Step 5 (Saaya), weeks 9-10 of the proposal (due 11/13), in order:** add `--save` and store every clip's prediction; finish the proposal's evaluation (CER by word length and signing speed, letter confusion matrix, inference time per clip); motion features (landmark velocities, wrist position and movement, as new columns); speed augmentation (thurushi); trim or augment the relaxed-hand lead-in (anna); LSTM vs Transformer on the same folds and seeds; more data (Kaggle or ChicagoFSWild; check a sample file's format early); recruit 3-5 outside signers now for week 11. The shared Step 5 handoff has the details.

### Original Step 4 brief

**Objective:** train a temporal model that beats baseline B on held-out signers.

**Why:** the static model guesses one frame at a time and relies on a hand-tuned rule to decide where letters start and end; that rule is where most errors come from. A temporal model reads the whole clip, can use motion (J, Z) and learns letter boundaries itself. CTC is what lets it train from (clip, word) pairs without frame-level labels.

**Setup**
1. `git pull` on `temporalwip`.
2. Copy `data_seq/` from Drive into the repo folder.
3. `pip install -r requirements.txt`.
4. Run the three tests above and `python eval_baseline.py`; confirm your numbers match.

**Tasks** (tests first: write the test, watch it fail, write the code, watch it pass)
1. Add `torch` to `requirements.txt`.
2. Resampling: resample each clip to one frame rate (e.g. 15 fps) using `clip.timestamps`, so anna's 29 fps clips look like everyone else's. Test: a 30 fps fake clip comes out with about half the frames.
3. `model_ctc.py`: 2-layer bidirectional LSTM → linear layer with `dataset.NUM_CLASSES` (27) outputs. Test: output shape for a fake batch.
4. Greedy decoder: most likely output per frame → merge repeats → drop blanks → `dataset.decode()`. Test: hand-made outputs decode to the expected word, including a double letter separated by a blank (L, blank, L → "LL").
5. `train_ctc.py`: for each fold in `signer_splits()`, train with CTC loss on the training clips and score the test signer with `metrics.score()`. First check: it can memorize ~10 clips almost perfectly (proves the training loop works).
6. Write `results/lstm_ctc.md` / `.json` with the same columns and per-letter tables as the baseline.

**Done when:** all three folds have CER, word accuracy and letter recall next to baseline B. Losing to the baseline on a first try is fine; the goal is a working pipeline.

**Watch for:** with only two training signers and 55 words, the model can memorize the words instead of learning letters. Compare training-signer scores with held-out scores.

## Decisions carried forward
- **Baseline is frozen.** Do not tune the static model; B is the reference.
- **S waits.** Check whether the temporal model fixes it first; then try thumb-position features (saaya's S→M).
- **Kaggle data waits.** Add the Google ASL Fingerspelling dataset only if the LSTM scores much worse on the held-out signer than on its training signers.
- **Step 5 options, chosen from Step 4's numbers:** Kaggle pretraining (held-out signer much worse), thumb features (S still poor), velocity features or augmentation (J, Z or double letters still poor).
- **Reports are written by the team.** AI-generated code is fine but must be disclosed.
