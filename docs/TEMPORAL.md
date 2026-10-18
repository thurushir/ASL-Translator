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

## Step 4: first LSTM + CTC model (next, Saaya or Anna)

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
