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

## Known limits / next steps
- J and Z still don't work (they need motion, i.e. the temporal model).
- Fast fingerspelling may skip letters, since the model is still per-frame.
- Next (step 3): load clips for training, landmark augmentation, test on a held-out signer, score the static model on clips.
