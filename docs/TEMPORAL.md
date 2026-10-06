# Temporal (continuous fingerspelling) work — `temporalwip` branch

The `main` branch stays as the static-letter version. This branch moves toward recognizing continuous fingerspelling.

## Step 1: setup (done)

| File | What it does |
|---|---|
| `features.py` | **The only place features are computed.** `frame_features()` gives 64 numbers for one frame; `sequence_features()` does the same for a whole clip. |
| `check_features.py` | Confirms `features.py` matches the original feature code. |
| `hand_utils.py` | `init_hands()` now actually uses its settings; `start_video()` can pass key presses to a script. |
| `train_model.py` | Also saves `baseline_results.json` (the static model's accuracy, our reference). |
| `collect_sequences.py` | Records clips of fingerspelled words as per-frame landmarks to `data_seq/<WORD>/<n>.npz`. |
| `check_sequences.py` | Summarizes recorded clips and flags bad ones. |
| `transcript.py` | Turns per-frame predictions into text that stays on screen (rules are at the top of the file). |
| `test_transcript.py` | Tests the transcript rules without a webcam. |

### Clip format (`.npz`)
- `landmarks`: (frames, 21, 3) raw x, y, z; rows are NaN when no hand was found
- `timestamps`: seconds since the clip started
- `word`: the label; `mirrored`: True (frames are mirrored like the live view)

Raw landmarks are saved (not features) so the feature design can change later without re-recording.

### Live controls (`predict_live.py`)
Hold a letter steadily to add it. Relax your hand briefly before repeating a letter (the LL in HELLO).
Drop your hand for 1s for a space. Keys: `space` = space, `backspace` = delete, `c` = clear, `q` = quit.

## How to verify
```bash
python check_features.py        # PASS: features match
python test_transcript.py       # PASS: all transcript tests passed
python process_data.py && python train_model.py   # ~92-93% accuracy, writes baseline_results.json
python collect_sequences.py --word HELLO           # r = start/stop, q = quit
python check_sequences.py
python predict_live.py
```

## Known limits / next steps
- J and Z still don't work (they need motion, i.e. the temporal model).
- Fast fingerspelling may skip letters, since the model is still per-frame.
- Next: record a sequence dataset, add landmark augmentation, train a sequence model.
