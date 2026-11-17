# LSTM + CTC averaged over 3 runs

Runs: `results/step7_kt_ours_2f.json`, `results/step7_kt_ours_2f_seed1.json`, `results/step7_kt_ours_2f_seed2.json`. Held-out signer scores; lower CER is better.
Compare a change by its **mean** over seeds; a gain smaller than the CER range between seeds may be noise.

| Model | Signer | CER per run (step7_kt_ours_2f / step7_kt_ours_2f_seed1 / step7_kt_ours_2f_seed2) | CER mean | CER range | Word acc mean | Letter recall mean | CER (no J/Z) mean |
|---|---|---|---|---|---|---|---|
| LSTM+CTC | kaggle_unseen | 84.7% / 82.6% / 81.7% | **83.0%** | 81.7%-84.7% | 0.0% | 17.2% | 84.9% |
| LSTM+CTC | all | 84.7% / 82.6% / 81.7% | **83.0%** | 81.7%-84.7% | 0.0% | 17.2% | 84.9% |
