# LSTM + CTC averaged over 3 runs

Runs: `results/step7_kt_ours.json`, `results/step7_kt_ours_seed1.json`, `results/step7_kt_ours_seed2.json`. Held-out signer scores; lower CER is better.
Compare a change by its **mean** over seeds; a gain smaller than the CER range between seeds may be noise.

| Model | Signer | CER per run (step7_kt_ours / step7_kt_ours_seed1 / step7_kt_ours_seed2) | CER mean | CER range | Word acc mean | Letter recall mean | CER (no J/Z) mean |
|---|---|---|---|---|---|---|---|
| LSTM+CTC | kaggle_unseen | 84.6% / 81.7% / 82.5% | **82.9%** | 81.7%-84.6% | 0.0% | 17.1% | 82.4% |
| LSTM+CTC | all | 84.6% / 81.7% / 82.5% | **82.9%** | 81.7%-84.6% | 0.0% | 17.1% | 82.4% |
