# LSTM + CTC averaged over 3 runs

Runs: `results/step7_kt.json`, `results/step7_kt_seed1.json`, `results/step7_kt_seed2.json`. Held-out signer scores; lower CER is better.
Compare a change by its **mean** over seeds; a gain smaller than the CER range between seeds may be noise.

| Model | Signer | CER per run (step7_kt / step7_kt_seed1 / step7_kt_seed2) | CER mean | CER range | Word acc mean | Letter recall mean | CER (no J/Z) mean |
|---|---|---|---|---|---|---|---|
| LSTM+CTC | kaggle_unseen | 59.2% / 59.2% / 58.6% | **59.0%** | 58.6%-59.2% | 0.0% | 47.1% | 57.3% |
| LSTM+CTC | all | 59.2% / 59.2% / 58.6% | **59.0%** | 58.6%-59.2% | 0.0% | 47.1% | 57.3% |
