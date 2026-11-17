# LSTM + CTC averaged over 3 runs

Runs: `results/step7_kaggle_2f.json`, `results/step7_kaggle_2f_seed1.json`, `results/step7_kaggle_2f_seed2.json`. Held-out signer scores; lower CER is better.
Compare a change by its **mean** over seeds; a gain smaller than the CER range between seeds may be noise.

| Model | Signer | CER per run (step7_kaggle_2f / step7_kaggle_2f_seed1 / step7_kaggle_2f_seed2) | CER mean | CER range | Word acc mean | Letter recall mean | CER (no J/Z) mean |
|---|---|---|---|---|---|---|---|
| LSTM+CTC | anna | 35.7% / 36.6% / 39.1% | **37.2%** | 35.7%-39.1% | 13.3% | 87.0% | 33.0% |
| LSTM+CTC | saaya | 6.8% / 8.1% / 7.9% | **7.6%** | 6.8%-8.1% | 71.7% | 94.4% | 7.0% |
| LSTM+CTC | thurushi | 18.6% / 22.0% / 16.7% | **19.1%** | 16.7%-22.0% | 42.8% | 83.1% | 20.4% |
| LSTM+CTC | all | 20.4% / 22.2% / 21.3% | **21.3%** | 20.4%-22.2% | 42.6% | 88.2% | 20.1% |
| Baseline B | anna | 42.0% | 42.0% | - | 7.9% | 65.0% | 38.5% |
| Baseline B | saaya | 39.9% | 39.9% | - | 21.8% | 68.2% | 33.7% |
| Baseline B | thurushi | 27.9% | 27.9% | - | 25.5% | 73.2% | 24.7% |
| Baseline B | all | 36.6% | 36.6% | - | 18.4% | 68.8% | 32.3% |
