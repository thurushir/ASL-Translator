# LSTM + CTC averaged over 3 runs

Runs: `results/step7_kaggle.json`, `results/step7_kaggle_seed1.json`, `results/step7_kaggle_seed2.json`. Held-out signer scores; lower CER is better.
Compare a change by its **mean** over seeds; a gain smaller than the CER range between seeds may be noise.

| Model | Signer | CER per run (step7_kaggle / step7_kaggle_seed1 / step7_kaggle_seed2) | CER mean | CER range | Word acc mean | Letter recall mean | CER (no J/Z) mean |
|---|---|---|---|---|---|---|---|
| LSTM+CTC | anna | 35.7% / 34.0% / 33.2% | **34.3%** | 33.2%-35.7% | 16.8% | 88.6% | 28.2% |
| LSTM+CTC | saaya | 13.3% / 13.6% / 7.9% | **11.6%** | 7.9%-13.6% | 59.8% | 91.2% | 10.8% |
| LSTM+CTC | thurushi | 22.8% / 27.5% / 29.1% | **26.5%** | 22.8%-29.1% | 27.7% | 75.1% | 27.6% |
| LSTM+CTC | all | 24.0% / 25.1% / 23.4% | **24.1%** | 23.4%-25.1% | 34.7% | 84.9% | 22.2% |
| Baseline B | anna | 42.0% | 42.0% | - | 7.9% | 65.0% | 38.5% |
| Baseline B | saaya | 39.9% | 39.9% | - | 21.8% | 68.2% | 33.7% |
| Baseline B | thurushi | 27.9% | 27.9% | - | 25.5% | 73.2% | 24.7% |
| Baseline B | all | 36.6% | 36.6% | - | 18.4% | 68.8% | 32.3% |
