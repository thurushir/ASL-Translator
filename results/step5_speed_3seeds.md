# LSTM + CTC averaged over 3 runs

Runs: `results/step5_speed.json`, `results/step5_speed_seed1.json`, `results/step5_speed_seed2.json`. Held-out signer scores; lower CER is better.
Compare a change by its **mean** over seeds; a gain smaller than the CER range between seeds may be noise.

| Model | Signer | CER per run (step5_speed / step5_speed_seed1 / step5_speed_seed2) | CER mean | CER range | Word acc mean | Letter recall mean | CER (no J/Z) mean |
|---|---|---|---|---|---|---|---|
| LSTM+CTC | anna | 27.7% / 35.7% / 30.4% | **31.3%** | 27.7%-35.7% | 17.4% | 82.2% | 25.3% |
| LSTM+CTC | saaya | 9.4% / 11.8% / 12.8% | **11.3%** | 9.4%-12.8% | 62.6% | 90.8% | 10.5% |
| LSTM+CTC | thurushi | 25.0% / 29.5% / 29.2% | **27.9%** | 25.0%-29.5% | 25.1% | 73.2% | 29.9% |
| LSTM+CTC | all | 20.7% / 25.7% / 24.1% | **23.5%** | 20.7%-25.7% | 35.0% | 82.1% | 21.9% |
| Baseline B | anna | 42.0% | 42.0% | - | 7.9% | 65.0% | 38.5% |
| Baseline B | saaya | 39.9% | 39.9% | - | 21.8% | 68.2% | 33.7% |
| Baseline B | thurushi | 27.9% | 27.9% | - | 25.5% | 73.2% | 24.7% |
| Baseline B | all | 36.6% | 36.6% | - | 18.4% | 68.8% | 32.3% |
