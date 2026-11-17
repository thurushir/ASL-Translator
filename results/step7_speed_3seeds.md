# LSTM + CTC averaged over 3 runs

Runs: `results/step6_check.json`, `results/step7_speed_seed1.json`, `results/step7_speed_seed2.json`. Held-out signer scores; lower CER is better.
Compare a change by its **mean** over seeds; a gain smaller than the CER range between seeds may be noise.

| Model | Signer | CER per run (step6_check / step7_speed_seed1 / step7_speed_seed2) | CER mean | CER range | Word acc mean | Letter recall mean | CER (no J/Z) mean |
|---|---|---|---|---|---|---|---|
| LSTM+CTC | anna | 27.7% / 36.9% / 30.8% | **31.8%** | 27.7%-36.9% | 17.2% | 84.7% | 25.4% |
| LSTM+CTC | saaya | 9.4% / 12.9% / 13.0% | **11.8%** | 9.4%-13.0% | 59.4% | 90.8% | 11.3% |
| LSTM+CTC | thurushi | 25.0% / 29.9% / 28.7% | **27.8%** | 25.0%-29.9% | 27.5% | 72.9% | 29.9% |
| LSTM+CTC | all | 20.7% / 26.6% / 24.2% | **23.8%** | 20.7%-26.6% | 34.7% | 82.8% | 22.2% |
| Baseline B | anna | 42.0% | 42.0% | - | 7.9% | 65.0% | 38.5% |
| Baseline B | saaya | 39.9% | 39.9% | - | 21.8% | 68.2% | 33.7% |
| Baseline B | thurushi | 27.9% | 27.9% | - | 25.5% | 73.2% | 24.7% |
| Baseline B | all | 36.6% | 36.6% | - | 18.4% | 68.8% | 32.3% |
