# LSTM + CTC averaged over 3 runs

Runs: `results/step6_motion_speed.json`, `results/step6_motion_speed_seed1.json`, `results/step6_motion_speed_seed2.json`. Held-out signer scores; lower CER is better.
Compare a change by its **mean** over seeds; a gain smaller than the CER range between seeds may be noise.

| Model | Signer | CER per run (step6_motion_speed / step6_motion_speed_seed1 / step6_motion_speed_seed2) | CER mean | CER range | Word acc mean | Letter recall mean | CER (no J/Z) mean |
|---|---|---|---|---|---|---|---|
| LSTM+CTC | anna | 38.7% / 31.8% / 32.2% | **34.2%** | 31.8%-38.7% | 18.4% | 91.0% | 29.1% |
| LSTM+CTC | saaya | 13.9% / 11.6% / 12.5% | **12.7%** | 11.6%-13.9% | 58.0% | 90.7% | 12.1% |
| LSTM+CTC | thurushi | 35.2% / 33.3% / 31.5% | **33.3%** | 31.5%-35.2% | 19.6% | 67.0% | 35.5% |
| LSTM+CTC | all | 29.3% / 25.6% / 25.4% | **26.7%** | 25.4%-29.3% | 32.0% | 82.9% | 25.6% |
| Baseline B | anna | 42.0% | 42.0% | - | 7.9% | 65.0% | 38.5% |
| Baseline B | saaya | 39.9% | 39.9% | - | 21.8% | 68.2% | 33.7% |
| Baseline B | thurushi | 27.9% | 27.9% | - | 25.5% | 73.2% | 24.7% |
| Baseline B | all | 36.6% | 36.6% | - | 18.4% | 68.8% | 32.3% |
