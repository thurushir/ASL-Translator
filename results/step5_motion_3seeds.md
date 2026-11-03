# LSTM + CTC averaged over 3 runs

Runs: `results/step5_motion.json`, `results/step5_motion_seed1.json`, `results/step5_motion_seed2.json`. Held-out signer scores; lower CER is better.
Compare a change by its **mean** over seeds; a gain smaller than the CER range between seeds may be noise.

| Model | Signer | CER per run (step5_motion / step5_motion_seed1 / step5_motion_seed2) | CER mean | CER range | Word acc mean | Letter recall mean | CER (no J/Z) mean |
|---|---|---|---|---|---|---|---|
| LSTM+CTC | anna | 34.8% / 30.9% / 25.1% | **30.3%** | 25.1%-34.8% | 22.0% | 88.7% | 25.4% |
| LSTM+CTC | saaya | 16.0% / 27.8% / 12.9% | **18.9%** | 12.9%-27.8% | 45.1% | 86.9% | 17.7% |
| LSTM+CTC | thurushi | 45.4% / 49.1% / 40.9% | **45.1%** | 40.9%-49.1% | 9.1% | 54.9% | 45.8% |
| LSTM+CTC | all | 32.1% / 35.9% / 26.3% | **31.4%** | 26.3%-35.9% | 25.4% | 76.8% | 29.6% |
| Baseline B | anna | 42.0% | 42.0% | - | 7.9% | 65.0% | 38.5% |
| Baseline B | saaya | 39.9% | 39.9% | - | 21.8% | 68.2% | 33.7% |
| Baseline B | thurushi | 27.9% | 27.9% | - | 25.5% | 73.2% | 24.7% |
| Baseline B | all | 36.6% | 36.6% | - | 18.4% | 68.8% | 32.3% |
