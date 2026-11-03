# LSTM + CTC averaged over 3 runs

Runs: `results/step5_lstm.json`, `results/step5_lstm_seed1.json`, `results/step5_lstm_seed2.json`. Held-out signer scores; lower CER is better.
Compare a change by its **mean** over seeds; a gain smaller than the CER range between seeds may be noise.

| Model | Signer | CER per run (step5_lstm / step5_lstm_seed1 / step5_lstm_seed2) | CER mean | CER range | Word acc mean | Letter recall mean | CER (no J/Z) mean |
|---|---|---|---|---|---|---|---|
| LSTM+CTC | anna | 30.1% / 32.6% / 31.1% | **31.3%** | 30.1%-32.6% | 18.2% | 83.7% | 26.4% |
| LSTM+CTC | saaya | 12.1% / 16.3% / 14.8% | **14.4%** | 12.1%-16.3% | 53.9% | 90.5% | 13.9% |
| LSTM+CTC | thurushi | 36.7% / 34.0% / 36.9% | **35.9%** | 34.0%-36.9% | 18.2% | 64.8% | 37.4% |
| LSTM+CTC | all | 26.3% / 27.7% / 27.6% | **27.2%** | 26.3%-27.7% | 30.1% | 79.7% | 25.9% |
| Baseline B | anna | 42.0% | 42.0% | - | 7.9% | 65.0% | 38.5% |
| Baseline B | saaya | 39.9% | 39.9% | - | 21.8% | 68.2% | 33.7% |
| Baseline B | thurushi | 27.9% | 27.9% | - | 25.5% | 73.2% | 24.7% |
| Baseline B | all | 36.6% | 36.6% | - | 18.4% | 68.8% | 32.3% |
