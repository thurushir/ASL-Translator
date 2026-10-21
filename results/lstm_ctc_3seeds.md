# LSTM + CTC averaged over 3 runs

Runs: `results/lstm_ctc.json`, `results/lstm_ctc_seed1.json`, `results/lstm_ctc_seed2.json`. Held-out signer scores; lower CER is better.
Compare a change by its **mean** over seeds; a gain smaller than the CER range between seeds may be noise.

| Model | Signer | CER per run (lstm_ctc / lstm_ctc_seed1 / lstm_ctc_seed2) | CER mean | CER range | Word acc mean | Letter recall mean | CER (no J/Z) mean |
|---|---|---|---|---|---|---|---|
| LSTM+CTC | anna | 29.8% / 29.5% / 36.9% | **32.1%** | 29.5%-36.9% | 23.8% | 84.7% | 26.2% |
| LSTM+CTC | saaya | 10.1% / 15.6% / 17.0% | **14.2%** | 10.1%-17.0% | 52.7% | 91.2% | 13.6% |
| LSTM+CTC | thurushi | 36.0% / 38.2% / 38.3% | **37.5%** | 36.0%-38.3% | 17.6% | 63.0% | 38.3% |
| LSTM+CTC | all | 25.3% / 27.8% / 30.7% | **27.9%** | 25.3%-30.7% | 31.4% | 79.7% | 26.1% |
| Baseline B | anna | 42.0% | 42.0% | - | 7.9% | 65.0% | 38.5% |
| Baseline B | saaya | 39.9% | 39.9% | - | 21.8% | 68.2% | 33.7% |
| Baseline B | thurushi | 27.9% | 27.9% | - | 25.5% | 73.2% | 24.7% |
| Baseline B | all | 36.6% | 36.6% | - | 18.4% | 68.8% | 32.3% |
