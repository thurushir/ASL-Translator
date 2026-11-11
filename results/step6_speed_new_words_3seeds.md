# LSTM + CTC averaged over 3 runs

Runs: `results/step6_speed_new_words.json`, `results/step6_speed_new_words_seed1.json`, `results/step6_speed_new_words_seed2.json`. Held-out signer scores; lower CER is better.
Compare a change by its **mean** over seeds; a gain smaller than the CER range between seeds may be noise.

| Model | Signer | CER per run (step6_speed_new_words / step6_speed_new_words_seed1 / step6_speed_new_words_seed2) | CER mean | CER range | Word acc mean | Letter recall mean | CER (no J/Z) mean |
|---|---|---|---|---|---|---|---|
| LSTM+CTC | anna | 33.2% / 31.3% / 29.8% | **31.4%** | 29.8%-33.2% | 19.4% | 84.0% | 26.1% |
| LSTM+CTC | saaya | 9.1% / 11.9% / 12.1% | **11.0%** | 9.1%-12.1% | 63.8% | 91.4% | 9.9% |
| LSTM+CTC | thurushi | 27.4% / 31.8% / 30.2% | **29.8%** | 27.4%-31.8% | 22.8% | 70.7% | 31.3% |
| LSTM+CTC | all | 23.2% / 25.0% / 24.0% | **24.1%** | 23.2%-25.0% | 35.4% | 82.0% | 22.4% |
| Baseline B | anna | 42.0% | 42.0% | - | 7.9% | 65.0% | 38.5% |
| Baseline B | saaya | 39.9% | 39.9% | - | 21.8% | 68.2% | 33.7% |
| Baseline B | thurushi | 27.9% | 27.9% | - | 25.5% | 73.2% | 24.7% |
| Baseline B | all | 36.6% | 36.6% | - | 18.4% | 68.8% | 32.3% |
