# Transformer + CTC averaged over 3 runs

Runs: `results/step5_tf.json`, `results/step5_tf_seed1.json`, `results/step5_tf_seed2.json`. Held-out signer scores; lower CER is better.
Compare a change by its **mean** over seeds; a gain smaller than the CER range between seeds may be noise.

| Model | Signer | CER per run (step5_tf / step5_tf_seed1 / step5_tf_seed2) | CER mean | CER range | Word acc mean | Letter recall mean | CER (no J/Z) mean |
|---|---|---|---|---|---|---|---|
| Transformer+CTC | anna | 40.7% / 37.4% / 40.6% | **39.6%** | 37.4%-40.7% | 8.3% | 72.2% | 36.2% |
| Transformer+CTC | saaya | 21.7% / 21.8% / 17.7% | **20.4%** | 17.7%-21.8% | 39.2% | 88.5% | 20.7% |
| Transformer+CTC | thurushi | 32.6% / 40.0% / 37.7% | **36.8%** | 32.6%-40.0% | 15.8% | 64.1% | 39.0% |
| Transformer+CTC | all | 31.7% / 33.1% / 32.0% | **32.3%** | 31.7%-33.1% | 21.1% | 74.9% | 32.0% |
| Baseline B | anna | 42.0% | 42.0% | - | 7.9% | 65.0% | 38.5% |
| Baseline B | saaya | 39.9% | 39.9% | - | 21.8% | 68.2% | 33.7% |
| Baseline B | thurushi | 27.9% | 27.9% | - | 25.5% | 73.2% | 24.7% |
| Baseline B | all | 36.6% | 36.6% | - | 18.4% | 68.8% | 32.3% |
