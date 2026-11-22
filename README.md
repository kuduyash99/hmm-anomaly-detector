# Anomaly Detector with Hidden Markov Models

Real-time anomaly detection on a two-channel machine sensor (temperature, vibration) using a Gaussian-emission
HMM (`hmmlearn`). The HMM is trained on normal data only; sliding windows of the live stream whose likelihood is
unusually low are flagged as anomalies.

## Files
| File | Purpose |
|---|---|
| `generate_data.py` | Simulates the sensor data: `data/normal_train.csv` (normal only) and `data/test_stream.csv` (normal + 4 injected faults, with an `is_anomaly` label column) |
| `anomaly_detector.py` | Model selection, training, threshold, streaming detection, evaluation, plot |
| `outputs/anomaly_detection.png` | Raw sensor readings with detected anomaly regions overlaid |

The data is simulated (3 machine operating modes: idle / normal / heavy load). To use your own data, replace the two CSVs
with the same columns (`temperature`, `vibration`, and `is_anomaly` in the test file).

## How to run
Requires **Python 3.10-3.13** (tested on 3.12). `hmmlearn` has no prebuilt package for Python 3.14 yet, so install fails on it.

```bash
pip install -r requirements.txt
python generate_data.py
python anomaly_detector.py
```

## Method
1. **Preprocessing:** readings are standardised using the normal data's mean and standard deviation.
2. **Model selection:** see below.
3. **Training:** a `GaussianHMM` (diagonal covariance) is fitted on all normal data, keeping the best of 5 random EM restarts.
4. **Threshold:** every 30-step sliding window of the normal data is scored (log-likelihood per time step). The threshold is
   the 0.5th percentile of that distribution, so only ~0.5% of normal windows would be flagged.
5. **Real-time detection:** `StreamDetector` receives one reading at a time, keeps the last 30 readings in a buffer and, once full,
   returns each window's score and whether it falls below the threshold.
6. **Visualisation:** flagged windows are shaded red on both sensor channels; true fault periods are marked in orange.

## Model selection: number of hidden states
HMMs with 1-6 states were fitted on the first 80% of the normal data. Each was compared with the **BIC**
(fit penalised by number of parameters; lower is better) and with log-likelihood on the held-out last 20%.

| States | BIC | Validation log-likelihood / step |
|---|---|---|
| 1 | 13452 | -3.013 |
| 2 | 5563 | -1.295 |
| **3** | **-3936** | **0.851** |
| 4 | -3861 | 0.851 |
| 5 | -3775 | 0.847 |
| 6 | -3678 | 0.779 |

**Chosen: 3 states.** BIC drops sharply up to 3 states and then rises, and validation likelihood stops improving after 3, so extra
states only add parameters without explaining more. This matches the machine's three operating modes (idle, normal, heavy load).

## Results (test stream, 2,200 steps, 4 injected faults)
- Window-level: precision 0.981, recall 0.639, F1 0.774 (a window counts as anomalous if it contains any faulty step, so windows that only
  touch a fault's edge are hard to catch, which lowers recall).
