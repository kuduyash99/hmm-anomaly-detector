"""Simulate machine sensor data (temperature, vibration) with 3 operating modes.
Writes data/normal_train.csv (normal only) and data/test_stream.csv (normal + injected anomalies).
Replace these CSVs with your own data if you have it (columns: temperature, vibration[, is_anomaly])."""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
FEATURES = ["temperature", "vibration"]

# Operating modes: idle, normal load, heavy load
MODE_MEANS = np.array([[40.0, 0.5], [60.0, 1.5], [80.0, 3.0]])
MODE_STD = np.array([[1.5, 0.08], [2.0, 0.15], [2.5, 0.25]])
TRANSITIONS = np.array([[0.97, 0.03, 0.00],
                        [0.02, 0.96, 0.02],
                        [0.00, 0.04, 0.96]])


def simulate_normal(n_steps):
    """Sample a Markov chain of operating modes and emit Gaussian sensor readings."""
    modes = np.empty(n_steps, dtype=int)
    modes[0] = 1
    for t in range(1, n_steps):
        modes[t] = RNG.choice(3, p=TRANSITIONS[modes[t - 1]])
    return MODE_MEANS[modes] + RNG.normal(size=(n_steps, 2)) * MODE_STD[modes]

def inject_anomalies(data):
    """Add 4 fault types to the stream and return (data, labels)."""
    data, labels = data.copy(), np.zeros(len(data), dtype=int)
    faults = [
        (400, 440, lambda x: x + np.array([0.0, 4.0])),                         # vibration spike
        (900, 960, lambda x: x + np.linspace(0, 25, len(x))[:, None] * [1, 0]),  # overheating drift
        (1400, 1450, lambda x: np.tile(x[0], (len(x), 1))),                     # sensor stuck
        (1800, 1850, lambda x: x + RNG.normal(size=x.shape) * [8.0, 1.0]),      # noise burst
    ]
    for start, end, fault in faults:
        data[start:end] = fault(data[start:end])
        labels[start:end] = 1
    return data, labels


if __name__ == "__main__":
    pd.DataFrame(simulate_normal(3000), columns=FEATURES).to_csv("data/normal_train.csv", index=False)
    stream, labels = inject_anomalies(simulate_normal(2200))
    test = pd.DataFrame(stream, columns=FEATURES)
    test["is_anomaly"] = labels
    test.to_csv("data/test_stream.csv", index=False)
    print("Wrote data/normal_train.csv (3000 rows) and data/test_stream.csv (2200 rows)")
