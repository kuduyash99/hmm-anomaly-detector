"""HMM anomaly detector: model selection, training, thresholding, streaming detection, plot.
Usage: python generate_data.py && python anomaly_detector.py"""

from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM

SEED = 42
FEATURES = ["temperature", "vibration"]
CANDIDATE_STATES = range(1, 7)
N_RESTARTS = 5            # random restarts per model (EM can get stuck in poor optima)
WINDOW = 30               # sliding window length (time steps)
THRESHOLD_PERCENTILE = 0.5  # windows scoring below this percentile of normal data are anomalies


def fit_hmm(X, n_states):
    """Fit a Gaussian HMM, keeping the best of several random restarts."""
    best_model, best_score = None, -np.inf
    for restart in range(N_RESTARTS):
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=SEED + restart)
        model.fit(X)
        score = model.score(X)
        if score > best_score:
            best_model, best_score = model, score
    return best_model


def bic(model, X):
    """Bayesian Information Criterion (lower is better) for a diagonal-covariance Gaussian HMM."""
    n, d = model.n_components, X.shape[1]
    n_params = (n - 1) + n * (n - 1) + 2 * n * d  # start probs + transitions + means + variances
    return n_params * np.log(len(X)) - 2 * model.score(X)


def select_n_states(X_train, X_val):
    """Fit models with different numbers of hidden states and pick the one with the lowest BIC."""
    rows = []
    for n in CANDIDATE_STATES:
        model = fit_hmm(X_train, n)
        rows.append({"n_states": n, "train_BIC": bic(model, X_train),
                     "val_loglik_per_step": model.score(X_val) / len(X_val)})
    table = pd.DataFrame(rows)
    return int(table.loc[table["train_BIC"].idxmin(), "n_states"]), table


def window_scores(model, X, window=WINDOW):
    """Per-step log-likelihood of every sliding window in X (stride 1)."""
    return np.array([model.score(X[i:i + window]) / window for i in range(len(X) - window + 1)])


class StreamDetector:
    """Real-time detector: feed one reading at a time; once WINDOW readings are buffered,
    each new reading returns (window_score, is_anomaly)."""

    def __init__(self, model, threshold, mean, std, window=WINDOW):
        self.model, self.threshold, self.mean, self.std = model, threshold, mean, std
        self.buffer = deque(maxlen=window)

    def update(self, reading):
        self.buffer.append((np.asarray(reading) - self.mean) / self.std)
        if len(self.buffer) < self.buffer.maxlen:
            return None
        score = self.model.score(np.array(self.buffer)) / self.buffer.maxlen
        return score, score < self.threshold


def plot_results(test, flagged, path):
    """Raw sensor readings with detected anomaly regions (red) and true fault periods (orange bar)."""
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(14, 7), sharex=True)
    for ax, feature in zip(axes, FEATURES):
        ax.plot(test[feature], color="steelblue", lw=0.8)
        in_region, start = False, 0
        for t, is_flagged in enumerate(np.append(flagged, False)):
            if is_flagged and not in_region:
                in_region, start = True, t
            elif not is_flagged and in_region:
                ax.axvspan(start, t, color="red", alpha=0.25, label="Detected anomaly")
                in_region = False
        true_idx = np.flatnonzero(test["is_anomaly"])
        ax.plot(true_idx, np.full(len(true_idx), ax.get_ylim()[0]), "|", color="orange", ms=8, label="True fault")
        ax.set_ylabel(feature)
        handles, names = ax.get_legend_handles_labels()
        unique = dict(zip(names, handles))  # drop duplicate legend entries
        ax.legend(unique.values(), unique.keys(), loc="upper left")
    axes[-1].set_xlabel("Time step")
    fig.suptitle("HMM anomaly detection on sensor stream")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    train = pd.read_csv("data/normal_train.csv")
    test = pd.read_csv("data/test_stream.csv")
    mean, std = train[FEATURES].mean().values, train[FEATURES].std().values
    X_train, X_test = (train[FEATURES].values - mean) / std, (test[FEATURES].values - mean) / std

    # 1) Model selection (chronological 80/20 split of the normal data)
    split = int(0.8 * len(X_train))
    best_n, table = select_n_states(X_train[:split], X_train[split:])
    print(table.round(3).to_string(index=False))
    print(f"\nSelected number of hidden states: {best_n} (lowest BIC)")

    # 2) Train final model on all normal data and set threshold from its likelihood distribution
    model = fit_hmm(X_train, best_n)
    train_scores = window_scores(model, X_train)
    threshold = np.percentile(train_scores, THRESHOLD_PERCENTILE)
    print(f"Threshold (percentile {THRESHOLD_PERCENTILE} of normal windows): {threshold:.3f} per step")

    # 3) Stream the test data one reading at a time
    detector = StreamDetector(model, threshold, mean, std)
    window_flags = []
    for reading in test[FEATURES].values:
        result = detector.update(reading)
        if result is not None:
            window_flags.append(result[1])
    window_flags = np.array(window_flags)

    # Window i covers steps i .. i+WINDOW-1; a window is truly anomalous if it contains any faulty step
    labels = test["is_anomaly"].values
    true_windows = np.array([labels[i:i + WINDOW].any() for i in range(len(window_flags))])
    tp = np.sum(window_flags & true_windows)
    precision = tp / max(window_flags.sum(), 1)
    recall = tp / true_windows.sum()
    print(f"\nWindow-level detection: precision={precision:.3f}  recall={recall:.3f}  "
          f"F1={2 * precision * recall / (precision + recall):.3f}")

    # 4) Visualise: shade every step covered by a flagged window
    flagged_steps = np.zeros(len(test), dtype=bool)
    for i in np.flatnonzero(window_flags):
        flagged_steps[i:i + WINDOW] = True
    plot_results(test, flagged_steps, "outputs/anomaly_detection.png")
    print("Saved outputs/anomaly_detection.png")


if __name__ == "__main__":
    main()
