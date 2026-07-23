# Cell 10b – Rolling Isotonic Calibrator (dynamic probability calibration)
import numpy as np
from sklearn.isotonic import IsotonicRegression
from collections import deque

class RollingCalibrator:
    """
    Maintains a rolling window of (raw_prediction, actual_outcome) pairs.
    Fits an IsotonicRegression on the last `window_size` games.
    Predicts calibrated probability for new raw predictions.
    """
    def __init__(self, window_size=250, out_of_bounds='clip'):
        self.window_size = window_size
        self.out_of_bounds = out_of_bounds
        self.buffer = deque(maxlen=window_size)
        self.model = None

    def update(self, raw_prob, outcome):
        """Add a new calibration point (raw_prob in [0,1], outcome 0/1)."""
        self.buffer.append((raw_prob, outcome))
        if len(self.buffer) >= 20:   # refit only after enough samples
            self._fit()

    def _fit(self):
        if len(self.buffer) < 5:
            return
        X = np.array([p for p, _ in self.buffer]).reshape(-1, 1)
        y = np.array([o for _, o in self.buffer])
        iso = IsotonicRegression(out_of_bounds=self.out_of_bounds, increasing=True)
        iso.fit(X.ravel(), y)
        self.model = iso

    def predict(self, raw_prob):
        """Calibrate a single raw probability (clips to [0.01,0.99])."""
        if self.model is None or len(self.buffer) < 5:
            return np.clip(raw_prob, 0.01, 0.99)
        raw = np.clip(raw_prob, 0.0, 1.0)
        cal = self.model.predict([raw])[0]
        return np.clip(cal, 0.01, 0.99)
from sklearn.linear_model import LogisticRegression
from collections import deque

class RollingPlattCalibrator:
    """
    Rolling Platt scaling: fits a logistic regression on a rolling window
    of (raw_margin, outcome) to calibrate win probabilities.
    """
    def __init__(self, window_size=300, min_samples=20, cold_start_fn=None,
                 fallback_scale=12.0, variance_aware=True, ref_total=225.0):
        self.window_size = window_size
        self.min_samples = min_samples
        self.buffer = deque(maxlen=window_size)   # stores (adj_margin, outcome)
        self.model = None
        # Cold-start: before enough in-season points accrue, defer to the
        # model's trained calibrator (a single coherent path) instead of a
        # bare sigmoid. Falls back to sigmoid(margin / fallback_scale).
        self.cold_start_fn = cold_start_fn
        self.fallback_scale = fallback_scale
        # Variance-aware: a given margin is less decisive in a high-scoring
        # (higher-variance) game, so scale the margin by sqrt(total / ref_total)
        # before calibrating. ref_total ~ league-average game total.
        self.variance_aware = variance_aware
        self.ref_total = ref_total

    def _adj(self, raw_margin, total):
        if self.variance_aware and total and total > 0:
            return raw_margin / np.sqrt(max(float(total), 1.0) / self.ref_total)
        return raw_margin

    def update(self, raw_margin, outcome, total=None):
        """Add a new calibration point (raw margin, 0/1 outcome, optional total)."""
        self.buffer.append((self._adj(raw_margin, total), outcome))
        if len(self.buffer) >= self.min_samples:
            self._fit()

    def _fit(self):
        X = np.array([x for x, _ in self.buffer]).reshape(-1, 1)
        y = np.array([y for _, y in self.buffer])
        # Use strong L2 regularization to avoid overfitting
        clf = LogisticRegression(C=0.1, solver='lbfgs', max_iter=100)
        clf.fit(X, y)
        self.model = clf

    def predict(self, raw_margin, total=None):
        """Return calibrated win probability (clipped to [0.01,0.99])."""
        if self.model is None or len(self.buffer) < self.min_samples:
            if self.cold_start_fn is not None:
                try:
                    return float(np.clip(self.cold_start_fn(raw_margin), 0.01, 0.99))
                except Exception:
                    pass
            # Fallback: sigmoid with scaling
            p = 1.0 / (1.0 + np.exp(-raw_margin / self.fallback_scale))
            return np.clip(p, 0.01, 0.99)
        prob = self.model.predict_proba(np.array([[self._adj(raw_margin, total)]]))[0, 1]
        return np.clip(prob, 0.01, 0.99)