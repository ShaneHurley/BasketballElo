"""Task 026/027: de-vig candidate transforms and leakage-free method selection.

Converts a book's raw implied probabilities (``q_i = 1 / decimal_i``, one per
side of a market) into "fair" probabilities with the vig removed, via several
candidate transforms. No transform is assumed universally correct (plan
Phase 2D): the best method depends on book/market/price range/horizon, so
Task 027's selector chooses among them using **only prior-fold** OOS Brier
score / log loss / calibration error, never the fold currently being
evaluated/tested.

Every candidate function returns probabilities that are finite, in ``[0, 1]``,
sum to 1 within numerical tolerance, and preserve the input's favorite/
underdog ordering (monotonicity) — enforced by ``_check_output`` so a bad
solver root cannot silently emit a nonsensical probability vector.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "DevigError",
    "multiplicative_devig",
    "power_devig",
    "odds_ratio_devig",
    "shin_devig",
    "DEVIG_METHODS",
    "devig",
    "brier_score",
    "log_loss_score",
    "calibration_error",
    "select_devig_method_from_folds",
    "store_devig_choice",
]


class DevigError(ValueError):
    """Raised when raw implied probabilities or a devig result fail a guard."""


_EPS = 1e-9


def _validate_probs(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or q.size < 2:
        raise DevigError("q must be a 1-D array of at least 2 implied probabilities")
    if not np.all(np.isfinite(q)):
        raise DevigError("implied probabilities must be finite")
    if np.any(q <= 0) or np.any(q >= 1):
        raise DevigError("implied probabilities must lie strictly inside (0, 1)")
    return q


def _check_output(p, q: np.ndarray) -> np.ndarray:
    """Numerical guards shared by every candidate transform (Task 026)."""
    p = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(p)):
        raise DevigError("devig output is not finite")
    if np.any(p < -1e-9) or np.any(p > 1 + 1e-9):
        raise DevigError("devig output outside [0, 1]")
    p = np.clip(p, 0.0, 1.0)
    total = float(p.sum())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise DevigError(f"devig output does not sum to 1 (sum={total})")
    p = p / total  # exact renormalization after clipping
    # Monotonicity: a larger raw implied probability must map to a larger (or
    # equal) fair probability — the transform may not reorder favorite/dog.
    rank_in = np.argsort(np.argsort(q))
    rank_out = np.argsort(np.argsort(p))
    if not np.array_equal(rank_in, rank_out):
        raise DevigError("devig output re-ordered favorite/underdog vs. input")
    return p


def multiplicative_devig(q) -> np.ndarray:
    """p_i = q_i / sum(q)."""
    q = _validate_probs(q)
    p = q / q.sum()
    return _check_output(p, q)


def _bisect(f, lo, hi, tol=1e-10, max_iter=200):
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol:
            return mid
        if flo * fmid < 0:
            hi = mid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def power_devig(q, k_bounds=(0.05, 20.0)) -> np.ndarray:
    """Solve sum(q_i^k) = 1 for k, then p_i = q_i^k (Shin/Jullien favorite-
    longshot power correction)."""
    q = _validate_probs(q)

    def f(k):
        return float(np.sum(q ** k) - 1.0)

    k = _bisect(f, *k_bounds)
    if k is None:
        return multiplicative_devig(q)
    p = q ** k
    return _check_output(p, q)


def odds_ratio_devig(q) -> np.ndarray:
    """Odds-ratio / logit-shift method: solve for c such that
    sum_i (c*q_i) / (1 - q_i + c*q_i) = 1."""
    q = _validate_probs(q)

    def transform(c):
        return (c * q) / (1.0 - q + c * q)

    def f(c):
        return float(transform(c).sum() - 1.0)

    # c spans orders of magnitude; bisect in log-space for stability.
    def f_log(x):
        return f(float(np.exp(x)))

    x = _bisect(f_log, -20.0, 20.0)
    if x is None:
        return multiplicative_devig(q)
    c = float(np.exp(x))
    p = transform(c)
    return _check_output(p, q)


def _shin_transform(q: np.ndarray, z: float, sigma: float) -> np.ndarray:
    disc = z ** 2 + 4.0 * (1.0 - z) * (q ** 2) / sigma
    disc = np.clip(disc, 0.0, None)
    denom = 2.0 * (1.0 - z)
    return (np.sqrt(disc) - z) / denom


def shin_devig(q) -> np.ndarray:
    """Shin (1992/1993) insider-trading/allocation model.

    Solves for the market's implied "insider fraction" z such that the
    resulting fair probabilities sum to 1. Uses the standard closed-form
    functional relationship, generalized to n outcomes (reduces to the
    textbook two-way formula when ``len(q) == 2``).
    """
    q = _validate_probs(q)
    sigma = float(q.sum())

    def f(z):
        return float(_shin_transform(q, z, sigma).sum() - 1.0)

    z = _bisect(f, 1e-9, 1.0 - 1e-6)
    if z is None:
        return multiplicative_devig(q)
    p = _shin_transform(q, z, sigma)
    return _check_output(p, q)


DEVIG_METHODS = {
    "multiplicative": multiplicative_devig,
    "power": power_devig,
    "odds_ratio": odds_ratio_devig,
    "shin": shin_devig,
}


def devig(q, method: str = "multiplicative") -> np.ndarray:
    if method not in DEVIG_METHODS:
        raise DevigError(f"unknown devig method: {method!r} (have {list(DEVIG_METHODS)})")
    return DEVIG_METHODS[method](q)


# ---------------------------------------------------------------------------
# Task 027: leakage-free method selection.
# ---------------------------------------------------------------------------

def brier_score(p_pred, outcome) -> float:
    p = np.asarray(p_pred, dtype=float)
    y = np.asarray(outcome, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss_score(p_pred, outcome, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1 - eps)
    y = np.asarray(outcome, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_error(p_pred, outcome, n_bins: int = 10) -> float:
    """Mean |predicted - observed| across equal-width probability bins
    (sample-weighted expected calibration error)."""
    p = np.asarray(p_pred, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if p.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    err_sum = 0.0
    total = 0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        err_sum += n * abs(float(p[mask].mean()) - float(y[mask].mean()))
        total += n
    return float(err_sum / total) if total else float("nan")


def select_devig_method_from_folds(
    folds: list[dict],
    *,
    methods: list[str] | None = None,
    global_default: str = "multiplicative",
    shrink_min_n: int = 200,
    loss_fn=brier_score,
):
    """Choose a de-vig method for each chronological fold using only the
    Brier/log-loss/calibration performance accumulated over *strictly prior*
    folds. Never looks at the fold it is choosing a method for.

    Parameters
    ----------
    folds : chronologically ordered list of ``{"q": [...], "outcome": [...]}``
        where each element of ``q`` is an array-like of raw implied
        probabilities for one game/quote and the matching element of
        ``outcome`` is the realized 0/1 result for ``q[0]``'s side (e.g. home
        win / home cover).
    shrink_min_n : minimum accumulated prior-fold sample size before trusting
        a data-driven "best method" pick over ``global_default`` (hierarchical
        shrinkage to the global default when evidence is thin).

    Returns
    -------
    (chosen, fold_losses) : ``chosen[i]`` is the method used for ``folds[i]``
    (decided from ``folds[:i]`` only); ``fold_losses[i]`` is each method's
    realized OOS loss on ``folds[i]`` (recorded for diagnostics/next-fold
    history, but ``chosen[i]`` never depends on ``fold_losses[i]``).
    """
    if methods is None:
        methods = list(DEVIG_METHODS)
    history = {m: {"loss_sum": 0.0, "n": 0} for m in methods}
    chosen: list[str] = []
    fold_losses: list[dict] = []

    for fold in folds:
        n_prior = min(v["n"] for v in history.values())
        if n_prior < shrink_min_n:
            method_for_fold = global_default
        else:
            avg_loss = {
                m: (history[m]["loss_sum"] / history[m]["n"]) for m in methods
            }
            method_for_fold = min(avg_loss, key=avg_loss.get)
        chosen.append(method_for_fold)

        losses_this_fold = {}
        for m in methods:
            preds = []
            outs = []
            for q, outcome in zip(fold["q"], fold["outcome"]):
                p = devig(q, method=m)
                preds.append(float(p[0]))
                outs.append(float(outcome))
            loss = loss_fn(preds, outs) if preds else float("nan")
            losses_this_fold[m] = loss
            if preds:
                history[m]["loss_sum"] += loss * len(preds)
                history[m]["n"] += len(preds)
        fold_losses.append(losses_this_fold)

    return chosen, fold_losses


def store_devig_choice(
    store: dict,
    key: tuple,
    *,
    method: str,
    n_obs: int,
    global_default: str = "multiplicative",
    shrink_min_n: int = 200,
) -> str:
    """Record/shrink the chosen method for one ``(book, market, horizon)`` key.

    With fewer than ``shrink_min_n`` observations the stored choice is forced
    back to ``global_default`` regardless of what looked best on the (small,
    noisy) available sample — hierarchical shrinkage per Task 027.
    """
    effective = method if n_obs >= shrink_min_n else global_default
    store[key] = {"method": effective, "n_obs": int(n_obs), "requested": method}
    return effective
