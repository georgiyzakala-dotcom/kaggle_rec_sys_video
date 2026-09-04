"""Small backward-compatible helpers used by exploratory notebooks."""


def prec_k(pred: list[int], true: list[int], k: int) -> float:
    """Compute set-based Precision@k with the competition's fixed denominator."""

    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be a positive integer")
    if k <= 0:
        raise ValueError("k must be positive")
    if not pred or not true:
        return 0.0
    pred_set = set(pred[:k])
    true_set = set(true)
    return len(pred_set.intersection(true_set)) / k
