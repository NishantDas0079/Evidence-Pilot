LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]


def classification_report(gold: list[str], pred: list[str], labels=LABELS) -> dict:
    """Per-class precision/recall/F1, accuracy and a confusion matrix.

    Predictions outside `labels` (for example "ERROR") count as wrong and never as a hit.
    """
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    per_class = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    columns = labels + ["other"]
    confusion = {g: {c: 0 for c in columns} for g in labels}
    for g, p in zip(gold, pred):
        if g in confusion:
            confusion[g][p if p in labels else "other"] += 1
    correct = sum(g == p for g, p in zip(gold, pred))
    return {
        "per_class": per_class,
        "confusion": confusion,
        "accuracy": correct / len(gold) if gold else 0.0,
        "n": len(gold),
    }
