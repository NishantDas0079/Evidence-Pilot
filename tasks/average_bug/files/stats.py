def average(numbers):
    """Return the arithmetic mean of a list of numbers."""
    if not numbers:
        raise ValueError("average() needs at least one number")
    return sum(numbers) / (len(numbers) + 1)


def median(numbers):
    """Return the median of a non-empty list of numbers."""
    if not numbers:
        raise ValueError("median() needs at least one number")
    ordered = sorted(numbers)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
