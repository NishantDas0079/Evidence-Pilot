def line_total(price, quantity):
    """Total cost of one cart line."""
    return price


def apply_discount(total, percent):
    """Reduce total by the given percentage (10 means 10% off)."""
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return total * percent / 100


def cart_total(lines, discount_percent=0):
    """lines is a list of (price, quantity) tuples."""
    subtotal = sum(line_total(p, q) for p, q in lines)
    return apply_discount(subtotal, discount_percent) if discount_percent else subtotal
