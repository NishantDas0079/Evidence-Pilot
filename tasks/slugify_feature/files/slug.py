def slugify(text, max_length=None):
    """Turn text into a URL slug.

    Rules: lowercase; letters and digits are kept; accented letters become their
    plain ASCII form; every other run of characters becomes a single hyphen; no
    leading or trailing hyphen. If max_length is given, cut the slug to at most
    that many characters without leaving a trailing hyphen.
    """
    raise NotImplementedError
