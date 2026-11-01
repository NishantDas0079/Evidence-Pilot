from slug import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_punctuation_and_spaces_collapse():
    assert slugify("  Hello,   World!! ") == "hello-world"


def test_accents_become_ascii():
    assert slugify("Crème Brûlée") == "creme-brulee"


def test_digits_kept():
    assert slugify("Top 10 Tips") == "top-10-tips"


def test_empty_and_symbols_only():
    assert slugify("") == ""
    assert slugify("!!!") == ""


def test_max_length_cuts_cleanly():
    assert slugify("hello world again", max_length=8) == "hello-wo"


def test_max_length_never_leaves_trailing_hyphen():
    assert slugify("hello world", max_length=6) == "hello"
