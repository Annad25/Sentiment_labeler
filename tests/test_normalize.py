from sentiment_eval.classifier import UNPARSEABLE, normalize


def test_clean_words():
    assert normalize("positive") == "positive"
    assert normalize("neutral") == "neutral"
    assert normalize("negative") == "negative"


def test_capitalisation_and_punctuation():
    assert normalize("Positive!") == "positive"
    assert normalize("NEGATIVE.") == "negative"
    assert normalize("  Neutral  ") == "neutral"


def test_full_sentence_outputs():
    assert normalize("The sentiment is positive.") == "positive"
    assert normalize("I think this is clearly negative.") == "negative"
    assert normalize("This snippet seems neutral overall.") == "neutral"


def test_empty_and_garbage():
    assert normalize("") == UNPARSEABLE
    assert normalize("???") == UNPARSEABLE
    assert normalize("happy") == UNPARSEABLE  # synonym, not in enum — we refuse to guess


def test_first_match_wins():
    # If the model contradicts itself we take the first keyword — predictable
    # and easy to defend.
    assert normalize("positive, actually no, negative") == "positive"
