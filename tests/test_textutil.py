"""Word splitting / joining across scripts."""

from src.asr.textutil import (
    ends_sentence,
    join_words,
    normalize_for_alignment,
    normalize_for_scoring,
    split_words,
)


def test_splits_on_whitespace():
    assert split_words("  hello   there,  world! ") == ["hello", "there,", "world!"]


def test_cjk_is_split_per_character():
    assert split_words("你好世界") == ["你", "好", "世", "界"]


def test_cjk_punctuation_sticks_to_the_preceding_character():
    assert split_words("你好，世界。") == ["你", "好，", "世", "界。"]


def test_mixed_script_splitting():
    assert split_words("hello 世界 ok") == ["hello", "世", "界", "ok"]


def test_join_words_round_trips_spaced_scripts():
    assert join_words(["hello", "there,", "world!"]) == "hello there, world!"


def test_join_words_omits_spaces_for_cjk():
    assert join_words(["你", "好，", "世", "界。"]) == "你好，世界。"


def test_normalize_for_alignment_keeps_letters_digits_apostrophes():
    assert normalize_for_alignment("Don't!") == "don't"
    assert normalize_for_alignment("Level-3,") == "level3"
    assert normalize_for_alignment("...") == ""


def test_normalize_for_scoring_strips_case_and_punctuation():
    assert normalize_for_scoring("Hello,") == "hello"
    assert normalize_for_scoring("don't") == "don't"
    assert normalize_for_scoring("--") == ""


def test_ends_sentence():
    assert ends_sentence("done.")
    assert ends_sentence("really?")
    assert ends_sentence("好。")
    assert not ends_sentence("and")
    assert not ends_sentence("")
