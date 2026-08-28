"""Tests for the NLP preprocessing pipeline.

These cover the decisions that are easy to get wrong and expensive when they
are: negation survival, escape-sequence damage, and the difference between the
classification and topic-extraction stopword lists.
"""

from __future__ import annotations

import pytest

from src.preprocessing import (
    KEYWORD_STOPWORDS,
    STOPWORDS,
    clean_for_classical,
    clean_for_transformer,
    detect_issue_categories,
    expand_contractions,
    extract_keywords,
    simple_stem,
    squash_elongation,
    tokenize,
)


class TestNegationHandling:
    """The single most consequential preprocessing decision for sentiment."""

    @pytest.mark.parametrize("word", ["not", "no", "never", "cannot", "very"])
    def test_negations_and_intensifiers_are_not_stopwords(self, word):
        assert word not in STOPWORDS

    def test_negation_survives_aggressive_cleaning(self):
        assert "not" in clean_for_classical("This is not good at all").split()

    def test_contractions_expand_to_explicit_negation(self):
        assert "do not" in expand_contractions("I don't like it")
        assert "not" in clean_for_classical("I don't like it").split()

    def test_negated_and_plain_forms_differ(self):
        assert clean_for_classical("not good") != clean_for_classical("good")


class TestNoiseRemoval:
    def test_urls_handles_and_html_are_stripped(self):
        cleaned = clean_for_transformer(
            "Check <b>this</b> https://example.com/x @support now"
        )
        assert "https" not in cleaned
        assert "@support" not in cleaned
        assert "<b>" not in cleaned
        assert "Check" in cleaned

    def test_hashtag_keeps_the_word(self):
        assert "terrible" in clean_for_transformer("#terrible service").lower()

    def test_literal_escape_sequences_do_not_fuse_with_words(self):
        """Corpora that store newlines as backslash-n must not create "nthe"."""
        cleaned = clean_for_transformer(r"Great spot.\nThe food was cold")
        assert "nthe" not in cleaned.lower()
        assert "The food" in cleaned

    def test_email_is_removed(self):
        assert "@" not in clean_for_transformer("mail me at a.b@c.com please")


class TestCleaningProfiles:
    def test_transformer_profile_preserves_case_and_punctuation(self):
        cleaned = clean_for_transformer("This is GREAT, really!")
        assert "GREAT" in cleaned
        assert "!" in cleaned

    def test_classical_profile_lowercases_and_depunctuates(self):
        cleaned = clean_for_classical("This is GREAT, really!")
        assert cleaned == cleaned.lower()
        assert "!" not in cleaned and "," not in cleaned

    def test_elongation_is_squashed_to_a_marker(self):
        assert squash_elongation("sooooo goooood!!!!!") == "soo good!!"

    @pytest.mark.parametrize("text", ["", "   ", None, 123])
    def test_degenerate_input_returns_empty_string(self, text):
        assert clean_for_transformer(text) == ""
        assert clean_for_classical(text) == ""


class TestStemming:
    @pytest.mark.parametrize(
        ("token", "expected"),
        [("shipping", "ship"), ("delivered", "deliver"), ("replies", "reply")],
    )
    def test_common_inflections_collapse(self, token, expected):
        assert simple_stem(token) == expected

    @pytest.mark.parametrize("token", ["good", "bad", "slow", "poor"])
    def test_short_sentiment_words_are_left_alone(self, token):
        assert simple_stem(token) == token


class TestIssueCategories:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("My package arrived three days late", "Delivery & Logistics"),
            ("The agent was rude and unhelpful", "Customer Support"),
            ("Completely overpriced for what you get", "Pricing & Value"),
            ("The app crashes on the checkout screen", "Usability & UX"),
        ],
    )
    def test_dominant_category_is_detected(self, text, expected):
        assert expected in detect_issue_categories(text)

    def test_multiple_topics_are_all_tagged(self):
        categories = detect_issue_categories(
            "Delivery was late and the support agent was rude about the refund"
        )
        assert len(categories) >= 2

    def test_unrelated_text_yields_no_categories(self):
        assert detect_issue_categories("Hello there") == []

    def test_result_is_capped(self):
        text = "delivery quality support price app food clean wait"
        assert len(detect_issue_categories(text, max_categories=2)) <= 2


class TestKeywordExtraction:
    def test_function_words_are_excluded_from_topics(self):
        """Negations are features for the model but noise in a word cloud."""
        words = dict(
            extract_keywords(["the delivery was not good but the food was fine"] * 5)
        )
        assert "not" not in words and "but" not in words
        assert "delivery" in words

    def test_negations_stay_available_to_the_classifier(self):
        assert "not" in KEYWORD_STOPWORDS and "not" not in STOPWORDS

    def test_url_fragments_never_become_keywords(self):
        words = dict(extract_keywords(["see https://example.com/page for details"] * 3))
        assert not {"https", "com", "www"} & set(words)

    def test_counts_are_accurate_and_ranked(self):
        results = extract_keywords(["refund refund delivery"], top_n=5)
        assert results[0] == ("refund", 2)

    def test_top_n_is_respected(self):
        assert len(extract_keywords(["alpha beta gamma delta epsilon"], top_n=3)) == 3


class TestTokenizer:
    def test_tokens_are_lowercased_words_only(self):
        assert tokenize("Hello, World! 123") == ["hello", "world"]

    def test_empty_input_is_safe(self):
        assert tokenize("") == [] and tokenize(None) == []
