"""
Tests for the difficulty classifier.

All tests run on CPU in milliseconds — no model, no GPU required.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agentserve.engine.difficulty import RequestDifficultyClassifier, DifficultyLevel


@pytest.fixture
def clf():
    return RequestDifficultyClassifier()


# ---------------------------------------------------------------------------
# Easy prompts
# ---------------------------------------------------------------------------

class TestEasyClassification:
    def test_classify_keyword(self, clf):
        d = clf.classify("Classify this sentence as positive or negative.")
        assert d.level == DifficultyLevel.EASY

    def test_yes_or_no_keyword(self, clf):
        d = clf.classify("Is this sentence grammatically correct? Answer yes or no.")
        assert d.level == DifficultyLevel.EASY

    def test_true_or_false_keyword(self, clf):
        d = clf.classify("True or false: the sky is green.")
        assert d.level == DifficultyLevel.EASY

    def test_label_keyword(self, clf):
        d = clf.classify("Label the following ticket as BUG, FEATURE, or QUESTION.")
        assert d.level == DifficultyLevel.EASY

    def test_extract_keyword(self, clf):
        d = clf.classify("Extract the version number from: 'Using torch==2.1.0'. Reply with the version only.")
        assert d.level == DifficultyLevel.EASY

    def test_json_schema_keyword(self, clf):
        d = clf.classify('Fill in the JSON schema: {"name": "", "version": ""}')
        assert d.level == DifficultyLevel.EASY

    def test_very_short_prompt(self, clf):
        d = clf.classify("Hi")
        assert d.level == DifficultyLevel.EASY

    def test_easy_estimated_output_tokens(self, clf):
        d = clf.classify("Yes or no: is water wet?")
        assert d.estimated_output_tokens == 20
        assert d.priority == 0


# ---------------------------------------------------------------------------
# Hard prompts
# ---------------------------------------------------------------------------

class TestHardClassification:
    def test_write_a_function(self, clf):
        d = clf.classify("Write a function in Python that implements binary search.")
        assert d.level == DifficultyLevel.HARD

    def test_implement_keyword(self, clf):
        d = clf.classify("Implement a binary search tree in Python.")
        assert d.level == DifficultyLevel.HARD

    def test_debug_keyword(self, clf):
        d = clf.classify("Debug this code and explain all the issues.")
        assert d.level == DifficultyLevel.HARD

    def test_long_prompt_is_hard(self, clf):
        # 600 repetitions × 3 words = 1800 words → ~2400 tokens (threshold: 2000)
        long_prompt = "explain this concept. " * 600
        d = clf.classify(long_prompt)
        assert d.level == DifficultyLevel.HARD

    def test_hard_estimated_output_tokens(self, clf):
        d = clf.classify("Write a function that implements merge sort.")
        assert d.estimated_output_tokens == 256
        assert d.priority == 2

    def test_write_tests(self, clf):
        d = clf.classify("Write unit tests for the following module.")
        assert d.level == DifficultyLevel.HARD


# ---------------------------------------------------------------------------
# Medium prompts
# ---------------------------------------------------------------------------

class TestMediumClassification:
    def test_generic_question(self, clf):
        d = clf.classify("What are the main differences between REST and GraphQL?")
        assert d.level == DifficultyLevel.MEDIUM

    def test_summary_request(self, clf):
        d = clf.classify("Summarize the following paragraph in two sentences.")
        assert d.level == DifficultyLevel.MEDIUM

    def test_medium_priority(self, clf):
        d = clf.classify("Explain what dependency injection means.")
        assert d.priority == 1
        assert d.estimated_output_tokens == 100


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

class TestPriorityOrder:
    def test_easy_has_lowest_priority_number(self, clf):
        easy   = clf.classify("Classify this as positive or negative.")
        medium = clf.classify("Explain dependency injection in 3 sentences.")
        hard   = clf.classify("Write a function that implements quicksort.")
        assert easy.priority < medium.priority < hard.priority

    def test_estimated_output_tokens_ordering(self, clf):
        easy   = clf.classify("Yes or no?")
        medium = clf.classify("Explain in 3 sentences.")
        hard   = clf.classify("Write a function to implement merge sort.")
        assert easy.estimated_output_tokens < medium.estimated_output_tokens < hard.estimated_output_tokens
