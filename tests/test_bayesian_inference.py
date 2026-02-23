"""Tests for Bayesian Inference Engine.

Students: DO NOT modify this file. Your task is to make all these tests pass
by implementing the functions and classes in lib/bayesian_inference.py.
"""

import math
import pytest

from lib.bayesian_inference import (
    BayesTheorem,
    MedicalDiagnostic,
    NaiveBayes,
    GridApproximation,
)


# ── BayesTheorem ─────────────────────────────────────────────────────────

class TestBayesPosterior:
    def test_equal_priors(self):
        bt = BayesTheorem()
        result = bt.posterior(prior=0.5, likelihood=0.9, false_positive=0.1)
        assert abs(result - 0.9) < 1e-10

    def test_rare_disease(self):
        bt = BayesTheorem()
        result = bt.posterior(prior=0.01, likelihood=0.95, false_positive=0.05)
        # P(E) = 0.95*0.01 + 0.05*0.99 = 0.059
        expected = (0.95 * 0.01) / (0.95 * 0.01 + 0.05 * 0.99)
        assert abs(result - expected) < 1e-10

    def test_certain_evidence(self):
        bt = BayesTheorem()
        result = bt.posterior(prior=0.5, likelihood=1.0, false_positive=0.0)
        assert abs(result - 1.0) < 1e-10

    def test_no_evidence(self):
        """When likelihood equals false_positive, posterior equals prior."""
        bt = BayesTheorem()
        result = bt.posterior(prior=0.3, likelihood=0.5, false_positive=0.5)
        assert abs(result - 0.3) < 1e-10


class TestSequentialUpdate:
    def test_single_observation(self):
        bt = BayesTheorem()
        obs = [{"likelihood": 0.9, "false_positive": 0.1}]
        result = bt.sequential_update(0.5, obs)
        assert abs(result["final_posterior"] - 0.9) < 1e-10
        assert len(result["posteriors"]) == 1

    def test_two_observations_increase_confidence(self):
        bt = BayesTheorem()
        obs = [
            {"likelihood": 0.9, "false_positive": 0.1},
            {"likelihood": 0.9, "false_positive": 0.1},
        ]
        result = bt.sequential_update(0.5, obs)
        assert result["final_posterior"] > 0.9

    def test_posteriors_list_length(self):
        bt = BayesTheorem()
        obs = [{"likelihood": 0.8, "false_positive": 0.2}] * 5
        result = bt.sequential_update(0.5, obs)
        assert len(result["posteriors"]) == 5


class TestPriorSensitivity:
    def test_returns_correct_count(self):
        bt = BayesTheorem()
        results = bt.prior_sensitivity([0.01, 0.1, 0.5, 0.9], 0.9, 0.1)
        assert len(results) == 4

    def test_higher_prior_higher_posterior(self):
        bt = BayesTheorem()
        results = bt.prior_sensitivity([0.01, 0.5, 0.99], 0.9, 0.1)
        posteriors = [r["posterior"] for r in results]
        for i in range(len(posteriors) - 1):
            assert posteriors[i] < posteriors[i + 1]

    def test_contains_prior_and_posterior(self):
        bt = BayesTheorem()
        results = bt.prior_sensitivity([0.5], 0.9, 0.1)
        assert "prior" in results[0]
        assert "posterior" in results[0]


# ── MedicalDiagnostic ────────────────────────────────────────────────────

class TestMedicalDiagnostic:
    def test_ppv_rare_disease(self):
        """For rare diseases with imperfect tests, PPV is surprisingly low."""
        md = MedicalDiagnostic(sensitivity=0.99, specificity=0.95, prevalence=0.001)
        ppv = md.positive_predictive_value()
        assert ppv < 0.05  # less than 5% despite 99% sensitivity

    def test_npv_rare_disease(self):
        """NPV should be very high for rare diseases."""
        md = MedicalDiagnostic(sensitivity=0.99, specificity=0.95, prevalence=0.001)
        npv = md.negative_predictive_value()
        assert npv > 0.999

    def test_ppv_common_disease(self):
        """PPV should be higher when disease is common."""
        md = MedicalDiagnostic(sensitivity=0.99, specificity=0.95, prevalence=0.5)
        ppv = md.positive_predictive_value()
        assert ppv > 0.9

    def test_interpret_returns_all_keys(self):
        md = MedicalDiagnostic(sensitivity=0.99, specificity=0.95, prevalence=0.01)
        result = md.interpret()
        expected_keys = {"ppv", "npv", "sensitivity", "specificity", "prevalence"}
        assert set(result.keys()) == expected_keys

    def test_perfect_test(self):
        md = MedicalDiagnostic(sensitivity=1.0, specificity=1.0, prevalence=0.01)
        assert abs(md.positive_predictive_value() - 1.0) < 1e-10
        assert abs(md.negative_predictive_value() - 1.0) < 1e-10


# ── NaiveBayes ───────────────────────────────────────────────────────────

class TestNaiveBayes:
    def test_predict_simple(self):
        nb = NaiveBayes(smoothing=1.0)
        labels = ["spam", "spam", "ham", "ham"]
        features = [["buy", "now"], ["free", "money"], ["hello", "friend"], ["hi", "there"]]
        nb.fit(labels, features)
        prediction = nb.predict(["buy", "free"])
        assert prediction == "spam"

    def test_predict_ham(self):
        nb = NaiveBayes(smoothing=1.0)
        labels = ["spam", "spam", "ham", "ham"]
        features = [["buy", "now"], ["free", "money"], ["hello", "friend"], ["hi", "there"]]
        nb.fit(labels, features)
        prediction = nb.predict(["hello", "hi"])
        assert prediction == "ham"

    def test_predict_proba_sums_to_one(self):
        nb = NaiveBayes(smoothing=1.0)
        labels = ["a", "b", "a"]
        features = [["x"], ["y"], ["x"]]
        nb.fit(labels, features)
        proba = nb.predict_proba(["x"])
        total = sum(proba.values())
        assert abs(total - 1.0) < 1e-10

    def test_predict_proba_keys_match_classes(self):
        nb = NaiveBayes(smoothing=1.0)
        labels = ["cat", "dog", "cat"]
        features = [["meow"], ["bark"], ["purr"]]
        nb.fit(labels, features)
        proba = nb.predict_proba(["meow"])
        assert set(proba.keys()) == {"cat", "dog"}

    def test_fit_returns_self(self):
        nb = NaiveBayes()
        result = nb.fit(["a"], [["x"]])
        assert result is nb

    def test_smoothing_handles_unseen_words(self):
        """With smoothing, unseen words should not cause zero probabilities."""
        nb = NaiveBayes(smoothing=1.0)
        labels = ["spam", "ham"]
        features = [["buy"], ["hello"]]
        nb.fit(labels, features)
        proba = nb.predict_proba(["unseen_word"])
        for p in proba.values():
            assert p > 0


# ── GridApproximation ────────────────────────────────────────────────────

class TestGridApproximation:
    def test_uniform_prior_sums_to_one(self):
        ga = GridApproximation(grid_size=101)
        prior = ga.uniform_prior()
        assert abs(sum(prior) - 1.0) < 1e-10

    def test_uniform_prior_length(self):
        ga = GridApproximation(grid_size=51)
        prior = ga.uniform_prior()
        assert len(prior) == 51

    def test_beta_prior_sums_to_one(self):
        ga = GridApproximation(grid_size=101)
        prior = ga.beta_prior(2, 5)
        assert abs(sum(prior) - 1.0) < 1e-10

    def test_beta_prior_symmetric(self):
        """Beta(2,2) should be symmetric around 0.5."""
        ga = GridApproximation(grid_size=101)
        prior = ga.beta_prior(2, 2)
        # First and last should be equal (both at p near 0 and near 1)
        assert abs(prior[1] - prior[-2]) < 1e-10

    def test_binomial_likelihood_peak(self):
        """Likelihood should peak near the observed proportion."""
        ga = GridApproximation(grid_size=101)
        likelihood = ga.binomial_likelihood(heads=7, tails=3)
        max_idx = likelihood.index(max(likelihood))
        grid = [i / (101 - 1) for i in range(101)]
        assert abs(grid[max_idx] - 0.7) < 0.02


class TestCoinBias:
    def test_map_estimate_fair_coin(self):
        ga = GridApproximation(grid_size=101)
        result = ga.estimate_coin_bias(heads=50, tails=50)
        assert abs(result["map_estimate"] - 0.5) < 0.02

    def test_map_estimate_biased_coin(self):
        ga = GridApproximation(grid_size=101)
        result = ga.estimate_coin_bias(heads=80, tails=20)
        assert abs(result["map_estimate"] - 0.8) < 0.02

    def test_mean_estimate_exists(self):
        ga = GridApproximation(grid_size=101)
        result = ga.estimate_coin_bias(heads=5, tails=5)
        assert "mean_estimate" in result
        assert 0 < result["mean_estimate"] < 1

    def test_credible_interval_95(self):
        ga = GridApproximation(grid_size=101)
        result = ga.estimate_coin_bias(heads=50, tails=50)
        lower, upper = result["credible_interval_95"]
        assert lower < 0.5 < upper

    def test_result_keys(self):
        ga = GridApproximation(grid_size=101)
        result = ga.estimate_coin_bias(heads=5, tails=5)
        expected_keys = {"grid", "posterior", "map_estimate", "mean_estimate", "credible_interval_95"}
        assert set(result.keys()) == expected_keys

    def test_posterior_sums_to_one(self):
        ga = GridApproximation(grid_size=101)
        result = ga.estimate_coin_bias(heads=7, tails=3)
        assert abs(sum(result["posterior"]) - 1.0) < 1e-10

    def test_more_data_narrows_interval(self):
        ga = GridApproximation(grid_size=101)
        result_few = ga.estimate_coin_bias(heads=5, tails=5)
        result_many = ga.estimate_coin_bias(heads=50, tails=50)
        width_few = result_few["credible_interval_95"][1] - result_few["credible_interval_95"][0]
        width_many = result_many["credible_interval_95"][1] - result_many["credible_interval_95"][0]
        assert width_many < width_few


class TestCredibleInterval:
    def test_credible_interval_contains_map(self):
        ga = GridApproximation(grid_size=101)
        result = ga.estimate_coin_bias(heads=7, tails=3)
        lower, upper = result["credible_interval_95"]
        assert lower <= result["map_estimate"] <= upper


# ── Edge Cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_one_flip(self):
        ga = GridApproximation(grid_size=101)
        result = ga.estimate_coin_bias(heads=1, tails=0)
        assert result["map_estimate"] == 1.0

    def test_all_tails(self):
        ga = GridApproximation(grid_size=101)
        result = ga.estimate_coin_bias(heads=0, tails=10)
        assert result["map_estimate"] == 0.0
