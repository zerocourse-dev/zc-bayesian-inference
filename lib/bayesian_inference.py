# Bayesian Inference Engine
#
# Implement Bayes' theorem and apply it to real-world problems:
# medical diagnosis, naive Bayes classification, and parameter estimation
# via grid approximation.
#
# Hint: Start with BayesTheorem (basic computation), then MedicalDiagnostic,
# then NaiveBayes, then GridApproximation.

import math


# ── Core Bayes' Theorem ──────────────────────────────────────────────────

class BayesTheorem:
    """Apply Bayes' theorem to compute posterior probabilities.

    P(H|E) = P(E|H) * P(H) / P(E)
    where P(E) = P(E|H)*P(H) + P(E|~H)*P(~H)

    Examples:
        >>> bt = BayesTheorem()
        >>> bt.posterior(prior=0.01, likelihood=0.95, false_positive=0.05)
        0.16101...
    """

    def posterior(self, prior, likelihood, false_positive):
        """Compute posterior probability using Bayes' theorem.

        P(H|E) = P(E|H) * P(H) / P(E)
        P(E) = P(E|H)*P(H) + P(E|~H)*P(~H)

        @param prior: float — P(H), prior probability of hypothesis
        @param likelihood: float — P(E|H), probability of evidence given hypothesis
        @param false_positive: float — P(E|~H), probability of evidence given NOT hypothesis
        @return: float — P(H|E), posterior probability

        Examples:
            >>> bt = BayesTheorem()
            >>> bt.posterior(prior=0.5, likelihood=1.0, false_positive=0.0)
            1.0
        """
        raise NotImplementedError("Implement BayesTheorem.posterior")

    def sequential_update(self, prior, observations):
        """Update beliefs sequentially with multiple observations.

        Each observation is a dict with 'likelihood' and 'false_positive' keys.
        Apply Bayes' theorem repeatedly, using each posterior as the next prior.

        @param prior: float — initial prior probability
        @param observations: list of dicts, each with 'likelihood' and 'false_positive'
        @return: dict with 'final_posterior' (float) and 'posteriors' (list of all intermediate posteriors)

        Examples:
            >>> bt = BayesTheorem()
            >>> obs = [{'likelihood': 0.9, 'false_positive': 0.1}]
            >>> result = bt.sequential_update(0.5, obs)
            >>> result['final_posterior']
            0.9
        """
        raise NotImplementedError("Implement BayesTheorem.sequential_update")

    def prior_sensitivity(self, priors, likelihood, false_positive):
        """Show how posterior changes across different prior values.

        @param priors: list of float — different prior values to test
        @param likelihood: float — P(E|H)
        @param false_positive: float — P(E|~H)
        @return: list of dicts, each with 'prior' and 'posterior' keys

        Examples:
            >>> bt = BayesTheorem()
            >>> results = bt.prior_sensitivity([0.01, 0.5, 0.99], 0.9, 0.1)
            >>> len(results)
            3
        """
        raise NotImplementedError("Implement BayesTheorem.prior_sensitivity")


# ── Medical Diagnostic ───────────────────────────────────────────────────

class MedicalDiagnostic:
    """Apply Bayes' theorem to medical test interpretation.

    Given a test's sensitivity and specificity plus disease prevalence,
    compute the positive/negative predictive values.

    Examples:
        >>> md = MedicalDiagnostic(sensitivity=0.99, specificity=0.95, prevalence=0.001)
        >>> md.positive_predictive_value()
        0.019...
    """

    def __init__(self, sensitivity, specificity, prevalence):
        """Store test characteristics.

        @param sensitivity: float — P(positive | disease), true positive rate
        @param specificity: float — P(negative | no disease), true negative rate
        @param prevalence: float — P(disease), base rate of disease
        """
        raise NotImplementedError("Implement MedicalDiagnostic.__init__")

    def positive_predictive_value(self):
        """Compute P(disease | positive test).

        Also known as PPV. Uses Bayes' theorem where:
        - prior = prevalence
        - likelihood = sensitivity
        - false_positive = 1 - specificity

        @return: float — probability of disease given positive test
        """
        raise NotImplementedError("Implement MedicalDiagnostic.positive_predictive_value")

    def negative_predictive_value(self):
        """Compute P(no disease | negative test).

        Uses Bayes' theorem where:
        - prior = 1 - prevalence (probability of no disease)
        - likelihood = specificity (P(negative | no disease))
        - false_positive = 1 - sensitivity (P(negative | disease))

        @return: float — probability of no disease given negative test
        """
        raise NotImplementedError("Implement MedicalDiagnostic.negative_predictive_value")

    def interpret(self):
        """Return a summary dict of all diagnostic values.

        @return: dict with keys: 'ppv', 'npv', 'sensitivity', 'specificity', 'prevalence'
        """
        raise NotImplementedError("Implement MedicalDiagnostic.interpret")


# ── Naive Bayes Classifier ───────────────────────────────────────────────

class NaiveBayes:
    """Naive Bayes classifier for categorical features.

    Assumes features are conditionally independent given the class label.
    Uses Laplace smoothing to handle unseen feature values.

    Examples:
        >>> nb = NaiveBayes(smoothing=1.0)
        >>> nb.fit(['spam', 'ham', 'spam'], [['buy', 'now'], ['hello', 'friend'], ['buy', 'free']])
        >>> nb.predict(['buy', 'stuff'])
        'spam'
    """

    def __init__(self, smoothing=1.0):
        """Initialize with smoothing parameter.

        @param smoothing: float — Laplace smoothing constant (default 1.0)
        """
        raise NotImplementedError("Implement NaiveBayes.__init__")

    def fit(self, labels, features):
        """Train the classifier on labeled data.

        For each class, compute:
        - P(class) = count(class) / total
        - P(word | class) = (count(word in class) + smoothing) / (total words in class + smoothing * vocab_size)

        @param labels: list of str — class labels
        @param features: list of lists of str — feature values (bag of words)
        @return: self (for chaining)
        """
        raise NotImplementedError("Implement NaiveBayes.fit")

    def predict(self, features):
        """Predict the class label for a feature set.

        Compute log P(class) + sum(log P(feature | class)) for each class.
        Return the class with highest score.

        @param features: list of str — feature values
        @return: str — predicted class label
        """
        raise NotImplementedError("Implement NaiveBayes.predict")

    def predict_proba(self, features):
        """Predict class probabilities for a feature set.

        Compute log-probabilities, convert to probabilities, normalize to sum to 1.

        @param features: list of str — feature values
        @return: dict mapping class labels to probabilities
        """
        raise NotImplementedError("Implement NaiveBayes.predict_proba")


# ── Grid Approximation ──────────────────────────────────────────────────

class GridApproximation:
    """Bayesian parameter estimation using grid approximation.

    Estimates the posterior distribution over a parameter (e.g., coin bias)
    by evaluating the likelihood at many discrete grid points.

    Examples:
        >>> ga = GridApproximation(grid_size=101)
        >>> result = ga.estimate_coin_bias(heads=7, tails=3)
        >>> 0.6 < result['map_estimate'] < 0.8
        True
    """

    def __init__(self, grid_size=101):
        """Initialize with grid resolution.

        @param grid_size: int — number of points in the parameter grid
        """
        raise NotImplementedError("Implement GridApproximation.__init__")

    def uniform_prior(self):
        """Return a uniform prior over the grid.

        @return: list of float — equal probabilities summing to 1
        """
        raise NotImplementedError("Implement GridApproximation.uniform_prior")

    def beta_prior(self, alpha, beta_param):
        """Return a Beta-distribution-shaped prior over the grid.

        Uses the Beta PDF: p^(alpha-1) * (1-p)^(beta-1), then normalizes.

        @param alpha: float — alpha parameter of Beta distribution
        @param beta_param: float — beta parameter of Beta distribution
        @return: list of float — prior probabilities summing to 1
        """
        raise NotImplementedError("Implement GridApproximation.beta_prior")

    def binomial_likelihood(self, heads, tails):
        """Compute likelihood of observing heads/tails for each grid point.

        For grid point p: likelihood = p^heads * (1-p)^tails

        @param heads: int — number of heads observed
        @param tails: int — number of tails observed
        @return: list of float — likelihood at each grid point
        """
        raise NotImplementedError("Implement GridApproximation.binomial_likelihood")

    def compute_posterior(self, prior, likelihood):
        """Compute posterior = prior * likelihood, then normalize.

        @param prior: list of float — prior probabilities
        @param likelihood: list of float — likelihood at each grid point
        @return: list of float — posterior probabilities summing to 1
        """
        raise NotImplementedError("Implement GridApproximation.compute_posterior")

    def estimate_coin_bias(self, heads, tails, prior=None):
        """Estimate coin bias from observed flips.

        Uses uniform prior if none provided.

        @param heads: int — number of heads
        @param tails: int — number of tails
        @param prior: list of float or None — prior distribution
        @return: dict with keys: 'grid', 'posterior', 'map_estimate', 'mean_estimate',
                 'credible_interval_95'
        """
        raise NotImplementedError("Implement GridApproximation.estimate_coin_bias")

    def credible_interval(self, posterior, confidence=0.95):
        """Compute the highest-density credible interval.

        Find the narrowest interval containing at least `confidence` probability mass.
        Uses the percentile method: find lower and upper quantiles.

        @param posterior: list of float — posterior probabilities
        @param confidence: float — confidence level (default 0.95)
        @return: tuple (lower, upper) — grid values bounding the interval
        """
        raise NotImplementedError("Implement GridApproximation.credible_interval")
