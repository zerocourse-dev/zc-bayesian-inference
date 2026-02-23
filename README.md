# Bayesian Inference Engine

A ZeroCourse project for Q13: Mathematics for Machine Learning (Week 6).

## What You'll Build

Implement a Bayesian inference engine that applies Bayes' theorem to real-world problems: medical diagnosis, naive Bayes classification, and parameter estimation via grid approximation.

### Classes and Methods

| Component | Description |
|-----------|-------------|
| `BayesTheorem.posterior(prior, likelihood, false_positive)` | Compute P(H\|E) using Bayes' theorem |
| `BayesTheorem.sequential_update(prior, observations)` | Update beliefs with multiple observations |
| `BayesTheorem.prior_sensitivity(priors, likelihood, false_positive)` | Show how posterior varies with prior |
| `MedicalDiagnostic(sensitivity, specificity, prevalence)` | Medical test interpreter |
| `MedicalDiagnostic.positive_predictive_value()` | P(disease \| positive test) |
| `MedicalDiagnostic.negative_predictive_value()` | P(no disease \| negative test) |
| `MedicalDiagnostic.interpret()` | Summary dict of all diagnostic values |
| `NaiveBayes(smoothing)` | Naive Bayes classifier with Laplace smoothing |
| `NaiveBayes.fit(labels, features)` | Train on labeled bag-of-words data |
| `NaiveBayes.predict(features)` | Predict most likely class |
| `NaiveBayes.predict_proba(features)` | Predict class probabilities |
| `GridApproximation(grid_size)` | Bayesian parameter estimation |
| `GridApproximation.uniform_prior()` | Uniform prior over grid |
| `GridApproximation.beta_prior(alpha, beta)` | Beta-shaped prior |
| `GridApproximation.binomial_likelihood(heads, tails)` | Coin flip likelihood |
| `GridApproximation.compute_posterior(prior, likelihood)` | Normalize prior * likelihood |
| `GridApproximation.estimate_coin_bias(heads, tails)` | Full coin bias estimation |
| `GridApproximation.credible_interval(posterior, confidence)` | 95% credible interval |

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the tests (they will all fail initially):
   ```bash
   python -m pytest tests/ --tb=short
   ```

3. Open `lib/bayesian_inference.py` and implement each class.

4. Run the tests again to check your progress.

## Suggested Implementation Order

1. **`BayesTheorem.posterior`** — the core formula, everything builds on this
2. **`BayesTheorem.sequential_update`** — apply posterior repeatedly
3. **`BayesTheorem.prior_sensitivity`** — loop over priors
4. **`MedicalDiagnostic`** — wraps BayesTheorem for medical context
5. **`NaiveBayes.fit`** — count words per class, compute log-probabilities
6. **`NaiveBayes.predict`** — sum log-probabilities, pick max
7. **`NaiveBayes.predict_proba`** — convert log-probs to probabilities
8. **`GridApproximation`** — `uniform_prior` first, then `binomial_likelihood`, then `compute_posterior`
9. **`estimate_coin_bias`** and **`credible_interval`** — tie it all together

## Tips

- **Bayes' formula:** `P(H|E) = P(E|H) * P(H) / P(E)` where `P(E) = P(E|H)*P(H) + P(E|~H)*(1-P(H))`
- **Medical PPV:** prior = prevalence, likelihood = sensitivity, false_positive = 1 - specificity
- **Medical NPV:** Think of it as "what's the probability I'm healthy given a negative test?" — flip the perspective
- **Naive Bayes:** Work in log-space to avoid floating-point underflow. Use `math.log`.
- **Laplace smoothing:** `P(word|class) = (count + smoothing) / (total_words_in_class + smoothing * vocab_size)`
- **Grid approximation:** Create a grid of values from 0 to 1, evaluate likelihood at each point, multiply by prior, normalize
- **Credible interval:** Sort cumulative probability, find the narrowest range containing 95% of the mass

## Running Tests

```bash
python -m pytest tests/                            # Run all tests
python -m pytest tests/ -v                         # Verbose output
python -m pytest tests/ -k "TestBayes"             # Run only Bayes theorem tests
python -m pytest tests/ -k "TestNaiveBayes"        # Run only NaiveBayes tests
python -m pytest tests/ -k "TestCoinBias"          # Run only coin bias tests
```
