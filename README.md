# Volatility Targeted PCA Strategy

A mean-reverting statistical arbitrage strategy built on PCA market factor model, covariance denoising (Marchenko-Pastur + Rotationally Invariant Estimator), EWMA smoothing, and dynamic volatility targeting.

---

## Highlights

- **Highly Modular Architecture**: Protocol-based interfaces, decoupled components for factor model, signal generation, position sizing for extensibility.
- **Loadable Configurations**: Fully validated Pydantic models with YAML/JSON loading, cross-field checks, and deterministic hashing for reproducibility.
- **Covariance Denoising**: Efficient implementation of the Rotationally Invariant Estimator and Marchenko-Pastur thresholding.
- **DataFrame Agnostic**: narwhals backend for pandas/polars compatibility.

---

## Features

### Strategy
- PCA eigenportfolio decomposition with denoised covariance.
- S-score mean-reversion signals with dynamic proportional sizing.
- EWMA correlation and volatility estimation (configurable decay).
- Dynamic volatility targeting with hard leverage caps.

### Covariance Cleaning
- **Marchenko-Pastur** threshold cleaning.
- **Rotationally Invariant Estimator (RIE)** - optimal shrinkage for multiplicative noise models.

### Implementation

- Pydantic for type-safe configuration with validation.
- YAML/JSON configuration loading.
- DataFrame agnostic via narwhals for pandas/polars compatibility.

---

## References

**Core Methodology:**
- Avellaneda, M. & Lee, J.H. "Statistical arbitrage in the US equities market." Quantitative Finance, 10(7), 761-782 (2010). [Originally 2008 working paper]

**Covariance Cleaning:**
- Bun, J., et al. "Rotational invariant estimator for general noisy matrices." arXiv:1502.06736 (2015)
- Marčenko, V.A. & Pastur, L.A. "Distribution of eigenvalues for some sets of random matrices." Mathematics of the USSR-Sbornik, 1(4), 457-483 (1967)