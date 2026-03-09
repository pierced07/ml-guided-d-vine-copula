# ML-Guided D-Vine Copula Model

A hybrid statistical-machine learning framework that uses **meta-learning to optimize copula family selection** in D-Vine structures, reducing computational cost while maintaining modeling accuracy.

## Motivation

Traditional classifiers such as Naive Bayes assume conditional independence between features. This assumption often fails in real-world datasets where dependencies exist between predictors.

Copula-based models address this by explicitly modeling dependency structures. However, standard Vine Copula implementations require **exhaustive fitting** of multiple copula families per pair, which becomes computationally prohibitive in high dimensions.

**This project introduces a machine learning-based selection mechanism** that predicts optimal copula families from data characteristics, eliminating the need for exhaustive fitting.

Inspired by:  
[A Copula-Based Supervised Learning Classification](https://doi.org/10.1080/15598608.2016.1278059) by Dr. Yuhui Chen. 

## Key Innovation

| Traditional Approach | ML-Guided Approach |
|---------------------|-------------------|
| Fit 5-10 copula families per pair | Train classifier once, predict directly |
| Select via AIC/BIC comparison | Select via meta-feature classification |
| O(n × k) fits per pair | O(1) prediction per pair |
| Computationally expensive | Efficient inference |

## Methodology

### D-Vine Structure
Given features X₁, ..., Xₔ, the joint distribution is decomposed using a D-Vine structure:


### ML-Based Copula Selection
1. **Generate synthetic training data** with known copula families (Gaussian, Clayton etc.)
2. **Extract meta-features** from each dataset:
   - Kendall's τ / Spearman's ρ
   - Upper/Lower tail dependence coefficients
   - Symmetry measures
   - Higher-order moments (skewness, kurtosis)
3. **Train a classifier** (Decision Tree / Random Forest) to map meta-features → copula family
4. **Apply trained selector** to real data for fast family prediction

## Implementation Details

- **Language:** Python 3.8+
- **Copula families:** Gaussian, Clayton, Student-t
- **Vine structure:** D-Vine
- **Selection method:** ML classifier (sklearn) + BIC fallback
- **Marginal estimation:** Kernel Density Estimation / Empirical CDF
- **Key libraries:** `numpy`, `scipy`, `scikit-learn`, `statsmodels`

## Installation

```bash
git clone https://github.com/pierced07/ml-guided-d-vine-copula.git
```
```bash
cd ml-guided-d-vine-copula
```
```bash
pip install -r requirements.txt
```

### To-Do
- [X] Organize repository
- [ ] Refactor code into src/ for modularity
- [ ] Implement classification tree