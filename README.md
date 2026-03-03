# D-Vine Copula Dependency Model

A copula-based supervised learning model that captures higher-order dependency structures using a D-Vine construction to improve classification performance.

## Motivation

Traditional classifiers such as Naive Bayes assume conditional independence between features. 
This assumption often fails in real-world datasets where dependencies exist between predictors.

This project implements a D-Vine copula-based supervised learning model to explicitly model pairwise and conditional dependencies between features.

Inspired by:  
[A Copula-Based Supervised Learning Classification](https://doi.org/10.1080/15598608.2016.1278059)

## Methodology

Given features X₁, ..., X_d, the joint distribution is decomposed using a D-Vine structure:

f(x₁,...,x_d) = ∏ marginals × ∏ pair-copulas

Copula families are selected using BIC.

## Implementation Details

- Language: Python
- Copula families: Gaussian, Clayton, Gumbel
- Vine structure: D-Vine
- Model selection: Bayesian Information Criterion (BIC)
- Marginal estimation: Kernel Density Estimation

- ## Installation

```bash
git clone https://github.com/pierced07/D-Vine-Copula-Dependency-Model.git
cd D-Vine-Copula-Dependency-Model
pip install -r requirements.txt

## Project Structure

├── dvine/
│   ├── copula.py
│   ├── vine.py
│   ├── model_selection.py
├── experiments/
├── notebooks/
├── README.md

## Future Work

- Extend to C-Vine structure
- Parallelize copula selection
- Explore reinforcement learning integration
