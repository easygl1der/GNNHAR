# Forecasting Realized Volatility with Spillover Effects: GNNHAR and Graph-Enhanced HAR Models

## Overview

This project extends the foundational work of Zhang Chao (published in *International Journal of Forecasting*) by applying graph-enhanced Heterogeneous Autoregressive (HAR) models to the Dow 30 stock universe. The primary contribution is a systematic investigation of whether Graph Neural Network HAR (GNNHAR) models exhibit greater performance advantages over linear Graph HAR (GHAR) models as the network size increases.

**Reference Repository:** [chaozhang-ox/GNNHAR](https://github.com/chaozhang-ox/GNNHAR) — the original implementation by Prof. Zhang Chao at HKUST Guangzhou (港科广)

**Supervision/Lab:** Based on the methodology of Prof. Zhang Chao's work at HKUST Guangzhou

---

## Research Motivation

The original Zhang Chao (2023) framework demonstrated that GNNHAR outperforms GHAR on S&P 100 data (100 assets). This project tests the **Network Scale Hypothesis**: whether the relative advantage of GNNHAR over GHAR increases with network size, due to neural networks' enhanced capacity to capture complex, nonlinear multi-hop interactions in larger graphs.

Two parallel experiments were conducted:
1. **Dow 30 (30 stocks)** — empirical study on actual market data
2. **Simulation (6 stocks)** — controlled environment testing scale effects

---

## Project Structure

```
GNNHAR/
├── src/gnnhar/
│   ├── linear.py            # Linear HAR/GHAR models with sklearn LinearRegression
│   ├── models.py            # PyTorch HAR/GHAR/GNNHAR1L/2L/3L models
│   └── mcs.py               # Model Confidence Set test (Hansen, Lunde & Nason)
├── scripts/
│   ├── compute_vol.py       # Compute daily variance from 5-minute return data
│   ├── data_subsample.py    # Subsample minutely LOBSTER data to 5-minute frequency
│   ├── summarize_results.py # Summarize MSE, QLIKE, and MCS test results
│   ├── summarize_regimes.py # Summarize results by market regimes
│   ├── plot_error_boxplots.py
│   └── simulation/          # Synthetic data generation and simulation runs
├── experiments/dow30/
│   ├── demo.py              # Main experiment script for Dow 30 analysis
│   ├── data/                # Dow 30 RV, IV, returns, and ticker-list data
│   ├── notebooks/           # Dow 30 exploratory notebooks
│   ├── results/             # Dow 30 result notes
│   └── assets/              # Experiment images and diagrams
├── data/mock/               # Six-stock simulation data and generated figures
├── reports/
│   ├── dow30/               # Latest Dow 30 report (Jan 2026)
│   └── simulation/          # Earlier simulation report
├── notebooks/               # General exploratory notebooks
└── GHAR.py / GNNHAR.py      # Backward-compatible wrappers for old commands
```

---

## Key Models

| Model | Description |
|-------|-------------|
| **HAR** | Heterogeneous Autoregressive baseline — captures daily/weekly/monthly volatility components |
| **GHAR** | Graph HAR — adds single graph convolution layer to capture cross-asset spillovers |
| **GNNHAR1L/2L/3L** | Multi-layer Graph Neural Network extensions (1, 2, or 3 GCN layers) with MLP readout |

**Adjacency Matrix:** Constructed via **Graphical Lasso (GLASSO)** on returns — captures sparse conditional dependence structure. Random adjacency matrices are used as robustness controls.

---

## Methodology Highlights

### HAR Features
- **Daily:** \(v_{t-1}\)
- **Weekly:** \(\frac{1}{4}\sum_{k=2}^{5}v_{t-k}\)
- **Monthly:** \(\frac{1}{17}\sum_{k=6}^{22}v_{t-k}\)

### Graph Construction (GLASSO)
\[
\hat{\Theta} = \arg\min_{\Theta \succ 0} \left\{ \text{tr}(S\Theta) - \log\det(\Theta) + \lambda \|\Theta\|_1 \right\}
\]

Adjacency \(A_{ij} = 1\) if \(\hat{\Theta}_{ij} \neq 0\), normalized as \(W = D^{-1/2} A D^{-1/2}\).

### Loss Functions
- **MSE:** Standard mean squared error
- **QLIKE:** Quasi-likelihood loss robust to outliers: \(\mathcal{L}_{\text{QLIKE}} = \frac{1}{NT}\sum\left(\frac{v}{\hat{v}} - \log\frac{v}{\hat{v}} - 1\right)\)

### Pseudo-IV Validation Framework
A key methodological contribution: synthetic "pseudo-IV" (random Gaussian with same mean/variance as true IV) distinguishes **genuine information content** from **parameter expansion effects**.

---

## Empirical Results (Dow 30, 2021–2026)

### Best Model Performance

| Rank | Model | Loss | Test MSE | Test QLIKE |
|------|-------|------|----------|------------|
| 1 | **GHAR+IV (GLASSO)** | MSE/QLIKE | **1.0045** | 0.000817 |
| 2 | GNNHAR+IV-3L (GLASSO) | MSE | 1.0071 | 0.000822 |
| 3 | GNNHAR+IV-2L (GLASSO) | MSE | 1.0421 | 0.000828 |
| 4 | HAR+IV | MSE | 1.0504 | 0.000839 |
| 5 | GHAR (GLASSO) | MSE | 1.0649 | 0.000849 |
| 6 | HAR | MSE | 1.1189 | 0.000868 |

### Key Findings

1. **Graph Structure Benefits:** GHAR (GLASSO) reduces MSE by 4.8% vs HAR; GHAR+IV reduces by 10.2% vs HAR+IV.

2. **Implied Volatility Information Content (Pseudo-IV Validation):**
   - **Linear models:** 82.4% of IV improvement is genuine information, 17.6% from parameter expansion
   - **Neural networks:** 65.6% genuine information, 34.4% parameter effects — but 63% more absolute information extracted

3. **QLIKE Training Instability:** Severe loss explosion ($10^9$) in deeper GNN architectures; MSE is more stable across all models.

4. **Network Scale Hypothesis (Preliminary):** Results suggest GHAR outperforms GNNHAR on Dow 30 (30 stocks), consistent with the hypothesis that GNN advantages emerge primarily in larger networks (like S&P 100). Further validation with 6-stock simulation is ongoing.

---

## Usage

### Data Preparation
1. Download LOBSTER data (minutely or higher frequency) for your stock universe
2. Run `python scripts/data_subsample.py` to subsample to 5-minute intervals
3. Run `python scripts/compute_vol.py` to compute daily variance

### Model Training
```bash
# Linear baseline (HAR)
python -m src.gnnhar.linear --universe DJIA --horizon 1 --window 22

# Graph-enhanced model (GHAR)
python -m src.gnnhar.linear --universe DJIA --horizon 1 --window 22 --adj_name glasso

# Neural network models (GNNHAR 1/2/3 layers)
python -m src.gnnhar.models --model_name GNNHAR2L --universe DJIA --adj_name glasso --loss MSE
```

The root-level `GHAR.py`, `GNNHAR.py`, `Summary_Results.py`, `Summary_Regime.py`, and `BoxPlot_Error.py` files remain as compatibility wrappers for older commands.

### Result Analysis
```bash
python scripts/summarize_results.py       # MSE, QLIKE, MCS tests
python scripts/summarize_regimes.py       # Results by market regime
python scripts/plot_error_boxplots.py     # Visualization
```

---

## Computing Environment

- **Hardware:** Nvidia A100 GPU (40GB), AMD EPYC 7713 64-Core @ 1.80GHz, 128 cores, 1TB RAM, Ubuntu 20.04.4 LTS
- **Python:** 3.8.18
- **PyTorch:** 2.0.1+cu117
- **Key packages:** numpy 1.22.3, pandas 2.0.3, scikit-learn 1.3.0, matplotlib 3.7.2

---

## References

- Zhang Chao et al. (2024). "Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks." *International Journal of Forecasting*.
- Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics*.
- Hansen, P.R., Lunde, A., & Nason, J.M. (2011). "The Model Confidence Set." *Econometrica*.

---

## Acknowledgments

This work is built upon the foundational methodology of **Prof. Zhang Chao** (港科广/HKUST Guangzhou). The pseudo-IV validation framework and scale-effect analysis extend the original GNNHAR framework to investigate network-size-dependent performance characteristics.
