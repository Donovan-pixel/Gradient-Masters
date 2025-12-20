# Distributional Forecasting of Real Income

## Overview
The objective of this project is to model real income probabilistically rather than through traditional point estimates. Conventional regression approaches, such as standard Gradient Boosting (e.g., XGBoost), only estimate the conditional mean and fail to capture uncertainty or the distributional shape of the data.

To address these limitations, we employ **XGBoostLSS**, an extension of XGBoost that models the entire conditional distribution by jointly estimating its parameters. This framework allows us to handle the specific characteristics of income data, which are strictly positive, right-skewed, and heteroskedastic.

## Key Features & Methodology
* **Probabilistic Modeling:** Uses **XGBoostLSS** with a **Gamma distribution** likelihood, which provided the best fit for the training data.
* **Optimization Framework:**
    * **Nested Cross-Validation:** Employed a 5x5-fold nested structure to ensure robust performance evaluation.
    * **Hyperparameter Tuning:** Conducted via **Bayesian optimization with Optuna** (3-hour budget per inner loop).
    * **Loss Function:** Optimized using **CRPS (Continuous Ranked Probability Score)**.
* **Feature Engineering:**
    * Binning of continuous variables (e.g., age, children).
    * Log-transformations to reduce skewness.
    * Creation of interaction features and mean target encoding for categorical variables.
* **Stability Measures:** Implemented L2 stabilization for gradient/Hessian calculations and an exponential response function to ensure parameter positivity.

## Results
* **Model Performance:**
    * **Cross-Validation:** The final optimized model achieved a mean **CRPS of 7949.84** across outer folds.
    * **Private Test Set:** Achieved a final **CRPS of 8051.15**, successfully **beating the teacher's benchmark**.
* **Inequality Analysis:**
    * Confirmed higher income inequality among females compared to males (higher Gini coefficient and coefficient of variation).
    * Monte Carlo simulations (1980 vs. 2010) show a persistent but slightly narrowing gender income gap.

## Project Structure
* `data/`: Contains raw training and test datasets.
* `notebooks/`: Jupyter notebooks used for EDA, feature engineering, and model training.
* `papers/`: Reference papers for Distributional Gradient Boosting and XGBoostLSS.
* `report.pdf`: The final project report detailing the theoretical background and findings.
* `requirements.txt`: List of Python dependencies.

## References & Key Papers
This project relies on the theoretical frameworks described in the papers located in the `papers/` directory:

* **XGBoostLSS: An Extension of XGBoost to Probabilistic Forecasting** ()
    * This is the core methodology used in our project. It extends the standard XGBoost algorithm to model all moments of a parametric distribution (Location, Scale, and Shape) rather than just the conditional mean.
* **NGBoost: Natural Gradient Boosting for Probabilistic Prediction** ()
    * A foundational paper in probabilistic boosting that introduces the use of "Natural Gradients" to jointly estimate distributional parameters. It serves as a key benchmark and theoretical inspiration for our probabilistic approach.
* **Distributional Gradient Boosting Machines** ()
    * Presents a unified framework for probabilistic gradient boosting. It discusses using Likelihood-based approaches and Normalizing Flows to approximate conditional distributions, providing broader context for the methods applied in this project.

## Installation
To reproduce the results, install the required dependencies:

```bash
pip install -r requirements.txt