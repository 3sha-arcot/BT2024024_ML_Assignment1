# Machine Learning Assignment 1

**Roll Number:** BT2024024  
**Topic:** Polynomial Regression

---

## What's in here

This repository contains the Python scripts used to train the polynomial regression models and run inference for the assignment:

- `polynomial_regression_var1.py`: Model for Part 1 (Steam Turbine Optimization, 6 features, max degree 10)
- `polynomial_regression_var2.py`: Model for Part 2 (Subterranean Thermal Reservoir Mapping, 3 spatial coordinates, max degree 20)
- `requirements.txt`: Python dependencies

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run

Place your training and test CSV files in the project folder and run:

```bash
# Run Part 1
python polynomial_regression_var1.py

# Run Part 2
python polynomial_regression_var2.py
```

### What the scripts do
1. Runs 5-fold cross-validation across polynomial degrees (up to degree 10 for var1, up to degree 20 for var2) using Ridge regression ($\alpha = 10^{-3}$) to determine the optimal degree.
2. Trains the final pipeline on the entire training set using the best degree.
3. Generates predictions on the test set and exports `BT2024024_pred_var1.csv` and `BT2024024_pred_var2.csv`.
4. Saves cross-validation and residual diagnostic plots to a `plots/` folder.

## Results Summary

- **var1 (Steam Turbine):** Best degree = **4**
  - 5-Fold CV MSE: 0.6998 | CV R²: 0.9265
  - Train MSE: 0.3501 | Train R²: 0.9641

- **var2 (Thermal Mapping):** Best degree = **8**
  - 5-Fold CV MSE: 0.2825 | CV R²: 0.9945
  - Train MSE: 0.1749 | Train R²: 0.9966
