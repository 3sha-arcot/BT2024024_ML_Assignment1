"""
ML Assignment 1 - Polynomial Regression
Problem: Phase 2 - Subterranean Thermal Reservoir Mapping (var2)
Roll Number: BT2024024

Predicts Thermal Anomaly Score (y) from 3 spatial coordinate offsets
using polynomial regression with cross-validation for degree selection.
Degree up to 20 per the problem statement.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    for candidate in [
        filename,
        os.path.join(SCRIPT_DIR, filename),
        os.path.join(os.path.dirname(SCRIPT_DIR), filename),
        os.path.join(SCRIPT_DIR, "BT2024024", filename),
    ]:
        if os.path.exists(candidate):
            return candidate
    return os.path.join(SCRIPT_DIR, filename)

TRAIN_PATH  = get_path("BT2024024_train_var2.csv")
TEST_PATH   = get_path("BT2024024_test_var2.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "BT2024024_pred_var2.csv")
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

FEATURES = ['x1', 'x2', 'x3']
TARGET   = 'y'

X_train = train_df[FEATURES].values
y_train = train_df[TARGET].values
X_test  = test_df[FEATURES].values

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"y_train stats: mean={y_train.mean():.4f}, std={y_train.std():.4f}, "
      f"min={y_train.min():.4f}, max={y_train.max():.4f}")

# ─────────────────────────────────────────────
# 2. Cross-Validation: Find Optimal Degree
# ─────────────────────────────────────────────
# According to problem: polynomial up to degree 20
MAX_DEGREE = 20
ALPHA      = 1e-3          # Ridge regularisation strength
CV_FOLDS   = 5

cv_mse_scores = {}
cv_r2_scores  = {}

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

print("\n--- Cross-Validation (Ridge Regression) ---")
for degree in range(1, MAX_DEGREE + 1):
    pipe = Pipeline([
        ('poly',   PolynomialFeatures(degree=degree, include_bias=True)),
        ('scaler', StandardScaler()),
        ('model',  Ridge(alpha=ALPHA))
    ])
    neg_mse = cross_val_score(pipe, X_train, y_train, cv=kf,
                              scoring='neg_mean_squared_error', n_jobs=-1)
    r2      = cross_val_score(pipe, X_train, y_train, cv=kf,
                              scoring='r2', n_jobs=-1)
    cv_mse_scores[degree] = -neg_mse.mean()
    cv_r2_scores[degree]  =  r2.mean()
    print(f"  Degree {degree:2d} | CV MSE: {cv_mse_scores[degree]:.5f} "
          f"| CV R2: {cv_r2_scores[degree]:.5f}")

# ─────────────────────────────────────────────
# 3. Select Best Degree
# ─────────────────────────────────────────────
best_degree = min(cv_mse_scores, key=cv_mse_scores.get)
print(f"\nBest degree for var2: {best_degree} "
      f"(CV MSE={cv_mse_scores[best_degree]:.5f}, "
      f"CV R2={cv_r2_scores[best_degree]:.5f})")

# ─────────────────────────────────────────────
# 4. Train Final Model on Full Training Data
# ─────────────────────────────────────────────
final_pipe = Pipeline([
    ('poly',   PolynomialFeatures(degree=best_degree, include_bias=True)),
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=ALPHA))
])
final_pipe.fit(X_train, y_train)

y_train_pred = final_pipe.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2  = r2_score(y_train, y_train_pred)
print(f"\nFinal model (degree={best_degree}) -- Train MSE: {train_mse:.5f}, Train R2: {train_r2:.5f}")

# ─────────────────────────────────────────────
# 5. Predict on Test Set
# ─────────────────────────────────────────────
y_test_pred = final_pipe.predict(X_test)
submission  = pd.DataFrame({'y': y_test_pred})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nPredictions saved to: {OUTPUT_PATH}")
print(f"Test predictions stats: mean={y_test_pred.mean():.4f}, "
      f"std={y_test_pred.std():.4f}, min={y_test_pred.min():.4f}, "
      f"max={y_test_pred.max():.4f}")

# ─────────────────────────────────────────────
# 6. Visualisations
# ─────────────────────────────────────────────

# 6a. CV MSE vs Degree
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

degrees = list(cv_mse_scores.keys())
mses    = [cv_mse_scores[d] for d in degrees]
r2s     = [cv_r2_scores[d]  for d in degrees]

axes[0].plot(degrees, mses, marker='o', color='teal', linewidth=2)
axes[0].axvline(x=best_degree, color='red', linestyle='--', label=f'Best degree={best_degree}')
axes[0].set_xlabel('Polynomial Degree', fontsize=12)
axes[0].set_ylabel('CV MSE', fontsize=12)
axes[0].set_title('var2: Cross-Validation MSE vs Polynomial Degree', fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(degrees, r2s, marker='s', color='purple', linewidth=2)
axes[1].axvline(x=best_degree, color='red', linestyle='--', label=f'Best degree={best_degree}')
axes[1].set_xlabel('Polynomial Degree', fontsize=12)
axes[1].set_ylabel('CV R2', fontsize=12)
axes[1].set_title('var2: Cross-Validation R2 vs Polynomial Degree', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'var2_cv_scores.png'), dpi=150, bbox_inches='tight')
plt.close()

# 6b. Residual Plot
residuals = y_train - y_train_pred
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_train_pred, residuals, alpha=0.4, color='teal', s=15)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('Predicted y', fontsize=12)
axes[0].set_ylabel('Residuals', fontsize=12)
axes[0].set_title('var2: Residual Plot (Train)', fontsize=13)
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_train, y_train_pred, alpha=0.4, color='purple', s=15)
lims = [min(y_train.min(), y_train_pred.min()), max(y_train.max(), y_train_pred.max())]
axes[1].plot(lims, lims, 'r--', linewidth=2, label='Perfect fit')
axes[1].set_xlabel('Actual y', fontsize=12)
axes[1].set_ylabel('Predicted y', fontsize=12)
axes[1].set_title('var2: Actual vs Predicted (Train)', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'var2_residuals.png'), dpi=150, bbox_inches='tight')
plt.close()

# 6c. 3D scatter of predictions vs coordinates (x1, x2)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], y_train,
                     c=y_train, cmap='viridis', alpha=0.5, s=10)
plt.colorbar(scatter, ax=ax, label='Thermal Anomaly Score (y)')
ax.set_xlabel('x1 (East-West)')
ax.set_ylabel('x2 (North-South)')
ax.set_zlabel('y (Thermal Anomaly)')
ax.set_title('var2: Training Data 3D Scatter (x1, x2, y)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'var2_3d_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\nPlots saved in plots/ directory.")
print("\n=== var2 SUMMARY ===")
print(f"  Best polynomial degree : {best_degree}")
print(f"  CV MSE                 : {cv_mse_scores[best_degree]:.5f}")
print(f"  CV R2                  : {cv_r2_scores[best_degree]:.5f}")
print(f"  Train MSE              : {train_mse:.5f}")
print(f"  Train R2               : {train_r2:.5f}")
