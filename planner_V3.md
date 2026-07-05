# Implementation Plan: Robust & Agnostic Piecewise Linear Regression

## 📌 Project Overview
**Goal:** Upgrade the existing continuous piecewise linear regression model from a synthetic-data proof-of-concept into a production-ready, dataset-agnostic algorithm. 

**Key Objectives:**
1.  Support N-dimensional feature matrices while isolating non-linear splines to a specific target column.
2.  Ensure mathematical stability during Ordinary Least Squares matrix operations.
3.  Implement a scale-invariant tuning loop that automatically adapts to any dataset's distribution using relative metrics and dynamic knot placement.

---

## 🛠️ Phase 1: Refactoring the Core Model Architecture
Transition the model to handle multi-dimensional arrays and prevent catastrophic failures from singular matrices.

*   [ ] **Step 1.1: Update the Constructor (`__init__`)**
    *   Add a `spline_idx` parameter with a default of `0`.
    *   Initialize `self.knots = np.array(knots)`, `self.spline_idx = spline_idx`, and `self.weights = None`.
*   [ ] **Step 1.2: Dynamic Feature Splining (`_transform_features`)**
    *   Add validation to ensure input `X` is always treated as a 2D matrix (shape `N x D`). Reshape 1D arrays automatically using `X.reshape(-1, 1)`.
    *   Construct the base design matrix containing a bias column (1s) and **all** original features from `X`.
    *   Isolate the target column: `spline_col = X[:, self.spline_idx]`.
    *   Calculate the Truncated Power Basis Functions iteratively for each knot, strictly applied to `spline_col`.
    *   Horizontally stack these computed basis terms onto the base design matrix.
*   [ ] **Step 1.3: Stable Matrix Solver (`fit`)**
    *   Remove `np.linalg.inv` to prevent crashes when knots are too close together or features are highly collinear.
    *   Implement `np.linalg.lstsq(X_trans, y, rcond=None)` to calculate the optimal weights safely using least-squares approximation.
*   [ ] **Step 1.4: Update Prediction (`predict`)**
    *   Add a safeguard check to raise a `ValueError` if `self.weights` is `None` (model not fitted).
    *   Apply `_transform_features(X)` and return the dot product with `self.weights`.

---

## 🧠 Phase 2: Building the Dataset-Agnostic Tuning Loop (Revised)

Replace hardcoded error thresholds with a scale-invariant controller protected by dynamic structural guardrails.

*   [ ] **Step 2.1: Establish Scale-Invariant Baselines**
    *   Calculate `baseline_mse = np.var(y_val)` to serve as the benchmark for a naive model.
*   [ ] **Step 2.2: Compute Structural Guardrails**
    *   Define `min_samples_per_bin = 20`. 
    *   Calculate the hard upper limit for bins: `absolute_max_bins = len(X_train) // min_samples_per_bin`.
    *   Define `min_spatial_delta = 0.03` (adjacent knots must be at least 3% of the total feature range apart).
*   [ ] **Step 2.3: Adaptive Knot Generator with Collinearity Preemption**
    *   Inside the tuning loop, iterate `bins` from 2 up to `absolute_max_bins`.
    *   Generate target quantiles: `quantiles = np.linspace(0, 1, bins + 1)[1:-1]`.
    *   Map quantiles to actual feature values: `knots = np.quantile(X_train[:, spline_idx], quantiles)`.
    *   **The Guardrail Check:** Calculate the distance between adjacent knots: `deltas = np.diff(knots)`. Compute the total range: `feature_range = X_train[:, spline_idx].max() - X_train[:, spline_idx].min()`.
    *   **Action:** If `np.any(deltas < (min_spatial_delta * feature_range))`, immediately trigger early stopping *before* fitting the model. Print: `"Tuning terminated: Minimum spatial bin width reached due to feature skewness."`
*   [ ] **Step 2.4: Relative Improvement Early Stopping**
    *   Fit the model using `np.linalg.lstsq` on the safe knot configuration.
    *   Evaluate `val_mse`. If the relative improvement `(best_val_mse - val_mse) / best_val_mse` is less than 0.5%, or if `val_mse` begins to rise, freeze the previous iteration's bin architecture as the final fixed model.

---

## 🏗️ Phase 3: Integration & Testing Pipeline
Create a robust entry point that handles unknown datasets properly before feeding them to the tuning loop.

*   [ ] **Step 3.1: Generic Data Ingestion**
    *   Write a helper function to load data (e.g., using `pandas.read_csv` or NumPy standard loading).
    *   Separate the dataset into feature matrix `X` and target vector `y`.
*   [ ] **Step 3.2: Scaling & Preprocessing (Crucial for `lstsq`)**
    *   Implement standard scaling (z-score normalization) for the features in `X`. Scaling ensures that the matrix operations in `np.linalg.lstsq` do not suffer from floating-point overflow or extreme eigenvalue disparities.
*   [ ] **Step 3.3: Execution & Verification**
    *   Split the normalized data into Training and Validation sets.
    *   Execute the `adaptive_tuning_loop`, specifying a specific `spline_idx` (e.g., column 2).
    *   Print the final number of bins selected and the final test metrics.