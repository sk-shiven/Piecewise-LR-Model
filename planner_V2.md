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

## 🧠 Phase 2: Building the Dataset-Agnostic Tuning Loop
Replace hardcoded error thresholds and static linear spacing with a flexible, scale-invariant controller.

*   [ ] **Step 2.1: Establish Scale-Invariant Baselines**
    *   Before the loop begins, calculate `baseline_mse = np.var(y_val)`. This represents the error of a naive model predicting only the mean of the validation set, providing a universal scale for "bad" performance regardless of the dataset's units.
*   [ ] **Step 2.2: Implement Data-Driven Knot Placement**
    *   Replace `np.linspace` with `np.quantile`.
    *   For a given number of bins, calculate uniform quantiles: `quantiles = np.linspace(0, 1, bins + 1)[1:-1]`.
    *   Map these quantiles to actual knot values using `np.quantile(X_train[:, spline_idx], quantiles)`. This guarantees knots are placed where data is dense, avoiding empty regions caused by outliers.
*   [ ] **Step 2.3: Relative Improvement Early Stopping**
    *   Inside the epoch loop, fit the model and evaluate `val_mse`.
    *   Calculate the percentage improvement: `improvement = (best_val_mse - val_mse) / best_val_mse`.
    *   If `val_mse < best_val_mse`, update the best model.
    *   **Stop Condition 1 (Diminishing Returns):** If `improvement` is positive but falls below a strict threshold (e.g., `< 0.005` or 0.5%), break the loop. The added complexity is no longer yielding meaningful gains.
    *   **Stop Condition 2 (Overfitting):** If `val_mse` increases compared to the previous best iteration, break the loop and revert to the previously saved optimal model.

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