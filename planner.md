# Implementation Plan: Continuous Piecewise Linear Regression (Linear Splines)

## 📌 Project Overview
**Goal:** Build a custom machine learning model from scratch that divides a dataset into spatial bins along the x-axis and fits local regression lines.
**Key Constraint:** The regression lines must be mathematically "stitched" together at the bin boundaries (known as "knots") to guarantee a continuous prediction curve without any vertical discontinuities (jumps or cliffs).

### 📖 Theoretical Foundation
The objective is to create a model that captures non-linear relationships using linear components. The naive approach of splitting data into independent bins and fitting separate regressions results in discontinuities at bin boundaries (the "cliff problem").

To solve this, we use **Linear Splines**. A spline is a piecewise polynomial function that is continuous and has continuous derivatives up to a certain degree. For continuous piecewise linear regression, we use splines of degree 1.

We achieve continuity by using **Truncated Power Basis Functions**. Rather than fitting independent models, we construct a global design matrix that inherently enforces continuity at the knots.
If we have knots $k_1, k_2, \dots, k_n$, we map a single feature $x$ into a higher-dimensional space:
$1, x, (x-k_1)_+, (x-k_2)_+, \dots, (x-k_n)_+$
Where $(x - k_i)_+ = \max(0, x - k_i)$.

This means:
- Before the first knot, the line has a base slope and intercept (defined by the $1$ and $x$ terms).
- After passing knot $k_i$, the term $(x-k_i)_+$ becomes non-zero. The model learns a coefficient (weight) for this term, which acts as a "slope adjustment" or "hinge".
- Because the adjustment $(x-k_i)_+$ is exactly $0$ at the knot $x = k_i$ and grows linearly thereafter, the line seamlessly bends at the knot without any vertical jump.
- By using ordinary least squares (OLS) on this expanded design matrix, we find the optimal base slope, intercept, and slope adjustments simultaneously.

**Assumptions & Tech Stack:**
*   **Language:** Python 3.x
*   **Libraries:** `numpy` (for matrix math, data generation, and linear algebra), `matplotlib.pyplot` (for visualizing the data, the naive "cliff" problem, and the final continuous spline).
*   **Architecture:** Object-Oriented Programming (OOP) conforming to the `scikit-learn` API style (`.fit()` and `.predict()` methods).

---

## 🛠️ Phase 1: Environment Setup & Data Generation
Before building the model, we need a dataset that a standard straight line would fail to predict (underfitting) to prove our custom model works. We will generate synthetic data using a non-linear underlying function.

*   [ ] **Step 1.1:** Import necessary modules.
    *   `import numpy as np`
    *   `import matplotlib.pyplot as plt`
*   [ ] **Step 1.2:** Generate a synthetic non-linear dataset. 
    *   Use `np.random.seed` for reproducibility.
    *   Use `np.linspace(min_val, max_val, num_points)` to generate 200 evenly spaced $X$ values.
    *   Generate true $y$ values using a non-linear function, e.g., $y_{true} = \sin(x)$ or a custom cubic function.
    *   Add Gaussian noise using `np.random.normal(loc=0, scale=noise_std, size=num_points)` to create a realistic scatter: $y = y_{true} + \text{noise}$.
*   [ ] **Step 1.3:** Plot the raw scatter data.
    *   Use `plt.scatter(X, y)` to visually confirm the non-linear trend.

---

## 🚧 Phase 2: The Naive "Independent Bins" Model (The Cliff Problem)
Implement the initial, intuitive idea to observe why the discontinuity occurs. This demonstrates the necessity of the spline approach.

*   [ ] **Step 2.1:** Define knots.
    *   Choose 3 internal knot positions (e.g., $k_1, k_2, k_3$) that divide the x-axis into 4 distinct bins.
*   [ ] **Step 2.2 & 2.3:** Fit independent models per bin.
    *   Write a loop or list comprehension that iterates through each bin interval defined by the knots (and the data boundaries).
    *   For each bin, slice the $X$ and $y$ arrays: `mask = (X >= bin_start) & (X < bin_end)`.
    *   Calculate a standard Linear Regression for this isolated subset.
    *   *Linear Algebra Refresher:* To find weights $\theta$ for the subset, construct the design matrix $X_{subset\_bias} = [1, X_{subset}]$. Solve using the Normal Equation: $\theta = (X_{subset\_bias}^T X_{subset\_bias})^{-1} X_{subset\_bias}^T y_{subset}$. Use `np.linalg.inv` and the `@` operator for matrix multiplication.
*   [ ] **Step 2.4:** Plot the naive models.
    *   Plot the data scatter.
    *   For each bin, plot the best fit line `y_pred = X_subset_bias @ theta` only within the domain of that bin.
*   [ ] **Step 2.5:** **Milestone Check:** Observe the graph. You should clearly see "vertical cliffs" (discontinuities) where adjacent lines fail to meet at the knot boundaries.

---

## 🧠 Phase 3: The "Stitching" Math (Continuous Splines)
To fix the cliffs, we transition from independent local models to a single global model using **Truncated Power Basis Functions**.

*   [ ] **Step 3.1: Feature Engineering (The Magic Trick).** 
    Instead of just having $X = [x]$, we create new features for every knot $k$. The formula for a new feature is:
    $f_k(x) = \max(0, x - k)$
    *Translation:* If $x$ is past the knot, the feature activates and grows linearly. If it's before or exactly at the knot, it stays $0$.
*   [ ] **Step 3.2: Build the Global Design Matrix.**
    Create a transformation function. For an input vector $X$ and knots $[k_1, k_2, k_3]$:
    *   Initialize the matrix with a column of 1s (bias/intercept) and a column of the original $X$ values. `base = np.c_[np.ones_like(X), X]`
    *   For each knot $k_i$, compute the truncated power basis `np.maximum(0, X - k_i)`.
    *   Horizontally stack these new features: `X_transformed = np.c_[base, max_k1, max_k2, max_k3]`.
*   [ ] **Step 3.3: Fit the Global Model.**
    Run standard Ordinary Least Squares (OLS) on this newly expanded matrix.
    *   Solve $\theta = (X_{transformed}^T X_{transformed})^{-1} X_{transformed}^T y$ using `np.linalg.inv(X_trans.T @ X_trans) @ X_trans.T @ y`.
    *   *Why this works:* The weights $\theta$ correspond to: $[ \text{intercept}, \text{base\_slope}, \text{slope\_adjustment\_}k_1, \text{slope\_adjustment\_}k_2, \dots ]$. The line seamlessly transitions its slope exactly at each knot.
*   [ ] **Step 3.4:** Plot the global prediction.
    *   Calculate global predictions: `y_pred = X_transformed @ theta`.
    *   Plot `plt.plot(X, y_pred)`. It should trace the data beautifully with sharp, continuous bends exactly at each knot.

---

## 🏗️ Phase 4: Refactoring into Production OOP
Wrap the successful mathematical logic from Phase 3 into a clean, reusable Python class adhering to the scikit-learn API style.

*   [ ] **Step 4.1:** Define the class.
    *   `class PiecewiseLinearRegression:`
*   [ ] **Step 4.2:** Implement constructor `__init__(self, knots)`
    *   Store the list of knots: `self.knots = np.array(knots)`.
    *   Initialize `self.weights = None`.
*   [ ] **Step 4.3:** Implement `_transform_features(self, X)`
    *   A private helper method taking an array $X$ of shape (N, 1) or (N,).
    *   Create and return the expanded design matrix: `[1, X, max(0, X-k1), max(0, X-k2), ...]`.
    *   Ensure $X$ is flattened or properly reshaped to handle 1D and 2D single-feature arrays correctly.
*   [ ] **Step 4.4:** Implement `fit(self, X, y)`
    *   Transform the input $X$ by calling `X_trans = self._transform_features(X)`.
    *   Solve for `self.weights` using the Normal Equation: `self.weights = np.linalg.inv(X_trans.T @ X_trans) @ X_trans.T @ y`.
    *   *(Optional Extra Credit: Implement Mini-Batch Gradient Descent as an alternative solver, iterating through batches to update weights using gradients: $\nabla_\theta J(\theta) = \frac{1}{m} X_{batch}^T (X_{batch} \theta - y_{batch})$).*
*   [ ] **Step 4.5:** Implement `predict(self, X)`
    *   Verify that the model has been fitted (`self.weights` is not None).
    *   Transform $X$: `X_trans = self._transform_features(X)`.
    *   Return predictions: `return X_trans @ self.weights`.

---

## 🚀 Phase 5: Final Testing
*   [ ] **Step 5.1:** Instantiate your class: `model = PiecewiseLinearRegression(knots=[2, 4, 6])`
*   [ ] **Step 5.2:** Fit the model: `model.fit(X_train, y_train)`
*   [ ] **Step 5.3:** Generate predictions: `predictions = model.predict(X_test)`
*   [ ] **Step 5.4:** Plotting and Evaluation.
    *   Plot the fitted continuous line against the raw data.
    *   (Optional) Calculate evaluation metrics such as Mean Squared Error (MSE) or $R^2$ score to quantify the goodness-of-fit.