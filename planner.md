# Implementation Plan: Continuous Piecewise Linear Regression (Linear Splines)

## 📌 Project Overview
**Goal:** Build a custom machine learning model that divides a dataset into spatial bins (based on the x-axis) and fits local regression lines. 
**Key Constraint:** The lines must be mathematically "stitched" together at the bin boundaries (knots) to ensure a continuous prediction curve without vertical jumps.

**Assumptions & Tech Stack:**
*   **Language:** Python 3.x
*   **Libraries:** `numpy` (for matrix math/linear algebra), `matplotlib.pyplot` (for visualizing the cliffs and the stitched lines).
*   **Architecture:** Object-Oriented Programming (OOP) conforming to the `scikit-learn` API style (`.fit()` and `.predict()` methods).

---

## 🛠️ Phase 1: Environment Setup & Data Generation
Before building the model, we need a dataset that a standard straight line would fail to predict (underfitting) to prove our custom model works.

*   [ ] **Step 1.1:** Import `numpy` and `matplotlib`.
*   [ ] **Step 1.2:** Generate a synthetic non-linear dataset. 
    *   *Hint:* Create 200 points using a sine wave function or a cubic function: $y = \sin(x) + \text{noise}$.
    *   Use `np.linspace` for $x$ values and `np.random.normal` for the noise.
*   [ ] **Step 1.3:** Plot the raw scatter data to visually confirm it is non-linear.

---

## 🚧 Phase 2: The Naive "Independent Bins" Model (The Cliff Problem)
Implement your initial idea exactly as you conceived it to observe the discontinuity.

*   [ ] **Step 2.1:** Define 3 "knots" (boundaries) on the x-axis to divide your data into 4 distinct bins.
*   [ ] **Step 2.2:** Write a loop that slices the $X$ and $y$ arrays based on these bin boundaries.
*   [ ] **Step 2.3:** For each bin, calculate a standard Linear Regression.
    *   *Linear Algebra Refresher:* Use the Normal Equation $\theta = (X^T X)^{-1} X^T y$ to find the slope and intercept for each specific bin.
*   [ ] **Step 2.4:** Plot all 4 distinct lines over the scatter plot.
*   [ ] **Step 2.5:** **Milestone Check:** Observe the graph. You should clearly see the "vertical cliffs" where the bins meet.

---

## 🧠 Phase 3: The "Stitching" Math (Continuous Splines)
To fix the cliffs, we won't actually solve 4 separate regressions. Instead, we use a brilliant linear algebra trick called **Truncated Power Basis Functions**. 

We transform our features so that a single global regression automatically handles the local stitching.

*   [ ] **Step 3.1: Feature Engineering (The Magic Trick).** 
    Instead of just having $X = [x]$, we create new features for every knot $k$. The formula for a new feature is:
    $$f_k(x) = \max(0, x - k)$$
    *Translation:* If $x$ is past the knot, it activates. If it's before the knot, it stays $0$.
*   [ ] **Step 3.2: Build the Design Matrix.**
    Create a function that takes your original $X$ and your list of knots, and returns a new matrix with columns:
    `[ 1, x, max(0, x - k_1), max(0, x - k_2), max(0, x - k_3) ]`
*   [ ] **Step 3.3: Fit the Global Model.**
    Run standard Ordinary Least Squares (OLS) on this newly expanded matrix.
    *   *Why this works:* The weights assigned to those `max()` features act as "hinges." They tell the line exactly how much to bend at each knot, guaranteeing they stay attached!
*   [ ] **Step 3.4:** Plot the new prediction line. It should trace your data beautifully with sharp, continuous bends at each knot.

---

## 🏗️ Phase 4: Refactoring into Production OOP
Wrap the successful math from Phase 3 into a clean, reusable Python class.

*   [ ] **Step 4.1:** Define `class PiecewiseLinearRegression:`
*   [ ] **Step 4.2:** Implement `__init__(self, knots)`
    *   Store the list of knots as an instance variable.
    *   Initialize `self.weights = None`.
*   [ ] **Step 4.3:** Implement `_transform_features(self, X)`
    *   A private helper method that takes raw $X$ and outputs the expanded matrix with the $\max(0, x-k)$ columns and the bias column of 1s.
*   [ ] **Step 4.4:** Implement `fit(self, X, y)`
    *   Call `_transform_features`.
    *   Solve for `self.weights` using `np.linalg.inv(X.T @ X) @ X.T @ y`.
    *   *(Optional Extra Credit: Instead of exact matrix inversion, implement Mini-Batch Gradient Descent here to solve for the weights!)*
*   [ ] **Step 4.5:** Implement `predict(self, X)`
    *   Call `_transform_features`.
    *   Return the dot product of the transformed $X$ and `self.weights`.

---

## 🚀 Phase 5: Final Testing
*   [ ] **Step 5.1:** Instantiate your class: `model = PiecewiseLinearRegression(knots=[2, 4, 6])`
*   [ ] **Step 5.2:** `model.fit(X_train, y_train)`
*   [ ] **Step 5.3:** `predictions = model.predict(X_test)`
*   [ ] **Step 5.4:** Plot the final model against the raw data.