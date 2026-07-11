# Continuous Piecewise Linear Regression

A robust, dataset-agnostic custom machine learning model for Continuous Piecewise Linear Regression using Linear Splines. This project transitions a synthetic-data proof-of-concept into a production-ready algorithm capable of handling N-dimensional feature matrices while isolating non-linear splines to a specific target column.

## 🚀 Key Features

*   **Scikit-Learn Compatible API:** Object-Oriented design conforming to the standard `.fit()` and `.predict()` methodology.
*   **Multi-Dimensional Support:** Handles N-dimensional feature arrays, applying Truncated Power Basis Functions for splines strictly to a target column.
*   **Mathematical Stability:** Utilizes the pseudo-inverse (`np.linalg.pinv`) for Ordinary Least Squares (OLS) matrix operations, robustly handling singular matrices and highly collinear features without crashing.
*   **Adaptive Hyperparameter Tuning:** Implements a dynamic tuning loop that automatically adapts to the dataset's distribution to find the optimal number of bins (knots).
*   **Visual Validations:** Provides comprehensive standalone phase scripts that visualize data generation, naive discontinuous splines (The Cliff Problem), continuous stitched splines, dynamic tuning, and final evaluations.

## 🛠️ Tech Stack

*   **Python 3.x**
*   **NumPy:** Core library for vectorization, array manipulation, and linear algebra operations.
*   **Matplotlib:** Extensive data visualization and plot generation.
*   **scikit-learn:** Utilized for data splitting (`train_test_split`) and performance metrics (`mean_squared_error`, `r2_score`).

## 📁 Repository Structure

```
├── main.py                          # Unified orchestration script to run all phases
├── piecewise_linear_regression.py   # Core model class (`PiecewiseLinearRegression`)
├── data_visualisation.py            # Phase 1: Generates and visualizes synthetic data
├── discontinuous_splined_lr.py      # Phase 2: Demonstrates the naive independent bins issue
├── splined_lr.py                    # Phase 3: Implements global continuous splines (stitching)
├── Hyperparameter_Tuning_Loop.py    # Phase 4: Dynamic hyperparameter tuning loop
├── test.py                          # Phase 5: Final evaluation and testing multi-attribute data
├── planner_V3.md                    # Implementation roadmap and design objectives
└── README.md                        # This file
```

## ⚙️ Usage Instructions

### Running the Full Pipeline

The project includes a centralized orchestrator to sequentially execute and verify all development phases. To run the full suite:

```bash
python main.py
```
This will run through all the phase scripts and generate `.png` visualizations in the root directory.

### Using the Model Independently

You can integrate the `PiecewiseLinearRegression` model into your own workflows.

```python
import numpy as np
from piecewise_linear_regression import PiecewiseLinearRegression

# 1. Generate or load data
X = np.random.uniform(-10, 10, (1000, 2))
y = np.sin(X[:, 0]) * 3 + 1.5 * X[:, 1] + np.random.normal(0, 0.5, 1000)

# 2. Define knots (points on the x-axis to create hinges)
knots = [-5, 0, 5]

# 3. Instantiate the model
model = PiecewiseLinearRegression(knots=knots)

# 4. Fit the model
model.fit(X, y)

# 5. Make predictions
predictions = model.predict(X)
```

## 🗺️ Roadmap / Future Enhancements

As detailed in `planner_V3.md`, future updates aim to implement:
*   **Scale-Invariant Tuning Loop:** Replacing hardcoded error thresholds with a scale-invariant controller based on relative metrics.
*   **Adaptive Knot Generator:** Dynamic structural guardrails to prevent minimum spatial bin widths and collinearity preemption before fitting.
*   **Automated Target Column Splining:** Direct parameterization to designate specific features for non-linear splining in multi-dimensional datasets.
