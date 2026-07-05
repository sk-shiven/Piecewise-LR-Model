import numpy as np
import matplotlib.pyplot as plt

# Using the same data generation logic for continuity in testing
np.random.seed(42)
num_points = 1200
X1 = np.linspace(-10, 10, num_points)
y_true = np.sin(X1) * 3
noise_std = 0.5
y = y_true + np.random.normal(loc=0, scale=noise_std, size=num_points)

# Phase 2: Naive "Independent Bins" Model (1D for visualization)
# We will use just X1 for this phase since the cliff problem is best visualised in 1D
X = X1.reshape(-1, 1)

# Define knots
knots = [-5, 0, 5]
bins = [-np.inf] + knots + [np.inf]

plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=10, alpha=0.5, label='Data')

# Fit independent models per bin
for i in range(len(bins) - 1):
    bin_start = bins[i]
    bin_end = bins[i+1]

    # Mask for current bin
    mask = (X[:, 0] >= bin_start) & (X[:, 0] < bin_end)
    X_subset = X[mask]
    y_subset = y[mask]

    if len(X_subset) == 0:
        continue

    # Add bias term
    X_subset_bias = np.c_[np.ones_like(X_subset), X_subset]

    # Normal Equation: theta = (X^T * X)^-1 * X^T * y
    theta = np.linalg.inv(X_subset_bias.T @ X_subset_bias) @ X_subset_bias.T @ y_subset

    # Predict and plot
    y_pred = X_subset_bias @ theta
    plt.plot(X_subset, y_pred, linewidth=3, label=f'Bin {i+1} Fit')

for knot in knots:
    plt.axvline(knot, color='r', linestyle='--', alpha=0.5)

plt.title("Phase 2: Naive Independent Bins (The Cliff Problem)")
plt.xlabel("X1")
plt.ylabel("y")
plt.legend()
plt.savefig("phase2_plot.png")
print("Saved naive model plot to phase2_plot.png")
