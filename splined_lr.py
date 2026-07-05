import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
num_points = 1200
X1 = np.linspace(-10, 10, num_points)
y_true = np.sin(X1) * 3
noise_std = 0.5
y = y_true + np.random.normal(loc=0, scale=noise_std, size=num_points)

X = X1.reshape(-1, 1)

# Phase 3: The "Stitching" Math (Continuous Splines)
knots = [-5, 0, 5]

# Step 3.1 & 3.2: Build the Global Design Matrix
base = np.c_[np.ones_like(X), X]

# Create list to hold all features
features = [base]

# Add truncated power basis functions for each knot
for k in knots:
    features.append(np.maximum(0, X - k))

X_transformed = np.hstack(features)

# Step 3.3: Fit the Global Model using OLS
theta = np.linalg.inv(X_transformed.T @ X_transformed) @ X_transformed.T @ y

# Step 3.4: Plot the global prediction
y_pred = X_transformed @ theta

plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=10, alpha=0.5, label='Data')
plt.plot(X, y_pred, color='red', linewidth=3, label='Global Continuous Spline Fit')

for knot in knots:
    plt.axvline(knot, color='black', linestyle='--', alpha=0.5)

plt.title("Phase 3: Continuous Splines (Stitched)")
plt.xlabel("X1")
plt.ylabel("y")
plt.legend()
plt.savefig("stitched_plot.png")
print("Saved continuous spline model plot to stitched_plot.png")
