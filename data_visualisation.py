import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Generate a dataset of more than 1000 rows
num_points = 1200
X1 = np.linspace(-10, 10, num_points)
X2 = np.random.uniform(-5, 5, num_points)

# True y values using a non-linear function on X1 and linear on X2
# planner.md suggests sin(x) or custom cubic. Let's use sin(X1)
y_true = np.sin(X1) * 3 + 0.5 * X2**2

# Add Gaussian noise
noise_std = 0.5
y = y_true + np.random.normal(loc=0, scale=noise_std, size=num_points)

# Multiple attributes dataset
X = np.c_[X1, X2]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Plot the raw scatter data (using X1 for visualization as it is non-linear)
plt.scatter(X_train[:, 0], y_train, s=10, alpha=0.5, label='Train Data')
plt.scatter(X_val[:, 0], y_val, s=10, alpha=0.5, label='Validation Data')
plt.title("Phase 1: Generated Data")
plt.xlabel("X1")
plt.ylabel("y")
plt.legend()
plt.savefig("phase1_plot.png")
print("Saved plot to phase1_plot.png")