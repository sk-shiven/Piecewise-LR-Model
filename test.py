import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from piecewise_linear_regression import PiecewiseLinearRegression

# Step 1: Generate Dataset (as in Phase 1) with multiple attributes
np.random.seed(42)
num_points = 1200
X1 = np.linspace(-10, 10, num_points)
X2 = np.random.uniform(-5, 5, num_points)

# True y values using a non-linear function on X1 and linear on X2
y_true = np.sin(X1) * 3 + 1.5 * X2

noise_std = 0.5
y = y_true + np.random.normal(loc=0, scale=noise_std, size=num_points)

# Dataset with multiple attributes
X = np.c_[X1, X2]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5.1: Instantiate
# Choose knots that might capture the sine wave roughly
knots = [-7.5, -4.5, -1.5, 1.5, 4.5, 7.5]
model = PiecewiseLinearRegression(knots=knots)

# Step 5.2: Fit the model
model.fit(X_train, y_train)

# Step 5.3: Generate predictions
predictions_val = model.predict(X_val)
predictions_train = model.predict(X_train)

# Calculate metrics
mse_train = mean_squared_error(y_train, predictions_train)
mse_val = mean_squared_error(y_val, predictions_val)
r2_train = r2_score(y_train, predictions_train)
r2_val = r2_score(y_val, predictions_val)

print(f"Training MSE: {mse_train:.4f}, R2: {r2_train:.4f}")
print(f"Validation MSE: {mse_val:.4f}, R2: {r2_val:.4f}")

# Step 5.4: Plotting (Sort by X1 for plotting purposes)
sort_idx = np.argsort(X_val[:, 0])
X_val_sorted_1 = X_val[sort_idx, 0]
y_val_sorted = y_val[sort_idx]
predictions_val_sorted = predictions_val[sort_idx]

plt.figure(figsize=(10, 6))
plt.scatter(X_val_sorted_1, y_val_sorted, s=10, alpha=0.5, label='Validation Data')
plt.plot(X_val_sorted_1, predictions_val_sorted, color='red', linewidth=1, label='Predicted Spline (marginal on X1)')

for knot in knots:
    plt.axvline(knot, color='black', linestyle='--', alpha=0.5)

plt.title("Phase 5: Final Evaluation on Validation Set (Multi-attribute)")
plt.xlabel("X1")
plt.ylabel("y")
plt.legend()
plt.savefig("phase5_plot.png")
print("Saved final evaluation plot to phase5_plot.png")
