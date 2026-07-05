import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from piecewise_linear_regression import PiecewiseLinearRegression

def get_evenly_spaced_knots(X_feature, num_knots):
    min_val = np.min(X_feature)
    max_val = np.max(X_feature)
    return np.linspace(min_val, max_val, num_knots + 2)[1:-1]

np.random.seed(42)
num_points = 1200
X1 = np.linspace(-10, 10, num_points)
X2 = np.random.uniform(-5, 5, num_points)

# True y values
y_true = np.sin(X1) * 3 + 1.5 * X2
noise_std = 0.5
y = y_true + np.random.normal(loc=0, scale=noise_std, size=num_points)

# Dataset with multiple attributes
X = np.c_[X1, X2]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Phase 4: Dynamic Hyperparameter Tuning Loop")

max_epochs = 20
current_bins = 2
train_mses = []
val_mses = []
optimal_knots = None
best_val_mse = float('inf')

mse_threshold_underfit = 2.0  # Adjust as appropriate for this data
mse_threshold_overfit = 0.1   # Adjust as appropriate

for epoch in range(max_epochs):
    num_knots = max(1, current_bins - 1)
    knots = get_evenly_spaced_knots(X_train[:, 0], num_knots)

    model = PiecewiseLinearRegression(knots=knots)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    train_mse = mean_squared_error(y_train, y_pred_train)
    val_mse = mean_squared_error(y_val, y_pred_val)

    train_mses.append(train_mse)
    val_mses.append(val_mse)

    print(f"Epoch: {epoch+1} | Bins: {current_bins} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}")

    if val_mse < best_val_mse:
        best_val_mse = val_mse
        optimal_knots = knots

    # Dynamic adjustment logic
    if train_mse > mse_threshold_underfit:
        print(" -> Underfitting detected. Increasing bins drastically.")
        current_bins += 2
    elif val_mse > train_mse * 1.5 and train_mse < mse_threshold_underfit:
        print(" -> Overfitting detected (Val MSE significantly higher than Train MSE). Decreasing bins.")
        current_bins = max(2, current_bins - 1)
    else:
        print(" -> Acceptable range. Slightly increasing bins to explore better fit.")
        current_bins += 1

    if current_bins > 20: # cap the number of bins
        break

print(f"Optimal number of knots based on Validation MSE: {len(optimal_knots)}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_mses) + 1), train_mses, label='Train MSE', marker='o')
plt.plot(range(1, len(val_mses) + 1), val_mses, label='Validation MSE', marker='o')
plt.title("Phase 4: Dynamic Hyperparameter Tuning")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.savefig("phase4_tuning_plot.png")
print("Saved tuning plot to phase4_tuning_plot.png")
