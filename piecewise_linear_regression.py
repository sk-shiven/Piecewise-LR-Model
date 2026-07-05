import numpy as np

class PiecewiseLinearRegression:
    def __init__(self, knots):
        """
        Initializes the PiecewiseLinearRegression model.

        Args:
            knots (list or np.ndarray): Points on the x-axis to create hinges.
        """
        self.knots = np.array(knots)
        self.weights = None

    def _transform_features(self, X):
        """
        Transforms the input feature matrix X into the expanded design matrix
        using Truncated Power Basis Functions.

        Args:
            X (np.ndarray): Input feature matrix of shape (N, 1) or (N,).

        Returns:
            np.ndarray: The transformed design matrix.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        base = np.c_[np.ones((X.shape[0], 1)), X]
        features = [base]

        # Only transforming the first feature if multiple are provided
        # Based on the problem description, splines are usually defined on a single axis.
        # If X has multiple features, we'll apply knots to the first feature for now.
        x_target = X[:, 0].reshape(-1, 1)

        for k in self.knots:
            features.append(np.maximum(0, x_target - k))

        return np.hstack(features)

    def fit(self, X, y):
        """
        Fits the continuous piecewise linear regression model.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target values.
        """
        X_trans = self._transform_features(X)
        # Using pseudo-inverse (pinv) to handle potentially singular matrices
        # (e.g. highly collinear features or more features than samples)
        self.weights = np.linalg.pinv(X_trans.T @ X_trans) @ X_trans.T @ y
        return self

    def predict(self, X):
        """
        Predicts values using the fitted model.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.weights is None:
            raise ValueError("The model has not been fitted yet. Please call fit() first.")

        X_trans = self._transform_features(X)
        return X_trans @ self.weights
