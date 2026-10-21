import numpy as np

class LogisticRegression:
    """
    L2-Regularized Logistic Regression classifier using IRLS (Newton-Raphson).
    """
    def __init__(self, alpha=0.0, max_iter=100, tol=1e-5, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.w_ = None  # Learned weights (includes intercept if fit_intercept is True)

    def _sigmoid(self, z):
        """Numerically stable sigmoid function."""
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X, y):
        """
        Fit the L2-regularized logistic regression model using IRLS.
        """
        n_samples, n_features = X.shape

        # --- Step 1: Add intercept term (bias) ---
        if self.fit_intercept:
            X_aug = np.hstack([np.ones((n_samples, 1)), X])
            self.w_ = np.zeros(n_features + 1)
        else:
            X_aug = X
            self.w_ = np.zeros(n_features)

        # --- Step 2: Regularization setup ---
        reg_matrix = self.alpha * np.eye(self.w_.shape[0])
        if self.fit_intercept:
            reg_matrix[0, 0] = 0  # No regularization on bias term

        # --- Step 3: IRLS Loop ---
        for i in range(self.max_iter):
            w_old = self.w_.copy()

            # Step 3a: Predictions
            z = X_aug @ self.w_
            h = self._sigmoid(z)

            # Step 3b: Gradient (∇L)
            gradient = X_aug.T @ (h - y) + reg_matrix @ self.w_

            # Step 3c: Weight matrix R
            r_diag = h * (1 - h)
            r_diag = np.maximum(r_diag, 1e-10)  # Avoid zero diagonal entries
            R = np.diag(r_diag)

            # Step 3d: Hessian (H)
            hessian = X_aug.T @ R @ X_aug + reg_matrix

            # Step 3e: Update weights
            try:
                delta_w = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                print("Warning: Hessian not invertible, using pseudo-inverse.")
                delta_w = np.linalg.pinv(hessian) @ gradient

            self.w_ = w_old - delta_w

            # Step 3f: Check convergence
            weight_change = np.linalg.norm(self.w_ - w_old)
            if weight_change < self.tol:
                # print(f"Converged after {i+1} iterations.")
                break
        else:
            print(f"Warning: IRLS did not converge within {self.max_iter} iterations.")

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # Step 1: Add intercept term
        if self.fit_intercept:
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_aug = X

        # Step 2: Compute probabilities
        z = X_aug @ self.w_
        prob_y1 = self._sigmoid(z)
        prob_y0 = 1 - prob_y1

        # Step 3: Stack probabilities
        return np.column_stack([prob_y0, prob_y1])

    def predict(self, X):
        """
        Predict class labels (0 or 1) for samples in X.
        """
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)
import numpy as np

class LogisticRegressionOVR:
    """
    One-vs-Rest Logistic Regression for multi-class classification.
    Each class is trained as a separate binary classifier (class vs all).
    """
    def __init__(self, lr=0.1, n_iter=1000, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.verbose = verbose
        self.W = None  # weights for each classifier (K x D)
        self.b = None  # biases for each classifier (K,)
        self.classes_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # initialize weight matrix (K x D)
        self.W = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)

        for idx, cls in enumerate(self.classes_):
            # binary labels: 1 for this class, 0 for others
            y_binary = (y == cls).astype(np.float64)
            w = np.zeros(n_features)
            b = 0.0

            for i in range(self.n_iter):
                z = np.dot(X, w) + b
                y_pred = self._sigmoid(z)

                # compute gradients
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_binary))
                db = (1 / n_samples) * np.sum(y_pred - y_binary)

                # update parameters
                w -= self.lr * dw
                b -= self.lr * db

                if self.verbose and i % 200 == 0:
                    loss = -np.mean(
                        y_binary * np.log(y_pred + 1e-8) + (1 - y_binary) * np.log(1 - y_pred + 1e-8)
                    )
                    print(f"Class {cls}, iter {i}, loss={loss:.4f}")

            self.W[idx, :] = w
            self.b[idx] = b

        return self

    def predict_proba(self, X):
        # compute sigmoid probabilities for all classes
        z = np.dot(X, self.W.T) + self.b
        probs = self._sigmoid(z)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        # pick the class with the highest probability
        class_indices = np.argmax(probs, axis=1)
        return self.classes_[class_indices]
