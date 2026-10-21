import numpy as np

class GaussianBasisFeatures:
    def __init__(self, n_centers=100, sigma=5.0, random_state=None):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers_ = None
        self.random_state = random_state # For reproducibility if using random sampling

    def fit(self, X, y=None):
        """
        Select n_centers random points from X as RBF centers.
        """
        # TODO: Determine the centers (mu_j).
        # Strategy: Randomly sample n_centers points from X
        # print("GaussianBasisFeatures: Fit method needs implementation.")
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(X.shape[0], self.n_centers, replace=False)
        self.centers_ = X[indices]
        return self

    def transform(self, X):
        """
        Transform input X into Gaussian RBF features.
        Output shape: (n_samples, n_centers)
        """

        if self.centers_ is None:
            raise RuntimeError("Transformer is not fitted yet.")
        
        X_expanded =    X[:, np.newaxis,:]
        C_expanded = self.centres_[np.newaxis, :, :]
        sq_dist = np.sum((X_expanded-C_expanded)**2, axis=2)
        
        phi = np.exp(-sq_dist/(2*self.sigma**2))

        return phi

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

