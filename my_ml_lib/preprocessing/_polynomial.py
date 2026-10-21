import numpy as np
# May need itertools.combinations_with_replacement

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        # Add any other attributes needed to store fitted info if necessary

    def fit(self, X, y=None):
        # Usually fit doesn't do much here unless you need input dimensions
        self.n_input_features_ = X.shape[1]
        return self

    def transform(self, X):
        
        features = [np.ones((X.shape[0], 1))] if self.include_bias else []

        # Add polynomial powers
        for d in range(1, self.degree + 1):
            features.append(X ** d)

        # Concatenate all features horizontally
        return np.hstack(features)
        #print("PolynomialFeatures: Transform method needs implementation.")
        

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)