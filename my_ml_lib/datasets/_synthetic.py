import numpy as np

def make_noisy_sine(n_samples=100, noise=0.1, random_state=None):
    """
    Generates a noisy sine wave dataset.

    Args:
        n_samples (int): Number of data points to generate.
        noise (float): Standard deviation of Gaussian noise added to y.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        tuple: (X, y)
            X: numpy array of shape (n_samples, 1)
            y: numpy array of shape (n_samples,)
    """
    # Set seed for reproducibility if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Generate uniformly spaced x values
    X = np.random.uniform(0, 1, n_samples).reshape(-1, 1)

    # Generate true sine wave
    y_true = np.sin(2 * np.pi * X).ravel()

    # Add Gaussian noise
    y = y_true + np.random.normal(0, noise, n_samples)

    return X, y
