import numpy as np

class KFold:
    """
    K-Folds cross-validator.

    Provides train/test indices to split data in train/test sets.
    Splits dataset into k consecutive folds (without shuffling by default).
    Each fold is then used once as a validation set while the k - 1 remaining
    folds form the training set.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        """
        Initializes the KFold splitter.

        Args:
            n_splits (int): Number of folds. Must be at least 2.
            shuffle (bool): Whether to shuffle the data before splitting into batches.
                            Note that shuffling is done each time split() is called.
            random_state (int, optional): When shuffle is True, random_state affects the ordering
                                           of the indices. Pass an int for reproducible output.
        """
        if not isinstance(n_splits, int) or n_splits <= 1:
            raise ValueError("n_splits must be an integer greater than 1.")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean value.")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Args:
            X (array-like): Data to split, shape (n_samples, n_features).
            y (array-like, optional): The target variable for supervised learning problems.
                                      Used for stratification in StratifiedKFold, but not here.
            groups (array-like, optional): Group labels for the samples used while splitting the dataset
                                            into train/test set. Not used in standard KFold.

        Yields:
            tuple: (train_indices, test_indices) arrays for each split.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        # ✅ TODO 1: Implement shuffling logic if self.shuffle is True
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        # ✅ TODO 2: Compute fold sizes and distribute remainder samples across first folds
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1  # distribute remainder

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop  # move to the next fold

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits
