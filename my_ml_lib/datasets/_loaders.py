import os
import numpy as np
import pandas as pd

class DatasetNotFoundError(Exception):
    """Custom exception for when a dataset file is not found."""
    pass


def load_spambase(data_folder="data", filename="spambase.data"):
    """
    Loads the UCI Spambase dataset from a local file.
    Returns:
        tuple: (X, y) numpy arrays.
    """
    file_path = os.path.join(data_folder, filename)
    if not os.path.exists(file_path):
        raise DatasetNotFoundError(
            f"Dataset file not found at {file_path}. "
            f"Please download it from the UCI ML Repository and place it in '{data_folder}'."
        )

    columns = [
        # 48 word frequency features
        'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
        'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
        'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
        'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
        'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
        'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
        'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
        'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
        'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
        'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
        'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
        'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',

        # 6 character frequency features
        'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',

        # 3 capital letter features
        'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total',

        # Target label
        'is_spam'
    ]

    df = pd.read_csv(file_path, header=None, names=columns)

    X = df.drop('is_spam', axis=1).to_numpy()
    y = df['is_spam'].to_numpy()

    return X, y


def load_fashion_mnist(data_folder="data",
                       train_filename="fashion-mnist_train.csv",
                       test_filename="fashion-mnist_test.csv",
                       kind='train',
                       normalize=True):
    """
    Loads the Fashion-MNIST dataset from local CSV files.
    """
    if kind == 'train':
        filename = train_filename
    elif kind == 'test':
        filename = test_filename
    else:
        raise ValueError("kind must be 'train' or 'test'")

    file_path = os.path.join(data_folder, filename)
    if not os.path.exists(file_path):
        raise DatasetNotFoundError(
            f"Dataset file not found at {file_path}. "
            f"Please download it and place it in '{data_folder}'."
        )

    df = pd.read_csv(file_path)
    y = df['label'].to_numpy()
    X = df.drop('label', axis=1).to_numpy()

    if normalize:
        X = X.astype(np.float32) / 255.0

    return X, y
