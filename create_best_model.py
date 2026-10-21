# create_best_model.py
# ----------------------------------------------------
# Returns a NEW, UNTRAINED instance of your BEST model architecture.
# The autograder will:
#   model = initialize_best_model()
#   model.load_state_dict('saved_models/best_model.npz')
# So the shapes/parameter names MUST match the checkpoint you saved.

from my_ml_lib.nn.modules.base import Module
from my_ml_lib.nn.modules.linear import Linear
from my_ml_lib.nn.modules.activations import ReLU
from my_ml_lib.nn.modules.containers import Sequential
from my_ml_lib.nn.autograd import Value

class MyMLP(Module):
    """
    MLP used as the 'best model' for Fashion-MNIST.
    Architecture (example):
        784 -> 256 -> ReLU -> 128 -> ReLU -> 10
    """
    def __init__(self, n_features: int = 784, n_classes: int = 10):
        super().__init__()
        self.network = Sequential(
            Linear(n_features, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, n_classes)
        )

    def __call__(self, X: Value) -> Value:
        return self.network(X)

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{repr(self.network)}\n)"

def initialize_best_model():
    """
    The autograder calls this to create the model BEFORE loading weights.
    Must match exactly the architecture you trained and saved to 'best_model.npz'.
    """
    return MyMLP(784, 10)
