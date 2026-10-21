# my_ml_lib/nn/losses.py

import numpy as np
from .modules.base import Module
from .autograd import Value


### You would use this for one vs rest Logistic Regression
class BinaryCrossEntropyLoss(Module):
    """
    Computes the Binary Cross Entropy loss between logits and targets (0 or 1).
    Assumes the input is a single logit (pre-sigmoid score) per sample.
    """
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction type: {reduction}. Must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def __call__(self, logits: Value, targets: np.ndarray) -> Value:
        """
        Calculates the Binary Cross Entropy loss.

        Args:
            logits (Value): Input logits (pre-sigmoid scores), shape (batch_size,) or (batch_size, 1).
            targets (np.ndarray): Ground truth labels (0 or 1), shape (batch_size,).

        Returns:
            Value: The computed loss (scalar if reduction is 'mean' or 'sum').
        """
        # Ensure targets are float64 and have the correct shape for broadcasting
        targets_np = targets.astype(np.float64).reshape(-1, 1) # Ensure column vector
        targets_val = Value(targets_np)  # This is Step 3
        # --- TODO: Step 1 - Apply Sigmoid ---
        # Apply the sigmoid activation function to the input logits Value object.
        # This gives the predicted probabilities 'p'.
        probs = 1 / (1 + (-logits).exp())  # sigmoid(logits)

        # --- TODO: Step 2 - Numerical Stability (Clipping) ---
        # Clip the probabilities to avoid log(0).
        # Add a small epsilon or use np.clip on the data before the log.
        # Applying this *after* the Value operation requires care, or do it within log.
        # It might be simpler to add epsilon before the log operation itself.
        # Example: probs_clipped = probs + 1e-15 (but ensure this works with Value)
        # Alternatively, handle stability inside the Value.log() method.
        epsilon = 1e-12
        probs_stable = probs.clip(epsilon, 1 - epsilon)  # Assuming Value.clip exists; if not, implement manually

        # --- TODO: Step 3 - Wrap Targets ---
        # Wrap the numpy targets_np array into a Value object. It acts as a constant.
        # targets_val = Value(targets_np)
        #print("TODO: Wrap numpy targets into a Value object in BCE Loss.")
        #targets_val = Value(targets_np) # Placeholder

        # --- TODO: Step 4 - Calculate BCE Formula ---
        # Implement the BCE formula using Value operations:
        # Remember that (1 - targets_val) and (1 - probs_stable) should use Value subtraction.
        print("TODO: Implement the BCE formula using Value operations.")
        loss_elements = -(targets_val * probs_stable.log() + (1 - targets_val) * (1 - probs_stable).log())
        # --- TODO: Step 5 - Apply Reduction ---
        # Apply the specified reduction ('mean', 'sum', or 'none') to loss_elements.
        # Hint: Use loss_elements.mean(), loss_elements.sum(), or return loss_elements directly.
        if self.reduction == 'mean':
            loss = loss_elements.mean()
        elif self.reduction == 'sum':
            loss = loss_elements.sum()
        else:  # 'none'
            loss = loss_elements

        return loss

    def __repr__(self):
        """String representation."""
        return f"BinaryCrossEntropyLoss(reduction='{self.reduction}')"


class CrossEntropyLoss(Module):
    """
    Computes the cross-entropy loss between input logits and target class indices.
    Combines LogSoftmax and NLLLoss.
    """
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction type: {reduction}. Must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction
        # No internal storage needed if using pure autograd approach

    def __call__(self, input_logits: Value, target: np.ndarray) -> Value:
        """
        Computes the cross-entropy loss using only Value operations.

        Args:
            input_logits (Value): Raw logits, shape (batch_size, n_classes).
            target (np.ndarray): Ground truth labels (class indices 0 to n_classes-1), shape (batch_size,).

        Returns:
            Value: The computed loss.
        """
        batch_size, n_classes = input_logits.data.shape

        # --- TODO: Step 1 - Calculate LogSoftmax ---
        # Implement numerically stable LogSoftmax using Value operations.

        # logsoftmax(x)_i = x_i - log(sum(exp(x_j))) over classes j.
        #   a) Find max logit per sample (use numpy on .data for stability, keepdims=True).
        #   b) Subtract max logit from input_logits Value (broadcasting numpy array is fine).
        #   c) Exponentiate the stable logits (Value.exp()).
        #   d) Sum the exponentiated logits over the class dimension (Value.sum(axis=1, keepdims=True)).
        #   e) Take the logarithm of the sum (Value.log()).
        #   f) Subtract the log_sum_exp from the stable_logits to get log_probs.
        # 1a: Find max per sample (numpy only, for stability)
        max_logits = np.max(input_logits.data, axis=1, keepdims=True)
        # 1b: Subtract max_logits from input logits (broadcasting works)
        stable_logits = input_logits - Value(max_logits)
        # 1c: Exponentiate
        exp_logits = stable_logits.exp()
        # 1d: Sum over classes
        sum_exp = exp_logits.sum(axis=1, keepdims=True)
        # 1e: Take log
        log_sum_exp = sum_exp.log()
        # 1f: Compute log softmax
        log_probs = stable_logits - log_sum_exp  # shape: (batch_size, n_classes)
        print("TODO: Implement numerically stable LogSoftmax using Value ops in CE Loss.")
        log_probs = input_logits # Placeholder, shape (batch_size, n_classes)

        # --- TODO: Step 2 - Create One-Hot Targets ---
        # Create a one-hot encoded version of the target numpy array.
        # Shape should be (batch_size, n_classes).
        # Hint: Use np.zeros and indexing like y_one_hot[np.arange(batch_size), target] = 1.0
        y_one_hot_np = np.zeros((batch_size, n_classes), dtype=np.float64)
        y_one_hot_np[np.arange(batch_size), target] = 1.0

        # --- TODO: Step 3 - Wrap One-Hot Targets ---
        # Wrap the y_one_hot_np numpy array in a Value object. This acts as a constant.
        y_one_hot_val = Value(y_one_hot_np)
      

        # --- TODO: Step 4 - Calculate NLL ---
        # Calculate the negative log likelihood using element-wise multiplication and summation.
        # NLL = -sum(y_one_hot * log_probs) over classes.
        #   a) Multiply the one-hot target Value by the log_probs Value element-wise.
        #   b) Negate the result.
        #   c) Sum the results over the class dimension (axis=1) to get loss per sample.
        nll_per_sample = -(y_one_hot_val * log_probs).sum(axis=1)  # shape: (batch_size,)

        # --- TODO: Step 5 - Apply Reduction ---
        # Apply the specified reduction ('mean', 'sum', or 'none') to nll_per_sample.
        # Hint: Use nll_per_sample.mean(), nll_per_sample.sum(), or return directly.
        if self.reduction == 'mean':
            loss = nll_per_sample.mean()
        elif self.reduction == 'sum':
            loss = nll_per_sample.sum()
        else:  # 'none'
            loss = nll_per_sample

        return loss

    def __repr__(self):
        """String representation."""
        return f"CrossEntropyLoss(reduction='{self.reduction}')"