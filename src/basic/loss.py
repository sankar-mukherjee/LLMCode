# - **Loss Functions:** Implement **Cross-Entropy** or **MSE** from scratch using PyTorch tensor operations (e.g., `torch.log`, `torch.sum`).
# - Implement **numerically stable Softmax**.
# - Implement **Binary Cross-Entropy (BCE) loss** from scratch.
# - Combine **Softmax + Cross-Entropy** into a single efficient operator.


import torch

def mse_loss(pred, target):
    # pred, target: same shape
    diff = pred - target
    loss = torch.mean(diff ** 2)
    return loss


def cross_entropy_loss(logits, targets):
    """
    Implement Categorical Cross-Entropy loss from scratch.
    
    This combines LogSoftmax + NLLLoss:
    1. Apply softmax to logits to get probabilities
    2. Take log of probabilities
    3. Select the log probability of the true class
    4. Negate and average
    
    Args:
        logits: Raw unnormalized scores (batch_size, num_classes)
        targets: Class indices (batch_size,)    
    Returns:
        Loss value
    """
    # Step 1: Compute log_softmax from scratch
    # For numerical stability, subtract max value
    max_logits = torch.max(logits, dim=1, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)
    sum_exp = torch.sum(exp_logits, dim=1, keepdim=True)
    log_softmax = (logits - max_logits) - torch.log(sum_exp)
    
    # Step 2: Gather the log probabilities of the true classes
    batch_size = logits.shape[0]
    # Create indices for gathering
    batch_indices = torch.arange(batch_size, device=logits.device)
    log_probs_of_true_class = log_softmax[batch_indices, targets]
    
    # Step 3: Negative log likelihood
    nll = -log_probs_of_true_class

    loss = torch.mean(nll)
    return loss


def binary_cross_entropy(pred, target, eps=1e-8):
    """
    pred: sigmoid output in [0,1]
    target: {0,1}
    """
    pred = torch.clamp(pred, eps, 1 - eps)
    loss = - (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    return loss.mean()


def softmax_from_scratch(logits, dim=-1):
    """
    Numerically stable softmax implementation.
    
    The naive implementation exp(x) / sum(exp(x)) can overflow/underflow.
    Stable version: exp(x - max(x)) / sum(exp(x - max(x)))
    
    This works because:
    softmax(x) = exp(x_i) / Σexp(x_j)
               = exp(x_i - c) / Σexp(x_j - c)  for any constant c
    
    We choose c = max(x) to prevent overflow.
    
    Args:
        logits: Input tensor of any shape
        dim: Dimension along which to apply softmax
    
    Returns:
        Probabilities (same shape as logits)
    """
    # Step 1: Subtract max for numerical stability
    # This prevents overflow when exponentiating large values
    max_logits = torch.max(logits, dim=dim, keepdim=True)[0]
    shifted_logits = logits - max_logits
    
    # Step 2: Compute exponentials
    exp_logits = torch.exp(shifted_logits)
    
    # Step 3: Normalize by sum
    sum_exp = torch.sum(exp_logits, dim=dim, keepdim=True)
    softmax_probs = exp_logits / sum_exp
    
    return softmax_probs

