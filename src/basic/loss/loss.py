"""
Loss Functions Implementation
- MSE Loss: Mean squared error for regression tasks
- Cross-Entropy Loss: Numerically stable implementation
- Binary Cross-Entropy (BCE): For binary classification
- KL Divergence: For Gaussian distributions (VAEs, probabilistic models)
- Cosine Similarity Loss: For embedding similarity
- Contrastive Loss: For metric learning
"""

import torch
import torch.nn as nn


def mse_loss(pred, target):
    diff = pred - target
    return torch.mean(diff ** 2)


def softmax(logits, dim=-1):
    """Numerically stable softmax."""
    max_logits = torch.max(logits, dim=dim, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)
    sum_exp = torch.sum(exp_logits, dim=dim, keepdim=True)
    return exp_logits / sum_exp


def log_softmax(logits, dim=-1):
    """Numerically stable log-softmax."""
    max_logits = torch.max(logits, dim=dim, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)
    sum_exp = torch.sum(exp_logits, dim=dim, keepdim=True)
    return (logits - max_logits) - torch.log(sum_exp)


def cross_entropy_loss(logits, targets):
    """
    Categorical Cross-Entropy loss (LogSoftmax + NLLLoss).
    
    Args:
        logits: (batch_size, num_classes)
        targets: (batch_size,) class indices
    """
    log_probs = log_softmax(logits, dim=1)
    batch_size = logits.shape[0]
    batch_indices = torch.arange(batch_size, device=logits.device)
    log_probs_of_true_class = log_probs[batch_indices, targets]
    return -torch.mean(log_probs_of_true_class)


def binary_cross_entropy(pred, target, eps=1e-8):
    """
    Binary cross-entropy for sigmoid outputs.
    
    Args:
        pred: Sigmoid output in [0,1]
        target: Binary labels {0,1}
    """
    pred = torch.clamp(pred, eps, 1 - eps)
    loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    return loss.mean()


def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    """
    KL divergence between two normal distributions: KL(P||Q)
        P ~ N(mu_p, sigma_p^2)
        Q ~ N(mu_q, sigma_q^2)
    """
    if torch.any(sigma_p <= 0) or torch.any(sigma_q <= 0):
        raise ValueError("Standard deviations must be positive.")

    term1 = torch.log(sigma_q / sigma_p)
    term2 = (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2)
    return term1 + term2 - 0.5


def cosine_similarity_loss(x1, x2, eps=1e-8):
    """
    Cosine similarity loss for embeddings.
    
    Args:
        x1, x2: (batch_size, dim) embeddings
    """
    dot = torch.sum(x1 * x2, dim=1)
    norm_x1 = torch.sqrt(torch.sum(x1**2, dim=1) + eps)
    norm_x2 = torch.sqrt(torch.sum(x2**2, dim=1) + eps)
    cos_sim = dot / (norm_x1 * norm_x2)
    return (1.0 - cos_sim).mean()


def contrastive_loss(x1, x2, labels, margin=1.0):
    """
    Contrastive loss for metric learning.
    
    Args:
        x1, x2: (batch_size, dim) embeddings
        labels: (batch_size,) 1=similar, 0=dissimilar
        margin: Margin for dissimilar pairs
    """
    distances = torch.norm(x1 - x2, dim=1)
    pos_loss = labels * distances.pow(2)
    neg_loss = (1 - labels) * torch.clamp(margin - distances, min=0).pow(2)
    return (pos_loss + neg_loss).mean()


if __name__ == "__main__":
    print("="*50)
    print("Testing Loss Functions")
    print("="*50)
    
    # Test 1: Softmax
    print("\n[Test 1] Softmax")
    print("-" * 50)
    logits = torch.randn(4, 10)
    
    custom = softmax(logits, dim=1)
    pytorch = torch.softmax(logits, dim=1)
    diff = (custom - pytorch).abs().max().item()
    
    print(f"  Sum of probs: {custom.sum(dim=1)}")
    print(f"  Diff: {diff:.8f} {'✓ PASS' if diff < 1e-6 else '✗ FAIL'}")
    
    # Test 2: Log-Softmax
    print("\n[Test 2] Log-Softmax")
    print("-" * 50)
    custom = log_softmax(logits, dim=1)
    pytorch = torch.log_softmax(logits, dim=1)
    diff = (custom - pytorch).abs().max().item()
    
    print(f"  Diff: {diff:.8f} {'✓ PASS' if diff < 1e-6 else '✗ FAIL'}")
    
    # Test 3: MSE Loss
    print("\n[Test 3] MSE Loss")
    print("-" * 50)
    pred = torch.randn(10, 5)
    target = torch.randn(10, 5)
    
    custom = mse_loss(pred, target)
    pytorch = nn.MSELoss()(pred, target)
    diff = (custom - pytorch).abs().item()
    
    print(f"  Custom: {custom:.6f}, PyTorch: {pytorch:.6f}")
    print(f"  Diff: {diff:.8f} {'✓ PASS' if diff < 1e-6 else '✗ FAIL'}")
    
    # Test 4: Cross-Entropy Loss
    print("\n[Test 4] Cross-Entropy Loss")
    print("-" * 50)
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))
    
    custom = cross_entropy_loss(logits, targets)
    pytorch = nn.CrossEntropyLoss()(logits, targets)
    diff = (custom - pytorch).abs().item()
    
    print(f"  Custom: {custom:.6f}, PyTorch: {pytorch:.6f}")
    print(f"  Diff: {diff:.8f} {'✓ PASS' if diff < 1e-6 else '✗ FAIL'}")
    
    # Test 5: Binary Cross-Entropy
    print("\n[Test 5] Binary Cross-Entropy")
    print("-" * 50)
    pred = torch.sigmoid(torch.randn(10, 1))
    target = torch.randint(0, 2, (10, 1)).float()
    
    custom = binary_cross_entropy(pred, target)
    pytorch = nn.BCELoss()(pred, target)
    diff = (custom - pytorch).abs().item()
    
    print(f"  Custom: {custom:.6f}, PyTorch: {pytorch:.6f}")
    print(f"  Diff: {diff:.8f} {'✓ PASS' if diff < 1e-6 else '✗ FAIL'}")
    
    # Test 6: KL Divergence
    print("\n[Test 6] KL Divergence (Gaussian)")
    print("-" * 50)
    mu_p = torch.tensor([0.0, 1.0])
    sigma_p = torch.tensor([1.0, 2.0])
    mu_q = torch.tensor([0.5, 0.5])
    sigma_q = torch.tensor([1.5, 1.0])
    
    custom = kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q)
    
    # PyTorch KL divergence for normal distributions
    p = torch.distributions.Normal(mu_p, sigma_p)
    q = torch.distributions.Normal(mu_q, sigma_q)
    pytorch = torch.distributions.kl_divergence(p, q)
    diff = (custom - pytorch).abs().max().item()
    
    print(f"  Custom: {custom}")
    print(f"  PyTorch: {pytorch}")
    print(f"  Diff: {diff:.8f} {'✓ PASS' if diff < 1e-6 else '✗ FAIL'}")
    
    # Test 7: Cosine Similarity Loss
    print("\n[Test 7] Cosine Similarity Loss")
    print("-" * 50)
    x1 = torch.randn(8, 128)
    x2 = torch.randn(8, 128)
    
    custom = cosine_similarity_loss(x1, x2)
    pytorch = 1 - nn.functional.cosine_similarity(x1, x2).mean()
    diff = (custom - pytorch).abs().item()
    
    print(f"  Custom: {custom:.6f}, PyTorch: {pytorch:.6f}")
    print(f"  Diff: {diff:.8f} {'✓ PASS' if diff < 1e-6 else '✗ FAIL'}")
    
    # Test 8: Contrastive Loss
    print("\n[Test 8] Contrastive Loss")
    print("-" * 50)
    x1 = torch.randn(8, 128)
    x2 = torch.randn(8, 128)
    labels = torch.randint(0, 2, (8,)).float()
    margin = 1.0
    
    custom = contrastive_loss(x1, x2, labels, margin)
    
    # PyTorch reference implementation
    distances = torch.norm(x1 - x2, dim=1)
    pos_loss = labels * distances.pow(2)
    neg_loss = (1 - labels) * torch.clamp(margin - distances, min=0).pow(2)
    pytorch = (pos_loss + neg_loss).mean()
    diff = (custom - pytorch).abs().item()
    
    print(f"  Custom: {custom:.6f}, PyTorch: {pytorch:.6f}")
    print(f"  Diff: {diff:.8f} {'✓ PASS' if diff < 1e-6 else '✗ FAIL'}")
    
    print("\n" + "="*50)
    print("All tests complete!")
    print("="*50)
