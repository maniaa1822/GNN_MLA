import torch

def accuracy(output, labels):
    """
    Calculates accuracy for node classification.

    Args:
        output (torch.Tensor): Model output logits or log-probabilities (Nodes x Classes).
        labels (torch.Tensor): Ground truth labels (Nodes).

    Returns:
        float: Accuracy percentage.
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels) * 100.0

# Add other utility functions here if needed, e.g.,
# - Early stopping logic
# - Plotting functions
# - Configuration loading
