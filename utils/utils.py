import torch
from sklearn.metrics import f1_score

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

def macro_f1_score(output, labels):
    """
    Calculates the Macro F1-score for node classification.

    Args:
        output (torch.Tensor): Model output logits or log-probabilities (Nodes x Classes).
        labels (torch.Tensor): Ground truth labels (Nodes).

    Returns:
        float: Macro F1-score.
    """
    preds = output.max(1)[1].type_as(labels)
    # Move tensors to CPU and convert to numpy for sklearn
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    return f1_score(labels_np, preds_np, average='macro')

# Add other utility functions here if needed, e.g.,
# - Early stopping logic
# - Plotting functions
# - Configuration loading

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
