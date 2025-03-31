import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json # For saving loss history

from data.load_data import load_node_classification_data
from models.gnn_model import GNNModelWithMLA, GNNModelBaseline # Import both models
from utils.utils import accuracy

# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='Cora', help='Dataset to use (Cora, CiteSeer, PubMed).')
parser.add_argument('--data_root', type=str, default='../data_cache', help='Directory for dataset cache.')
# MLA specific args
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads in MLA.')
parser.add_argument('--kv_compress_dim', type=int, default=32, help='KV compression dimension in MLA.')
parser.add_argument('--q_compress_dim', type=int, default=32, help='Query compression dimension in MLA.')
parser.add_argument('--num_base_layers', type=int, default=1, help='Number of base GNN layers before MLA (used by MLA model).')
parser.add_argument('--use_pos_enc', action='store_true', default=False, help='Whether to use positional encoding in MLA model.')
# Model selection args
parser.add_argument('--model_type', type=str, default='mla', choices=['mla', 'baseline'], help='Type of model to train (mla or baseline).')
parser.add_argument('--baseline_layers', type=int, default=4, help='Number of layers for the baseline GNN model.')
parser.add_argument('--equalize_params', action='store_true', default=False, help='Adjust model parameters to be approximately equal.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
data, num_features, num_classes = load_node_classification_data(args.dataset, args.data_root)
if data is None:
    print(f"Failed to load dataset: {args.dataset}")
    exit()

# Parameter balancing logic
if args.equalize_params:
    # For CORA dataset with default settings (hidden_dim=64):
    # - Estimate baseline params: ~4 × 64 × 64 + output params
    # - MLA params depend heavily on positional encodings (nodes × hidden_dim) and compression layers
    
    # Reduce MLA parameters by:
    # 1. Disabling positional encoding
    args.use_pos_enc = False
    # 2. Reducing compression dimensions
    args.kv_compress_dim = 32
    args.q_compress_dim = 32
    # 3. Reducing number of attention heads
    args.num_heads = 4
    
    # Increase baseline parameters by adding more layers
    args.baseline_layers = 4
    
    print("Parameter equalization enabled - adjusted model configurations")

# Create both models to count parameters
mla_model = GNNModelWithMLA(in_features=num_features,
                        hidden_channels=args.hidden,
                        out_features=num_classes,
                        num_nodes=data.num_nodes,
                        num_heads=args.num_heads,
                        kv_compression_dim=args.kv_compress_dim,
                        q_compression_dim=args.q_compress_dim,
                        num_base_layers=args.num_base_layers,
                        dropout=args.dropout,
                        use_pos_enc=args.use_pos_enc)  # Pass the flag

baseline_model = GNNModelBaseline(in_features=num_features,
                         hidden_channels=args.hidden,
                         out_features=num_classes,
                         num_layers=args.baseline_layers,
                         dropout=args.dropout)

# Compare model parameters
print("\n--- Model Parameter Comparison ---")
print(f"MLA Model Parameters: {count_parameters(mla_model):,}")
print(f"Baseline Model Parameters: {count_parameters(baseline_model):,}")
print("--------------------------------\n")

# Model and optimizer
print(f"--- Training Configuration ---")
print(f"Dataset: {args.dataset}")
print(f"Model Type: {args.model_type.upper()}")
print(f"Epochs: {args.epochs}")
print(f"Learning Rate: {args.lr}")
print(f"Weight Decay: {args.weight_decay}")
print(f"Hidden Units: {args.hidden}")
print(f"Dropout: {args.dropout}")
print(f"Seed: {args.seed}")
print(f"CUDA Enabled: {args.cuda}")

if args.model_type == 'mla':
    print(f"MLA Heads: {args.num_heads}")
    print(f"MLA KV Dim: {args.kv_compress_dim}")
    print(f"MLA Q Dim: {args.q_compress_dim}")
    print(f"MLA Base Layers: {args.num_base_layers}")
    model = mla_model  # Use already created MLA model
elif args.model_type == 'baseline':
    print(f"Baseline Layers: {args.baseline_layers}")
    model = baseline_model  # Use already created baseline model
else:
    raise ValueError(f"Unknown model type: {args.model_type}")

print(f"Number of parameters in {args.model_type.upper()} model: {count_parameters(model):,}")

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    print("Moving model and data to CUDA device.")
    model.cuda()
    data = data.to('cuda')

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    acc_train = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad(): # Disable gradient calculation for validation
        output = model(data)
        loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
        acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])

    print(f'Epoch: {epoch+1:04d}',
          f'loss_train: {loss_train.item():.4f}',
          f'acc_train: {acc_train.item():.4f}',
          f'loss_val: {loss_val.item():.4f}',
          f'acc_val: {acc_val.item():.4f}',
          f'time: {time.time() - t:.4f}s')
    # Return training and validation loss for recording
    return loss_train.item(), loss_val.item()

def test():
    model.eval()
    with torch.no_grad(): # Disable gradient calculation for testing
        output = model(data)
        loss_test = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
        acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])

    print(f"\n--- {args.model_type.upper()} Model Test Results ---")
    print(f"Test Loss: {loss_test.item():.4f}")
    print(f"Test Accuracy: {acc_test.item():.4f}")
    print(f"------------------------------------")
    return acc_test.item() # Return test accuracy for comparison

# Train model
t_total = time.time()
print(f"\n--- Starting Training: {args.model_type.upper()} Model ---")
best_val_loss = float('inf')
epochs_no_improve = 0
patience = 10 # Example patience for early stopping

# Lists to store loss history
train_losses = []
val_losses = []

for epoch in range(args.epochs):
    loss_train_epoch, loss_val_epoch = train(epoch)
    train_losses.append(loss_train_epoch)
    val_losses.append(loss_val_epoch)

    # Simple early stopping based on validation loss
    # if loss_val_epoch < best_val_loss:
    #     best_val_loss = val_loss
    #     epochs_no_improve = 0
    #     # Optionally save the best model
    #     # torch.save(model.state_dict(), f'{args.model_type}_best_model.pth')
    # else:
    #     epochs_no_improve += 1
    # if epochs_no_improve == patience:
    #     print(f"Early stopping triggered after {epoch+1} epochs.")
    #     break

print(f"\n--- Optimization Finished: {args.model_type.upper()} Model ---")
print(f"Total training time: {time.time() - t_total:.4f}s")

# Load best model if early stopping was used and model saved
# if epochs_no_improve == patience:
#     print("Loading best model for testing...")
#     model.load_state_dict(torch.load(f'{args.model_type}_best_model.pth'))

# Testing
test_accuracy = test()

print(f"\nFinal Test Accuracy for {args.model_type.upper()} model: {test_accuracy:.4f}")

# Save loss history
loss_history_file = f'loss_history_{args.dataset}_{args.model_type}.npz'
np.savez(loss_history_file, train_losses=np.array(train_losses), val_losses=np.array(val_losses))
print(f"Loss history saved to {loss_history_file}")
