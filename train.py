import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json # For saving loss history
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.load_data import load_node_classification_data
from models.gnn_model import GNNModelWithMLA, GNNModelBaseline, GATModelBaseline # Import all three models
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
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.') # Changed back to 64
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='CiteSeer', 
                      help='Dataset to use (Cora, CiteSeer, PubMed, Flickr, Reddit, ogbn-arxiv).')
parser.add_argument('--data_root', type=str, default='../data_cache', help='Directory for dataset cache.')
# MLA specific args
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads in MLA.') # Increased from 2
parser.add_argument('--kv_compress_dim', type=int, default=32, help='KV compression dimension in MLA.') # Increased from 16
parser.add_argument('--q_compress_dim', type=int, default=32, help='Query compression dimension in MLA.') # Increased from 16
parser.add_argument('--num_base_layers', type=int, default=1, help='Number of base GNN layers before MLA (used by MLA model).')
parser.add_argument('--use_pos_enc', action='store_true', default=False, help='Whether to use positional encoding in MLA model.')
# Model selection args
parser.add_argument('--model_type', type=str, default='mla', choices=['mla', 'gcn_baseline', 'gat_baseline'], help='Type of model to train (mla, gcn_baseline, gat_baseline).')
parser.add_argument('--gcn_layers', type=int, default=2, help='Number of layers for the GCN baseline model.') # Renamed from baseline_layers, default 2
parser.add_argument('--gat_layers', type=int, default=2, help='Number of layers for the GAT baseline model.') # Default 2
parser.add_argument('--gat_heads', type=int, default=4, help='Number of attention heads for the GAT baseline model.') # Reduced from 8
parser.add_argument('--equalize_params', action='store_true', default=False, help='Adjust model parameters (MLA, GCN, GAT) to be approximately equal.') # Updated help text

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
    
    # Increase GCN baseline parameters by adding more layers
    args.gcn_layers = 4 # Adjusted GCN layers
    # Adjust GAT baseline parameters for comparable count
    args.gat_layers = 3 # Increase layers similar to GCN
    args.gat_heads = 1  # Reduce heads significantly to control parameter count
    
    print("Parameter equalization enabled - adjusted MLA, GCN, and GAT configurations (GAT layers=3, heads=1)")

# Create all models to count parameters
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
                         num_layers=args.gcn_layers, # Use gcn_layers
                         dropout=args.dropout)

gat_baseline_model = GATModelBaseline(in_features=num_features,
                               hidden_channels=args.hidden,
                               out_features=num_classes,
                               num_layers=args.gat_layers,
                               heads=args.gat_heads,
                               dropout=args.dropout)


# Compare model parameters
print("\n--- Model Parameter Comparison ---")
print(f"MLA Model Parameters: {count_parameters(mla_model):,}")
print(f"GCN Baseline Model Parameters: {count_parameters(baseline_model):,}")
print(f"GAT Baseline Model Parameters: {count_parameters(gat_baseline_model):,}")
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
elif args.model_type == 'gcn_baseline':
    print(f"GCN Baseline Layers: {args.gcn_layers}")
    model = baseline_model  # Use already created GCN baseline model
elif args.model_type == 'gat_baseline':
    print(f"GAT Baseline Layers: {args.gat_layers}")
    print(f"GAT Baseline Heads: {args.gat_heads}")
    model = gat_baseline_model # Use already created GAT baseline model
else:
    raise ValueError(f"Unknown model type: {args.model_type}")

print(f"Number of parameters in selected {args.model_type.upper()} model: {count_parameters(model):,}")

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Learning rate scheduler for better stability
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

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
patience = 50 # Example patience for early stopping

# Lists to store loss history
train_losses = []
val_losses = []

for epoch in range(args.epochs):
    loss_train_epoch, loss_val_epoch = train(epoch)
    train_losses.append(loss_train_epoch)
    val_losses.append(loss_val_epoch)

    # Learning rate scheduling
    scheduler.step(loss_val_epoch)

    # Simple early stopping based on validation loss
    if loss_val_epoch < best_val_loss:
        best_val_loss = loss_val_epoch # Corrected variable name
        epochs_no_improve = 0
        # Optionally save the best model
        # torch.save(model.state_dict(), f'{args.model_type}_best_model.pth')
    else:
        epochs_no_improve += 1
    if epochs_no_improve == patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

print(f"\n--- Optimization Finished: {args.model_type.upper()} Model ---")
print(f"Total training time: {time.time() - t_total:.4f}s")

# Load best model if early stopping was used and model saved
# if epochs_no_improve == patience: # Keep this commented unless saving/loading best model is implemented
#     print("Loading best model for testing...")
#     model.load_state_dict(torch.load(f'{args.model_type}_best_model.pth'))

# Testing
test_accuracy = test()

print(f"\nFinal Test Accuracy for {args.model_type.upper()} model: {test_accuracy:.4f}")

# Save loss history
loss_history_file = f'loss_history_{args.dataset}_{args.model_type}.npz'
np.savez(loss_history_file, train_losses=np.array(train_losses), val_losses=np.array(val_losses))
print(f"Loss history saved to {loss_history_file}")
