import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os.path as osp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

# Assuming load_data is correctly modified in data.load_data
from data.load_data import load_data
# Import the new QM9 models
from models.gnn_model import QM9GraphRegGAT, QM9GraphRegMLA

# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to calculate unnormalized MAE
def calculate_unnormalized_mae(predictions, targets, target_stds):
    """Calculates the Mean Absolute Error and unnormalizes it."""
    # Ensure target_stds is on the same device as predictions/targets
    target_stds = target_stds.to(predictions.device)
    # Calculate MAE per target (L1 Loss)
    mae_per_target = torch.abs(predictions - targets)
    # Unnormalize: Multiply MAE by std dev for each target
    unnormalized_mae_per_target = mae_per_target * target_stds
    # Average across all targets and samples in the batch/epoch
    return unnormalized_mae_per_target.mean().item()

# --- Modified train/evaluate functions to select targets ---

def train_epoch(model, loader, criterion, optimizer, device, target_indices_tensor):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        # Select target columns
        targets = batch.y[:, target_indices_tensor.to(device)] # Ensure indices are on correct device
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, targets) # Compare with selected targets
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, selected_target_stds, device, target_indices_tensor):
    model.eval()
    total_mae = 0
    total_loss = 0 # Also track normalized loss if needed
    for batch in loader:
        batch = batch.to(device)
        # Select target columns
        targets = batch.y[:, target_indices_tensor.to(device)] # Ensure indices are on correct device
        out = model(batch)
        loss = criterion(out, targets) # Compare with selected targets
        # Calculate MAE using selected targets and their corresponding std devs
        mae = calculate_unnormalized_mae(out, targets, selected_target_stds)
        total_loss += loss.item() * batch.num_graphs
        total_mae += mae * batch.num_graphs # Accumulate unnormalized MAE weighted by batch size
    avg_loss = total_loss / len(loader.dataset)
    avg_mae = total_mae / len(loader.dataset)
    return avg_loss, avg_mae


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='QM9 Graph Regression Comparison (GAT vs MLA)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.') # Reduced default
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--data_root', type=str, default='../data_cache', help='Directory for dataset cache.')
    # GAT specific args
    parser.add_argument('--gat_layers', type=int, default=2, help='Number of layers for the GAT model.') # Reduced default
    parser.add_argument('--gat_heads', type=int, default=4, help='Number of attention heads for the GAT model.') # Reduced default
    # MLA specific args
    parser.add_argument('--mla_heads', type=int, default=2, help='Number of attention heads in MLA.') # Reduced default
    parser.add_argument('--mla_kv_dim', type=int, default=16, help='KV compression dimension in MLA.') # Reduced default
    parser.add_argument('--mla_q_dim', type=int, default=16, help='Query compression dimension in MLA.') # Reduced default
    parser.add_argument('--mla_base_layers', type=int, default=1, help='Number of base GNN layers before MLA.')
    parser.add_argument('--mla_layers', type=int, default=1, help='Number of MLA layers.')
    parser.add_argument('--target_indices', type=int, nargs='+', default=None,
                        help='Indices of QM9 targets to predict (0-18). Predicts all if None.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader.')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    print("--- Configuration ---")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Hidden Dim: {args.hidden}")
    print(f"Dropout: {args.dropout}")
    print("---------------------\n")

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load QM9 data
    # load_data now returns: dataset, num_node_features, num_total_targets, dataset_obj
    dataset_obj, num_node_features, num_total_targets, _ = load_data('QM9', args.data_root)
    if dataset_obj is None:
        print("Failed to load QM9 dataset.")
        exit()

    # Determine target indices and number of targets to predict
    if args.target_indices is None:
        target_indices = list(range(num_total_targets))
        print(f"Predicting all {num_total_targets} targets.")
    else:
        target_indices = sorted(list(set(args.target_indices))) # Ensure unique and sorted
        if any(i < 0 or i >= num_total_targets for i in target_indices):
             raise ValueError(f"Invalid target indices. Must be between 0 and {num_total_targets - 1}.")
        print(f"Predicting targets with indices: {target_indices}")
    num_selected_targets = len(target_indices)
    # Create tensor on CPU first, then move to device later if needed inside loops
    target_indices_tensor = torch.tensor(target_indices, dtype=torch.long)

    # Retrieve target statistics (mean/std) - crucial for unnormalization
    # PyG QM9 stores targets normalized. We need std dev for reporting MAE in original scale.
    # Calculate std dev directly from the dataset targets.
    print("Calculating target standard deviations from the dataset...")
    try:
        # Concatenate all target tensors from the dataset
        # This might be memory-intensive for very large datasets, but QM9 should be manageable.
        # Concatenate all target tensors from the dataset
        all_y = torch.cat([data.y for data in dataset_obj], dim=0)
        all_target_stds = all_y.std(dim=0) # Calculate std dev for ALL targets first
        # Select the std devs for the chosen targets
        selected_target_stds = all_target_stds[target_indices_tensor].to(device) # Select stds and move to device
        print(f"Using calculated standard deviations for selected targets (shape: {selected_target_stds.shape})")
        # Ensure no selected target has std dev of 0
        selected_target_stds[selected_target_stds == 0] = 1.0
    except Exception as e:
        print(f"Error calculating target std devs: {e}. Using placeholder ones.")
        selected_target_stds = torch.ones(num_selected_targets, device=device) # Fallback placeholder


    # Split data (standard QM9 split)
    # ~134k graphs total. Split: 110k train, 10k val, remaining test
    perm = torch.randperm(len(dataset_obj), generator=torch.Generator().manual_seed(args.seed))
    train_idx = perm[:110000]
    val_idx = perm[110000:120000]
    test_idx = perm[120000:]

    train_dataset = dataset_obj[train_idx]
    val_dataset = dataset_obj[val_idx]
    test_dataset = dataset_obj[test_idx]

    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # --- Create DataLoaders with multiple workers ---
    print(f"Using {args.num_workers} workers for DataLoaders.")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.cuda)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.cuda)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.cuda)

    # Models to compare
    models_to_train = {
        'GAT': QM9GraphRegGAT(node_feature_dim=num_node_features,
                              edge_feature_dim=dataset_obj.num_edge_features,
                              hidden_channels=args.hidden,
                              out_features=num_selected_targets, # Use selected number of targets
                              num_layers=args.gat_layers,
                              heads=args.gat_heads,
                              dropout=args.dropout).to(device),
        'MLA': QM9GraphRegMLA(node_feature_dim=num_node_features,
                              edge_feature_dim=dataset_obj.num_edge_features,
                              hidden_channels=args.hidden,
                              out_features=num_selected_targets, # Use selected number of targets
                              num_heads=args.mla_heads,
                              kv_compression_dim=args.mla_kv_dim,
                              q_compression_dim=args.mla_q_dim,
                              num_base_layers=args.mla_base_layers,
                              num_mla_layers=args.mla_layers,
                              dropout=args.dropout).to(device)
    }

    # Loss criterion (MAE)
    criterion = nn.L1Loss()

    results = {}

    print("\n--- Model Parameter Counts ---")
    for name, model in models_to_train.items():
        print(f"{name}: {count_parameters(model):,} parameters")
    print("----------------------------\n")


    # Training loop for each model
    for model_name, model in models_to_train.items():
        print(f"--- Training {model_name} Model ---")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        best_val_mae = float('inf')
        epochs_no_improve = 0
        patience = 30 # Early stopping patience

        t_start_train = time.time()
        for epoch in range(args.epochs):
            t_epoch_start = time.time()
            # Pass target_indices_tensor to train_epoch
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, target_indices_tensor)
            # Pass selected_target_stds and target_indices_tensor to evaluate
            val_loss, val_mae = evaluate(model, val_loader, criterion, selected_target_stds, device, target_indices_tensor)
            epoch_time = time.time() - t_epoch_start

            print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Val MAE (Unnorm): {val_mae:.4f}, Time: {epoch_time:.2f}s')

            scheduler.step(val_mae) # Step scheduler based on unnormalized MAE

            # Early stopping
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                epochs_no_improve = 0
                # Optionally save best model state
                # torch.save(model.state_dict(), f'qm9_{model_name}_best.pth')
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered for {model_name} after {epoch+1} epochs.")
                break

        train_time = time.time() - t_start_train
        print(f"--- Finished Training {model_name} ({train_time:.2f}s) ---")

        # Load best model if saved
        # if epochs_no_improve >= patience:
        #     model.load_state_dict(torch.load(f'qm9_{model_name}_best.pth'))

        # Final Test Evaluation
        print(f"--- Evaluating {model_name} on Test Set ---")
        # Pass selected_target_stds and target_indices_tensor to evaluate
        test_loss, test_mae = evaluate(model, test_loader, criterion, selected_target_stds, device, target_indices_tensor)
        print(f"Test Loss (Norm): {test_loss:.4f}")
        print(f"Test MAE (Unnorm - selected targets): {test_mae:.4f}") # Clarify MAE is for selected targets
        print("----------------------------------------\n")
        results[model_name] = {'test_mae': test_mae, 'test_loss': test_loss, 'train_time': train_time}

    # Print final comparison
    print("--- Final Comparison Results (Test Set) ---")
    for name, metrics in results.items():
        print(f"{name}: MAE (Unnorm) = {metrics['test_mae']:.4f}, Loss (Norm) = {metrics['test_loss']:.4f}, Time = {metrics['train_time']:.2f}s")
    print("-------------------------------------------")

if __name__ == '__main__':
    main()
