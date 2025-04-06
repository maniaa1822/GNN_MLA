import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt # Added for plotting
import os # Added for path joining
import os.path as osp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

# Assuming load_data is correctly modified in data.load_data
from load_data import load_data
# Import the new QM9 models
from models.gnn_model import QM9GraphRegGAT, QM9GraphRegMLA, QM9GraphRegMLA_NoBases

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

# --- Modified train/evaluate functions to select targets and return MAE ---

def train_epoch(model, loader, criterion, optimizer, selected_target_stds, device, target_indices_tensor):
    model.train()
    total_loss = 0
    total_mae = 0
    for batch in loader:
        batch = batch.to(device)
        # Select target columns
        targets = batch.y[:, target_indices_tensor.to(device)] # Ensure indices are on correct device
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, targets) # Compare with selected targets
        loss.backward()
        optimizer.step()
        # Calculate unnormalized MAE for the batch
        mae = calculate_unnormalized_mae(out.detach(), targets, selected_target_stds) # Use detach() for MAE calc
        total_loss += loss.item() * batch.num_graphs
        total_mae += mae * batch.num_graphs # Accumulate unnormalized MAE
    avg_loss = total_loss / len(loader.dataset)
    avg_mae = total_mae / len(loader.dataset)
    return avg_loss, avg_mae

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


# --- Plotting Function (Adapted from model_comparison.py) ---
def plot_qm9_comparison(results_dict, save_path=None, param_counts=None, target_indices=None):
    """Plot comparison of performance metrics (Loss and MAE) for QM9 models"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12)) # Adjusted size

    # Define colors and labels for models
    colors = {
        'GAT': 'r', 
        'MLA': 'b',
        'MLA-NoBases': 'g'
    }
    labels = {
        'GAT': 'GAT Baseline', 
        'MLA': 'MLA with GCN Base',
        'MLA-NoBases': 'MLA without GCN Base'
    }

    for model_type, results in results_dict.items():
        if results is None or not all(k in results for k in ['train_losses', 'val_losses', 'train_maes', 'val_maes']):
            print(f"Skipping plotting for {model_type}: Missing history data.")
            continue # Skip if results or history are missing

        epochs = range(1, len(results['train_losses']) + 1)
        color = colors.get(model_type, 'k') # Default to black if type unknown

        # If parameter counts are provided, include them in the label
        if param_counts and model_type in param_counts:
            label = f"{labels.get(model_type, model_type)} ({param_counts[model_type]:,} params)"
        else:
            label = labels.get(model_type, model_type)

        # Plotting: Train Loss, Val Loss, Train MAE, Val MAE
        ax1.plot(epochs, results['train_losses'], color=color, linestyle='-', label=label)
        ax2.plot(epochs, results['val_losses'], color=color, linestyle='-', label=label)
        ax3.plot(epochs, results['train_maes'], color=color, linestyle='-', label=label)
        ax4.plot(epochs, results['val_maes'], color=color, linestyle='-', label=label)

    ax1.set_title('Training Loss (Normalized)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Validation Loss (Normalized)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    ax3.set_title('Training MAE (Unnormalized)')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('MAE')
    ax3.legend()
    ax3.grid(True)

    ax4.set_title('Validation MAE (Unnormalized)')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('MAE')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()

    # Construct title
    title_parts = ["QM9 Performance Comparison"]
    if target_indices:
        title_parts.append(f"(Targets: {target_indices})")
    if param_counts:
        title_parts.append("(Equalized Parameters approx.)") # Assuming params are roughly equalized
    plt.suptitle(' '.join(title_parts), fontsize=16)

    plt.subplots_adjust(top=0.92) # Adjust title position

    if save_path:
        # Ensure directory exists if save_path includes directories
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig) # Close the figure after saving/showing


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='QM9 Graph Regression Comparison (GAT vs MLA)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.') # Reduced default
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--data_root', type=str, default='../data_cache', help='Directory for dataset cache.')
    # GAT specific args
    parser.add_argument('--gat_layers', type=int, default=2, help='Number of layers for the GAT model.') # Reduced default
    parser.add_argument('--gat_heads', type=int, default=4, help='Number of attention heads for the GAT model.') # Reduced default
    # MLA specific args
    parser.add_argument('--mla_heads', type=int, default=2, help='Number of attention heads in MLA.') # Reduced default
    parser.add_argument('--mla_kv_dim', type=int, default=8, help='KV compression dimension in MLA.') # Reduced default
    parser.add_argument('--mla_q_dim', type=int, default=8, help='Query compression dimension in MLA.') # Reduced default
    parser.add_argument('--mla_base_layers', type=int, default=1, help='Number of base GNN layers before MLA.')
    parser.add_argument('--mla_layers', type=int, default=1, help='Number of MLA layers.')
    parser.add_argument('--mla_nobases_layers', type=int, default=2, help='Number of MLA layers in the MLA-NoBases model.')
    parser.add_argument('--target_indices', type=int, nargs='+', default=[1,2,3],
                        help='Indices of QM9 targets to predict (0-18). Predicts all if None.')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for DataLoader.')
    parser.add_argument('--skip_train', action='store_true', default=False,
                        help='Skip training and only plot existing results') # Added skip_train

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
                              dropout=args.dropout).to(device),
        'MLA-NoBases': QM9GraphRegMLA_NoBases(
                              node_feature_dim=num_node_features,
                              edge_feature_dim=dataset_obj.num_edge_features,
                              hidden_channels=args.hidden,
                              out_features=num_selected_targets, # Use selected number of targets
                              num_heads=args.mla_heads,
                              kv_compression_dim=args.mla_kv_dim,
                              q_compression_dim=args.mla_q_dim,
                              num_mla_layers=args.mla_nobases_layers,
                              dropout=args.dropout).to(device)
    }

    # Loss criterion (MAE)
    criterion = nn.L1Loss()

    results = {}

    print("\n--- Model Parameter Counts ---")
    for name, model in models_to_train.items():
        print(f"{name}: {count_parameters(model):,} parameters")
    print("----------------------------\n")

    # --- Training or Loading Results ---
    results_file = f"QM9_comparison_results_{'_'.join(sorted(models_to_train.keys()))}.npz"
    param_counts_dict = {name: count_parameters(model) for name, model in models_to_train.items()} # Store param counts

    if not args.skip_train:
        print(f"\n--- Starting Training Loop ---")
        results = {} # Initialize results dict
        # Training loop for each model
        for model_name, model in models_to_train.items():
            print(f"--- Training {model_name} Model ---")
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

            best_val_mae = float('inf')
            epochs_no_improve = 0
            patience = 30 # Early stopping patience
            best_model_state = None

            # History tracking
            train_losses, val_losses, train_maes, val_maes = [], [], [], []

            t_start_train = time.time()
            for epoch in range(args.epochs):
                t_epoch_start = time.time()
                # Pass selected_target_stds and target_indices_tensor to train_epoch
                train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, selected_target_stds, device, target_indices_tensor)
                # Pass selected_target_stds and target_indices_tensor to evaluate
                val_loss, val_mae = evaluate(model, val_loader, criterion, selected_target_stds, device, target_indices_tensor)
                epoch_time = time.time() - t_epoch_start

                # Store history
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_maes.append(train_mae)
                val_maes.append(val_mae)

                print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Time: {epoch_time:.2f}s')

                scheduler.step(val_mae) # Step scheduler based on unnormalized MAE

                # Early stopping
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    epochs_no_improve = 0
                    best_model_state = model.state_dict().copy() # Save best model state
                    # Optionally save best model state to file immediately
                    # torch.save(model.state_dict(), f'qm9_{model_name}_best.pth')
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered for {model_name} after {epoch+1} epochs.")
                    break

            train_time = time.time() - t_start_train
            print(f"--- Finished Training {model_name} ({train_time:.2f}s) ---")

            # Load best model state for testing
            if best_model_state:
                model.load_state_dict(best_model_state)
                print("Loaded best model state for final evaluation.")
            # Or load from file if saved immediately:
            # if os.path.exists(f'qm9_{model_name}_best.pth'):
            #     model.load_state_dict(torch.load(f'qm9_{model_name}_best.pth'))

            # Final Test Evaluation
            print(f"--- Evaluating {model_name} on Test Set ---")
            # Pass selected_target_stds and target_indices_tensor to evaluate
            test_loss, test_mae = evaluate(model, test_loader, criterion, selected_target_stds, device, target_indices_tensor)
            print(f"Test Loss (Norm): {test_loss:.4f}")
            print(f"Test MAE (Unnorm - selected targets): {test_mae:.4f}") # Clarify MAE is for selected targets
            print("----------------------------------------\n")

            # Store results including history
            results[model_name] = {
                'test_mae': test_mae,
                'test_loss': test_loss,
                'train_time': train_time,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_maes': train_maes,
                'val_maes': val_maes,
                'epochs_trained': len(train_losses)
            }

        # Save results after training all models
        save_data = {
            **results,
            'parameters': param_counts_dict,
            'config': vars(args)
        }
        np.savez(results_file, **save_data)
        print(f"Results saved to {results_file}")

    else: # Skip training, load results
        print(f"\n--- Skipping Training - Loading Results from {results_file} ---")
        if osp.exists(results_file):
            try:
                loaded_data = np.load(results_file, allow_pickle=True)
                # Load results only for the models defined in models_to_train
                results = {}
                loaded_successfully = False
                for model_name in models_to_train.keys():
                    if model_name in loaded_data:
                        results[model_name] = loaded_data[model_name].item()
                        loaded_successfully = True

                if loaded_successfully:
                    print(f"Loaded results for {list(results.keys())}")
                    # Load param counts if available in the file
                    if 'parameters' in loaded_data:
                         param_counts_dict = loaded_data['parameters'].item()
                else:
                    print(f"No results found for {list(models_to_train.keys())} in {results_file}")
                    results = {} # Ensure results is empty if loading failed

            except Exception as e:
                print(f"Error loading results: {e}")
                results = {} # Ensure results is empty on error
        else:
            print(f"Results file not found: {results_file}. Cannot load results.")
            results = {} # Ensure results is empty if file not found

    # --- Plotting and Final Comparison (only if results exist) ---
    if not results:
        print("No results available to plot or compare. Exiting.")
        return

    # Plot comparison
    plot_save_path = f"QM9_comparison_plot_{'_'.join(sorted(results.keys()))}.png"
    plot_qm9_comparison(results, plot_save_path, param_counts=param_counts_dict, target_indices=target_indices)

    # Print final comparison
    print("\n--- Final Comparison Results (Test Set MAE - Unnormalized) ---")
    best_model_name = None
    best_mae = float('inf')
    for name, metrics in results.items():
        mae = metrics.get('test_mae', float('inf')) # Use .get for safety if loading partial results
        loss = metrics.get('test_loss', float('inf'))
        time_val = metrics.get('train_time', float('nan'))
        print(f"{name}: MAE = {mae:.4f}, Loss = {loss:.4f}, Time = {time_val:.2f}s")
        if mae < best_mae:
            best_mae = mae
            best_model_name = name

    if best_model_name:
        print(f"\nBest performing model (lowest MAE): {best_model_name} ({best_mae:.4f})")
    print("----------------------------------------------------------")


if __name__ == '__main__':
    main()
