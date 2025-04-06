import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
# Duplicates removed below
# import argparse
# import time
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import scheduler
# import numpy as np
# import matplotlib.pyplot as plt

from load_data import load_node_classification_data
# Import all models
from models.gnn_model import GNNModelWithMLA, GNNModelBaseline, GATModelBaseline, GNNModelOnlyMLA
from utils.utils import accuracy

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Updated train_model to handle scheduler and clipping flags
def train_model(model, data, optimizer, epochs=200, patience=10, model_name="Model", device='cpu',
                use_clipping=True, clip_value=1.0, use_scheduler=False):
    """Train a model and return training history and test accuracy"""
    model.to(device)
    data = data.to(device)
    
    # For early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Initialize scheduler if requested
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True) # Example patience for scheduler

    # Lists to store metrics history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    t_total = time.time()
    
    print(f"\n--- Starting Training: {model_name} ---")
    
    for epoch in range(epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(data)
        
        # Training metrics
        loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        acc_train = accuracy(output[data.train_mask], data.y[data.train_mask])
        loss_train.backward()
        
        # Conditionally apply gradient clipping
        if use_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        # Validation metrics
        model.eval()
        with torch.no_grad():
            output = model(data)
            loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
            acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
        
        # Step the scheduler if used
        if scheduler:
            scheduler.step(loss_val)

        # Store metrics
        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
        train_accs.append(acc_train.item())
        val_accs.append(acc_val.item())
        
        # Print progress
        print(f'Epoch: {epoch+1:04d}',
              f'loss_train: {loss_train.item():.4f}',
              f'acc_train: {acc_train.item():.4f}',
              f'loss_val: {loss_val.item():.4f}',
              f'acc_val: {acc_val.item():.4f}',
              f'time: {time.time() - t:.4f}s')
        
        # Early stopping check with nan/inf handling
        if not torch.isfinite(loss_val):
            print(f"Warning: Non-finite validation loss detected. Early stopping.")
            if best_model_state is not None:
                # Restore the previous best model if available
                model.load_state_dict(best_model_state)
                print("Restored previous best model.")
            else:
                print("No previous best model to restore. Training failed.")
                # Return partial results if training fails early
                return {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accs': train_accs,
                    'val_accs': val_accs,
                    'test_loss': float('nan'),
                    'test_acc': 0.0,
                    'epochs_trained': len(train_losses),
                    'early_failure': True
                }
            break
            
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve == patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    print(f"Training time: {time.time() - t_total:.4f}s")
    
    # Load best model for testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test metrics
    model.eval()
    with torch.no_grad():
        output = model(data)
        loss_test = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
        acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    
    print(f"\n--- {model_name} Test Results ---")
    print(f"Test Loss: {loss_test.item():.4f}")
    print(f"Test Accuracy: {acc_test.item():.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_loss': loss_test.item(),
        'test_acc': acc_test.item(),
        'epochs_trained': len(train_losses)
    }

def plot_comparison(results_dict, dataset_name, save_path=None, param_counts=None):
    """Plot comparison of performance metrics for multiple models"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    # Define colors and labels for all potential models
    colors = {'mla': 'b', 'gcn_baseline': 'r', 'gat_baseline': 'g', 'mla_only': 'm'} # Added mla_only
    labels = {'mla': 'MLA (GCN Base)', 'gcn_baseline': 'GCN Baseline', 'gat_baseline': 'GAT Baseline', 'mla_only': 'MLA Only'} # Added mla_only and updated mla label

    for model_type, results in results_dict.items():
        if results is None: continue # Skip if results are missing
        epochs = range(1, len(results['train_losses']) + 1)
        color = colors.get(model_type, 'k') # Default to black if type unknown
        
        # If parameter counts are provided, include them in the label
        if param_counts and model_type in param_counts:
            label = f"{labels.get(model_type, model_type)} ({param_counts[model_type]:,} params)"
        else:
            label = labels.get(model_type, model_type)

        # Training loss
        ax1.plot(epochs, results['train_losses'], color=color, linestyle='-', label=label)
        # Validation loss
        ax2.plot(epochs, results['val_losses'], color=color, linestyle='-', label=label)
        # Training accuracy
        ax3.plot(epochs, results['train_accs'], color=color, linestyle='-', label=label)
        # Validation accuracy
        ax4.plot(epochs, results['val_accs'], color=color, linestyle='-', label=label)

    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True)

    ax4.set_title('Validation Accuracy')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    
    # Update the title to indicate parameter equalization
    if param_counts:
        plt.suptitle(f'Performance Comparison on {dataset_name} Dataset (Equalized Parameters)', fontsize=16)
    else:
        plt.suptitle(f'Performance Comparison on {dataset_name} Dataset', fontsize=16)
        
    plt.subplots_adjust(top=0.92) # Adjust title position

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare MLA and Baseline GNN models with similar parameter counts')
    parser.add_argument('--dataset', type=str, default='Flickr', 
                      help='Dataset to use (Cora, CiteSeer, PubMed, Flickr, Reddit, ogbn-arxiv).')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension') # Changed from 8 to 64
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs') # Changed from 50 to 200
    parser.add_argument('--patience', type=int, default=200, help='Early stopping patience') # Changed from 50 to 10
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate') # Changed from 0.6 to 0.5
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).') # Changed from 0 to 5e-4
    
    # Model specific arguments
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads for MLA') # Kept at 4 (matches train.py)
    parser.add_argument('--kv_compress_dim', type=int, default=64, help='KV compression dimension for MLA') # Changed from 16 to 32
    parser.add_argument('--q_compress_dim', type=int, default=64, help='Q compression dimension for MLA') # Changed from 16 to 32
    parser.add_argument('--gcn_layers', type=int, default=2, help='Number of layers for GCN baseline') # Renamed from baseline_layers, default 2
    parser.add_argument('--gat_layers', type=int, default=2, help='Number of layers for GAT baseline')
    parser.add_argument('--gat_heads', type=int, default=2, help='Number of attention heads for GAT baseline')
    parser.add_argument('--mla_only_layers', type=int, default=2, help='Number of layers for MLA Only model') # Added for MLA Only
    parser.add_argument('--use_pos_enc', action='store_true', default=False, help='Use positional encoding in MLA models')
    parser.add_argument('--skip_train', action='store_true', default=False, help='Skip training and only plot existing results')
    # New arguments
    parser.add_argument('--no-grad-clip', action='store_true', default=True, help='Disable gradient clipping during training.')
    parser.add_argument('--use-lr-scheduler', action='store_true', default=False, help='Enable ReduceLROnPlateau learning rate scheduler.')
    all_model_choices = ['mla',
                          #'gcn_baseline',
                            #'gat_baseline',
                             # 'mla_only'
                              ]
    parser.add_argument('--models-to-compare', nargs='+', default=all_model_choices, choices=all_model_choices,
                        help=f'Which models to include in the comparison (default: all). Choose from: {all_model_choices}')

    args = parser.parse_args()
    
    # Determine clipping setting
    use_clipping = not args.no_grad_clip

    # CUDA setup
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Load data
    data, num_features, num_classes = load_node_classification_data(args.dataset)

    # --- Model Instantiation ---
    models = {}
    if 'mla' in args.models_to_compare:
        models['mla'] = GNNModelWithMLA(
            in_features=num_features,
            hidden_channels=args.hidden,
            out_features=num_classes,
            num_nodes=data.num_nodes,
            num_heads=args.num_heads,
            kv_compression_dim=args.kv_compress_dim,
            q_compression_dim=args.q_compress_dim,
            num_base_layers=1, # Assuming 1 base layer for MLA model in comparison
            dropout=args.dropout,
            use_pos_enc=args.use_pos_enc
        )
    if 'gcn_baseline' in args.models_to_compare:
        models['gcn_baseline'] = GNNModelBaseline(
            in_features=num_features,
            hidden_channels=args.hidden,
            out_features=num_classes,
            num_layers=args.gcn_layers,
            dropout=args.dropout
        )
    if 'gat_baseline' in args.models_to_compare:
        models['gat_baseline'] = GATModelBaseline(
            in_features=num_features,
            hidden_channels=args.hidden,
            out_features=num_classes,
            num_layers=args.gat_layers,
            heads=args.gat_heads,
            dropout=args.dropout
        )
    if 'mla_only' in args.models_to_compare:
        models['mla_only'] = GNNModelOnlyMLA(
            in_features=num_features,
            hidden_channels=args.hidden,
            out_features=num_classes,
            num_nodes=data.num_nodes,
            num_heads=args.num_heads,
            kv_compression_dim=args.kv_compress_dim,
            q_compression_dim=args.q_compress_dim,
            num_mla_layers=args.mla_only_layers,
            dropout=args.dropout,
            use_pos_enc=args.use_pos_enc
        )

    if not models:
        print("No models selected for comparison. Exiting.")
        return

    # --- Parameter Count ---
    param_counts_dict = {name: count_parameters(model) for name, model in models.items()}
    print("\n--- Model Parameter Comparison ---")
    # Define labels for printing parameter counts
    model_labels_print = {'mla': 'MLA (GCN Base)', 'gcn_baseline': 'GCN Baseline', 'gat_baseline': 'GAT Baseline', 'mla_only': 'MLA Only'}
    for name, count in param_counts_dict.items():
         print(f"{model_labels_print.get(name, name)} Model Parameters: {count:,}")
    print("-----------------------------------")

    # --- Optimizers ---
    optimizers = {name: optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                  for name, model in models.items()}

    # --- Training ---
    all_results = {name: None for name in args.models_to_compare} # Initialize results dict for selected models

    if not args.skip_train:
        print(f"\n--- Starting Training Loop ---")
        print(f"Gradient Clipping: {'Enabled' if use_clipping else 'Disabled'}")
        print(f"LR Scheduler: {'Enabled' if args.use_lr_scheduler else 'Disabled'}")
        for model_name in args.models_to_compare:
            print(f"\nTraining model: {model_labels_print.get(model_name, model_name)}")
            model_to_train = models[model_name]
            optimizer_to_use = optimizers[model_name]
            all_results[model_name] = train_model(
                model=model_to_train,
                data=data,
                optimizer=optimizer_to_use,
                epochs=args.epochs,
                patience=args.patience,  # Ensure patience is correctly passed from command line args
                model_name=model_labels_print.get(model_name, model_name),
                device=device,
                use_clipping=use_clipping, # Pass clipping flag
                use_scheduler=args.use_lr_scheduler # Pass scheduler flag
            )
    else:
        print("\n--- Skipping Training - Loading Existing Results ---")
        # Load results if training is skipped
        # --- Removed misplaced model instantiation code from here ---

        results_file = f"{args.dataset}_model_comparison_results.npz"
        if os.path.exists(results_file):
            try:
                loaded_data = np.load(results_file, allow_pickle=True)
                # Extract results for each model type if present
                # Load results only for the models specified in models_to_compare
                loaded_data = np.load(results_file, allow_pickle=True)
                loaded_successfully = False
                for model_name in args.models_to_compare:
                    if model_name in loaded_data:
                        all_results[model_name] = loaded_data[model_name].item()
                        loaded_successfully = True
                    elif model_name == 'gcn_baseline' and 'baseline' in loaded_data: # Handle old format for gcn_baseline
                        all_results['gcn_baseline'] = loaded_data['baseline'].item()
                        loaded_successfully = True

                if loaded_successfully:
                    print(f"Loaded results for selected models from {results_file}")
                else:
                    print(f"No results found for selected models in {results_file}")

            except Exception as e:
                print(f"Error loading results from {results_file}: {e}")
                print("Proceeding without loaded results.")
        else:
            print(f"Results file not found: {results_file}. Cannot load results.")

    # Filter out models that weren't trained or loaded
    valid_results = {k: v for k, v in all_results.items() if v is not None}

    if not valid_results:
        print("No valid results found to compare or plot. Exiting.")
        return # Exit if no results are available

    # Define labels here for use in print statements and plotting (ensure it covers all possible models)
    labels = {'mla': 'MLA (GCN Base)', 'gcn_baseline': 'GCN Baseline', 'gat_baseline': 'GAT Baseline', 'mla_only': 'MLA Only'}

    # Compare test performance
    print("\n--- Performance Comparison (Selected Models) ---")
    best_model = None
    best_acc = -1
    for model_type, results in valid_results.items():
        acc = results['test_acc']
        print(f"{labels.get(model_type, model_type)} Test Accuracy: {acc:.4f}") # Now labels is defined
        if acc > best_acc:
            best_acc = acc
            best_model = model_type
            
    if best_model:
        print(f"\nBest performing model: {labels.get(best_model, best_model)} ({best_acc:.4f})")
    print("-----------------------------------")
    
    # Plot comparison using only the valid results for selected models
    save_path = f"{args.dataset}_model_comparison_{'_'.join(sorted(valid_results.keys()))}.png" # Make filename specific
    plot_comparison(valid_results, args.dataset, save_path, param_counts=param_counts_dict)

    # Save detailed results (only if training was performed)
    if not args.skip_train:
        # Save only the results for the models that were actually trained/compared
        save_data = {
            **valid_results,
            'parameters': {k: v for k, v in param_counts_dict.items() if k in valid_results}, # Save params only for valid models
            'config': vars(args)
        }
        
        results_file = f"{args.dataset}_model_comparison_results.npz"
        np.savez(results_file, **save_data)
        print(f"Results saved to {results_file}") # Moved print inside if

if __name__ == "__main__":
    main()
