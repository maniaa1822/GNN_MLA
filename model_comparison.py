import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from data.load_data import load_node_classification_data
from models.gnn_model import GNNModelWithMLA, GNNModelBaseline
from utils.utils import accuracy

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, data, optimizer, epochs=200, patience=10, model_name="Model", device='cpu'):
    """Train a model and return training history and test accuracy"""
    model.to(device)
    data = data.to(device)
    
    # For early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
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
        optimizer.step()
        
        # Validation metrics
        model.eval()
        with torch.no_grad():
            output = model(data)
            loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
            acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
        
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
        
        # Early stopping check
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

def plot_comparison(mla_results, baseline_results, dataset_name, save_path=None):
    """Plot comparison of performance metrics for both models"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    epochs_mla = range(1, len(mla_results['train_losses']) + 1)
    epochs_baseline = range(1, len(baseline_results['train_losses']) + 1)
    
    ax1.plot(epochs_mla, mla_results['train_losses'], 'b-', label='MLA')
    ax1.plot(epochs_baseline, baseline_results['train_losses'], 'r-', label='Baseline')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Validation loss
    ax2.plot(epochs_mla, mla_results['val_losses'], 'b-', label='MLA')
    ax2.plot(epochs_baseline, baseline_results['val_losses'], 'r-', label='Baseline')
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Training accuracy
    ax3.plot(epochs_mla, mla_results['train_accs'], 'b-', label='MLA')
    ax3.plot(epochs_baseline, baseline_results['train_accs'], 'r-', label='Baseline')
    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True)
    
    # Validation accuracy
    ax4.plot(epochs_mla, mla_results['val_accs'], 'b-', label='MLA')
    ax4.plot(epochs_baseline, baseline_results['val_accs'], 'r-', label='Baseline')
    ax4.set_title('Validation Accuracy')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.suptitle(f'Performance Comparison on {dataset_name} Dataset (Equal Parameters)', fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare MLA and Baseline GNN models with similar parameter counts')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer', 'PubMed'], help='Dataset to use')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA')
    
    # Model specific arguments
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads for MLA')
    parser.add_argument('--kv_compress_dim', type=int, default=32, help='KV compression dimension')
    parser.add_argument('--q_compress_dim', type=int, default=32, help='Q compression dimension')
    parser.add_argument('--baseline_layers', type=int, default=4, help='Number of layers for baseline')
    parser.add_argument('--use_pos_enc', action='store_true', default=False, help='Use positional encoding in MLA')
    
    args = parser.parse_args()
    
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
    
    # Create models
    mla_model = GNNModelWithMLA(
        in_features=num_features,
        hidden_channels=args.hidden,
        out_features=num_classes,
        num_nodes=data.num_nodes,
        num_heads=args.num_heads,
        kv_compression_dim=args.kv_compress_dim,
        q_compression_dim=args.q_compress_dim,
        num_base_layers=1,
        dropout=args.dropout,
        use_pos_enc=args.use_pos_enc
    )
    
    baseline_model = GNNModelBaseline(
        in_features=num_features,
        hidden_channels=args.hidden,
        out_features=num_classes,
        num_layers=args.baseline_layers,
        dropout=args.dropout
    )
    
    # Parameter count
    mla_params = count_parameters(mla_model)
    baseline_params = count_parameters(baseline_model)
    
    print("\n--- Model Parameter Comparison ---")
    print(f"MLA Model Parameters: {mla_params:,}")
    print(f"Baseline Model Parameters: {baseline_params:,}")
    print(f"Difference: {mla_params - baseline_params:,} parameters")
    print(f"Ratio: MLA model is {mla_params / baseline_params:.2f}x larger")
    print("-----------------------------------")
    
    # Optimizers
    mla_optimizer = optim.Adam(mla_model.parameters(), lr=args.lr, weight_decay=5e-4)
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=args.lr, weight_decay=5e-4)
    
    # Train both models
    mla_results = train_model(
        mla_model, data, mla_optimizer, 
        epochs=args.epochs, patience=args.patience,
        model_name="MLA Model", device=device
    )
    
    baseline_results = train_model(
        baseline_model, data, baseline_optimizer, 
        epochs=args.epochs, patience=args.patience,
        model_name="Baseline Model", device=device
    )
    
    # Compare test performance
    print("\n--- Performance Comparison ---")
    print(f"MLA Test Accuracy: {mla_results['test_acc']:.4f}")
    print(f"Baseline Test Accuracy: {baseline_results['test_acc']:.4f}")
    print(f"Difference: {mla_results['test_acc'] - baseline_results['test_acc']:.4f}%")
    
    if mla_results['test_acc'] > baseline_results['test_acc']:
        print("MLA model outperforms Baseline model")
    else:
        print("Baseline model outperforms MLA model")
    print("-----------------------------------")
    
    # Plot comparison
    save_path = f"{args.dataset}_model_comparison.png"
    plot_comparison(mla_results, baseline_results, args.dataset, save_path)
    
    # Save detailed results
    results = {
        'mla': mla_results,
        'baseline': baseline_results,
        'parameters': {
            'mla': mla_params,
            'baseline': baseline_params,
            'ratio': mla_params / baseline_params
        },
        'config': vars(args)
    }
    
    np.savez(f"{args.dataset}_model_comparison_results.npz", **results)
    print(f"Results saved to {args.dataset}_model_comparison_results.npz")

if __name__ == "__main__":
    main()
