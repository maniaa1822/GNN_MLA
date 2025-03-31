import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_loss_curves(dataset_name):
    """Loads loss histories and plots comparison curves."""
    mla_history_file = f'loss_history_{dataset_name}_mla.npz'
    baseline_history_file = f'loss_history_{dataset_name}_baseline.npz'
    output_plot_file = f'{dataset_name}_loss_comparison.png'

    try:
        mla_data = np.load(mla_history_file)
        baseline_data = np.load(baseline_history_file)
    except FileNotFoundError as e:
        print(f"Error loading loss history files: {e}")
        print("Please ensure you have run the training for both 'mla' and 'baseline' models first.")
        return

    mla_train_losses = mla_data['train_losses']
    mla_val_losses = mla_data['val_losses']
    baseline_train_losses = baseline_data['train_losses']
    baseline_val_losses = baseline_data['val_losses']

    epochs = range(1, len(mla_train_losses) + 1)

    plt.figure(figsize=(12, 6))

    # Plotting Training Losses
    plt.plot(epochs, mla_train_losses, 'b-', label='MLA Train Loss')
    plt.plot(epochs, baseline_train_losses, 'r-', label='Baseline Train Loss')

    # Plotting Validation Losses
    plt.plot(epochs, mla_val_losses, 'b--', label='MLA Val Loss')
    plt.plot(epochs, baseline_val_losses, 'r--', label='Baseline Val Loss')

    plt.title(f'Training and Validation Loss Comparison on {dataset_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (NLL)')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0) # Ensure y-axis starts at 0 for loss

    # Save the plot
    plt.savefig(output_plot_file)
    print(f"Plot saved to {output_plot_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot loss curves for model comparison.')
    parser.add_argument('--dataset', type=str, default='PubMed', help='Dataset name used for loss history files (e.g., PubMed, Cora).')
    args = parser.parse_args()

    plot_loss_curves(args.dataset)
