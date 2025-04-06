import torch
import numpy as np
import argparse
from collections import OrderedDict

from models.gnn_model import GNNModelWithMLA, GNNModelBaseline
from load_data import load_node_classification_data

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_model_parameters(model, model_name="Model"):
    """Analyze parameter distribution across model components"""
    print(f"\n--- {model_name} Parameter Analysis ---")
    total_params = 0
    param_distribution = OrderedDict()
    
    # Iterate through named parameters to group them
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            
            # Get the component name (first part of parameter name before '.')
            component = name.split('.')[0] 
            if component not in param_distribution:
                param_distribution[component] = 0
            param_distribution[component] += param_count
    
    # Print parameter distribution
    print(f"Total trainable parameters: {total_params:,}")
    print("\nParameter distribution by component:")
    for component, params in param_distribution.items():
        percentage = params / total_params * 100
        print(f"  {component}: {params:,} parameters ({percentage:.2f}%)")
    
    # For MLA model, analyze sub-components if available
    if hasattr(model, 'mla_layer'):
        print("\nMLA Layer breakdown:")
        mla_params = 0
        mla_distribution = OrderedDict()
        
        for name, param in model.mla_layer.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                mla_params += param_count
                
                component = name.split('.')[0]
                if component not in mla_distribution:
                    mla_distribution[component] = 0
                mla_distribution[component] += param_count
        
        for component, params in mla_distribution.items():
            percentage = params / mla_params * 100
            print(f"  {component}: {params:,} parameters ({percentage:.2f}%)")
    
    print("-----------------------------------")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze model parameters.')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset to use')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden channels')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--kv_compress_dim', type=int, default=32, help='KV compression dimension')
    parser.add_argument('--q_compress_dim', type=int, default=32, help='Query compression dimension')
    parser.add_argument('--baseline_layers', type=int, default=4, help='Number of baseline layers')
    parser.add_argument('--use_pos_enc', action='store_true', default=False, help='Whether to use positional encoding')
    args = parser.parse_args()
    
    # Load dataset to get dimensions
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
        dropout=0.5,
        use_pos_enc=args.use_pos_enc
    )
    
    baseline_model = GNNModelBaseline(
        in_features=num_features,
        hidden_channels=args.hidden,
        out_features=num_classes,
        num_layers=args.baseline_layers,
        dropout=0.5
    )
    
    # Parameter comparison
    mla_params = count_parameters(mla_model)
    baseline_params = count_parameters(baseline_model)
    
    print("\n--- Model Parameter Comparison ---")
    print(f"MLA Model Parameters: {mla_params:,}")
    print(f"Baseline Model Parameters: {baseline_params:,}")
    print(f"Difference: {mla_params - baseline_params:,} parameters")
    print(f"Ratio: MLA model is {mla_params / baseline_params:.2f}x larger")
    print("-----------------------------------")
    
    # Detailed analysis
    analyze_model_parameters(mla_model, "MLA Model")
    analyze_model_parameters(baseline_model, "Baseline Model")
    
    if args.use_pos_enc:
        print("\nNote: Positional encoding is enabled, which adds significant parameters")
        print(f"(~{data.num_nodes * args.hidden:,} parameters for {data.num_nodes:,} nodes)")
    else:
        print("\nNote: Positional encoding is disabled to reduce parameter count")
    
if __name__ == "__main__":
    main()
