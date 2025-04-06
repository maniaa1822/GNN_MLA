import torch
from torch_geometric.datasets import Planetoid, Reddit, QM9 # Import QM9
import torch_geometric.transforms as T
import os.path as osp

def load_data(dataset_name, root=None):
    """
    Load a node classification or graph-level dataset.

    Args:
        dataset_name (str): Name of dataset ('Cora', 'CiteSeer', 'PubMed', 'Reddit', 'Flickr', 'QM9', etc)
        root (str, optional): Root directory where the dataset should be saved.
                             Defaults to './data'.
    
    Returns:
        data: A PyG Data object containing the graph
        num_features: Number of input node features (for node classification) or None (for graph datasets)
        num_classes: Number of output classes (for node classification) or None (for graph datasets)
        dataset: The raw dataset object (especially for graph datasets like QM9)
    """
    if root is None:
        root = osp.abspath(osp.join(osp.dirname(__file__), '..', 'data_cache')) # Use a consistent cache directory

    
    # PyTorch Geometric standard datasets (Cora, CiteSeer, PubMed)
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        # Just normalize features but don't convert to sparse tensor 
        # since that's causing compatibility issues
        transform = T.NormalizeFeatures()
        
        try:
            dataset = Planetoid(root=root, name=dataset_name, transform=transform)
            data = dataset[0]
            
            print(f"Loaded {dataset_name} dataset:")
            print(f"  Nodes: {data.num_nodes}")
            print(f"  Edges: {data.num_edges}")
            print(f"  Features: {dataset.num_features}")
            print(f"  Classes: {dataset.num_classes}")
            print(f"  Train nodes: {data.train_mask.sum().item()}")
            print(f"  Val nodes: {data.val_mask.sum().item()}")
            print(f"  Test nodes: {data.test_mask.sum().item()}")
            
            return data, dataset.num_features, dataset.num_classes
            
        except Exception as e:
            print(f"Error loading dataset '{dataset_name}': {e}")
            return None, None, None
    
    # Reddit dataset
    elif dataset_name == 'Reddit':
        try:
            transform = T.NormalizeFeatures()
            
            print(f"Loading {dataset_name} dataset. This may take a while...")
            dataset = Reddit(root=osp.join(root, 'Reddit'), transform=transform)
            data = dataset[0]
            
            print(f"Loaded {dataset_name} dataset:")
            print(f"  Nodes: {data.num_nodes}")
            print(f"  Edges: {data.num_edges}")
            print(f"  Features: {dataset.num_features}")
            print(f"  Classes: {dataset.num_classes}")
            print(f"  Train nodes: {data.train_mask.sum().item()}")
            print(f"  Val nodes: {data.val_mask.sum().item()}")
            print(f"  Test nodes: {data.test_mask.sum().item()}")
            
            return data, dataset.num_features, dataset.num_classes
            
        except Exception as e:
            print(f"Error loading Reddit dataset: {e}")
            print("Note: The Reddit dataset requires ~11GB of memory and may take a while to download.")
            return None, None, None
            
    # Flickr dataset
    elif dataset_name == 'Flickr':
        try:
            from torch_geometric.datasets import Flickr
            transform = T.NormalizeFeatures()
            
            print(f"Loading {dataset_name} dataset. This may take a while...")
            dataset = Flickr(root=osp.join(root, 'Flickr'), transform=transform)
            data = dataset[0]
            
            print(f"Loaded {dataset_name} dataset:")
            print(f"  Nodes: {data.num_nodes}")
            print(f"  Edges: {data.num_edges}")
            print(f"  Features: {data.num_features}")
            print(f"  Classes: {dataset.num_classes}")
            print(f"  Train nodes: {data.train_mask.sum().item()}")
            print(f"  Val nodes: {data.val_mask.sum().item()}")
            print(f"  Test nodes: {data.test_mask.sum().item()}")
            
            return data, data.num_features, dataset.num_classes
            
        except Exception as e:
            print(f"Error loading Flickr dataset: {e}")
            return None, None, None
            
    # OGB Datasets support
    elif dataset_name.startswith('ogbn-'):
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
            
            # Add required transformations
            transform = T.NormalizeFeatures()
            
            # Try to use default loading - might fail with PyTorch 2.6+
            try:
                # Load OGB dataset
                dataset = PygNodePropPredDataset(name=dataset_name, root=osp.join(root, 'OGB'), transform=transform)
                data = dataset[0]
                
                # Create train/val/test masks from OGB split
                split_idx = dataset.get_idx_split()
                
                # OGB datasets don't have built-in mask attributes like Planetoid
                # We need to create them
                num_nodes = data.num_nodes
                
                data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                
                data.train_mask[split_idx['train']] = True
                data.val_mask[split_idx['valid']] = True
                data.test_mask[split_idx['test']] = True
                
                # OGB datasets store target differently
                data.y = data.y.squeeze(1)  # Convert [num_nodes, 1] to [num_nodes]
                
                print(f"Loaded {dataset_name} dataset:")
                print(f"  Nodes: {data.num_nodes}")
                print(f"  Edges: {data.num_edges}")
                print(f"  Features: {data.x.size(1)}")
                print(f"  Classes: {len(torch.unique(data.y))}")
                print(f"  Train nodes: {data.train_mask.sum().item()}")
                print(f"  Val nodes: {data.val_mask.sum().item()}")
                print(f"  Test nodes: {data.test_mask.sum().item()}")
                
                return data, data.x.size(1), len(torch.unique(data.y))
            
            except Exception as e:
                print(f"Error with default OGB loading: {e}")
                print("You may need to downgrade PyTorch to use OGB datasets or update OGB")
                return None, None, None
            
        except ImportError:
            print("To use OGB datasets, you need to install the OGB package:")
            print("pip install ogb")
            return None, None, None
        except Exception as e:
            print(f"Error loading OGB dataset '{dataset_name}': {e}")
            return None, None, None, None

    # QM9 dataset (Graph Regression)
    elif dataset_name == 'QM9':
        try:
            print(f"Loading {dataset_name} dataset. This may take a while...")
            # QM9 requires specific preprocessing/transforms, but PyG handles basic loading.
            # We might need target-specific transforms later in the training script.
            path = osp.join(root, 'QM9')
            dataset = QM9(path)

            print(f"Loaded {dataset_name} dataset:")
            print(f"  Graphs: {len(dataset)}")
            # QM9 node features are categorical (atom type), edge features exist too.
            print(f"  Node Features Dim: {dataset.num_node_features}")
            print(f"  Edge Features Dim: {dataset.num_edge_features}")
            print(f"  Targets: {dataset.num_classes}") # num_classes holds the number of regression targets

            # For graph datasets, return the dataset object itself.
            # num_features and num_classes are less relevant here in the same way as node classification.
            return dataset, dataset.num_node_features, dataset.num_classes, dataset # Return dataset object

        except Exception as e:
            print(f"Error loading QM9 dataset: {e}")
            print("Ensure you have torch_geometric installed correctly.")
            return None, None, None, None

    else:
        print(f"Dataset '{dataset_name}' is not supported. Choose from: Cora, CiteSeer, PubMed, Reddit, Flickr, ogbn-arxiv, QM9")
        return None, None, None, None
