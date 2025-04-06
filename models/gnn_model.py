import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_add_pool, global_mean_pool # Import pooling
from torch.nn import Embedding, Linear, ModuleList, Sequential, ReLU, Dropout # More imports
from .mla_layer import MultiHeadLatentAttentionLayer

class GNNModelWithMLA(nn.Module):
    def __init__(self, in_features, hidden_channels, out_features, num_nodes, num_heads, 
                 kv_compression_dim, q_compression_dim, num_base_layers=1, dropout=0.0, 
                 use_pos_enc=True): # Added use_pos_enc parameter
        super().__init__()
        self.dropout = dropout
        self.num_base_layers = num_base_layers
        self.hidden_channels = hidden_channels
        self.use_pos_enc = use_pos_enc  # Store flag

        # Base GNN layers
        self.base_layers = nn.ModuleList()
        current_dim = in_features
        for _ in range(num_base_layers):
            self.base_layers.append(GCNConv(current_dim, hidden_channels))
            current_dim = hidden_channels

        # Learnable Positional Encoding - only create if enabled
        if self.use_pos_enc:
            self.pos_encoder = nn.Embedding(num_nodes, hidden_channels)
            nn.init.xavier_uniform_(self.pos_encoder.weight)

        # Multi-Head Latent Attention Layer
        self.mla_layer = MultiHeadLatentAttentionLayer(
            in_channels=hidden_channels, 
            out_channels=hidden_channels,
            num_heads=num_heads,
            kv_compression_dim=kv_compression_dim,
            q_compression_dim=q_compression_dim,
            dropout=dropout
        )
        
        # Layer normalization for residual connection
        self.norm = nn.LayerNorm(hidden_channels)

        # Output layer for node classification
        self.output_layer = nn.Linear(hidden_channels, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply base GNN layers
        for i in range(self.num_base_layers):
            x = self.base_layers[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Add Positional Encoding if enabled
        if self.use_pos_enc:
            node_indices = torch.arange(data.num_nodes, device=x.device)
            pos_encoding = self.pos_encoder(node_indices)
            x = x + pos_encoding # Add PE to node features

        # Store the input to MLA for residual connection
        residual = x
        
        # Apply MLA layer
        x = self.mla_layer(x, edge_index)
        
        # Add residual connection and normalize
        x = self.norm(x + residual)
        
        # Apply dropout after residual connection
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply output layer
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)


class GNNModelOnlyMLA(nn.Module):
    """A GNN model using only stacked Multi-Head Latent Attention layers."""
    def __init__(self, in_features, hidden_channels, out_features, num_nodes, num_heads,
                 kv_compression_dim, q_compression_dim, num_mla_layers=2,
                 dropout=0.0, use_pos_enc=False):
        super().__init__()
        self.in_features = in_features
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.num_mla_layers = num_mla_layers
        self.use_pos_enc = use_pos_enc

        # Optional Positional Encoding
        if self.use_pos_enc:
            # PE dimension should match the input to the first layer
            self.pos_encoder = nn.Embedding(num_nodes, in_features)
            nn.init.xavier_uniform_(self.pos_encoder.weight)

        self.mla_layers = nn.ModuleList()
        self.norms = nn.ModuleList() # Norms for residual connections

        # Projection for the first residual connection if in_features != hidden_channels
        self.res_proj = None
        if in_features != hidden_channels:
            self.res_proj = nn.Linear(in_features, hidden_channels, bias=False)
            nn.init.xavier_uniform_(self.res_proj.weight)


        current_dim = in_features
        for i in range(num_mla_layers):
            self.mla_layers.append(MultiHeadLatentAttentionLayer(
                in_channels=current_dim,
                out_channels=hidden_channels,
                num_heads=num_heads,
                kv_compression_dim=kv_compression_dim,
                q_compression_dim=q_compression_dim,
                dropout=dropout
            ))
            self.norms.append(nn.LayerNorm(hidden_channels))
            current_dim = hidden_channels # Input for next layer is output of current

        # Output layer
        self.output_layer = nn.Linear(hidden_channels, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Add Positional Encoding if enabled
        if self.use_pos_enc:
             node_indices = torch.arange(data.num_nodes, device=x.device)
             pos_encoding = self.pos_encoder(node_indices)
             x = x + pos_encoding # Add PE to initial features

        # Apply MLA layers with residuals
        for i in range(self.num_mla_layers):
            residual = x
            # Apply projection to residual only for the first layer if dimensions mismatch
            if i == 0 and self.res_proj is not None:
                 residual = self.res_proj(residual)
            # For subsequent layers, dimensions match hidden_channels
            elif i > 0 and self.in_features != self.hidden_channels:
                 # Ensure residual has hidden_channels dim for layers > 0
                 # This should already be the case as x is output of previous layer
                 pass


            x = self.mla_layers[i](x, edge_index)

            # Add residual connection and normalize
            x = self.norms[i](x + residual)

            # Apply activation and dropout (except maybe after the last MLA layer?)
            # Let's apply it after each layer for now.
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)


        # Apply output layer
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)


class GATModelBaseline(nn.Module):
    """A baseline GAT model without the MLA layer."""
    def __init__(self, in_features, hidden_channels, out_features, num_layers=2, heads=8, dropout=0.6, concat=False):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads
        self.concat = concat

        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GATConv(in_features, hidden_channels, heads=heads, dropout=dropout, concat=concat))
        # Output dimension depends on concat flag
        current_dim = hidden_channels * heads if concat else hidden_channels

        # Hidden layers (if num_layers > 2)
        for _ in range(1, num_layers - 1):
            self.layers.append(GATConv(current_dim, hidden_channels, heads=heads, dropout=dropout, concat=concat))
            # Update current_dim based on concat flag
            current_dim = hidden_channels * heads if concat else hidden_channels

        # Output GAT layer
        if num_layers > 1:
            self.layers.append(GATConv(current_dim, hidden_channels, heads=heads, dropout=dropout, concat=False)) 
            final_gat_out_dim = hidden_channels
        else:
            # For single layer case
            self.layers[0] = GATConv(in_features, hidden_channels, heads=heads, dropout=dropout, concat=False)
            final_gat_out_dim = hidden_channels

        # Output linear layer
        self.output_layer = nn.Linear(final_gat_out_dim, out_features)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply GAT layers
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            # Apply activation and dropout after each layer except the last GAT layer
            if i < self.num_layers - 1:
                x = F.elu(x) # GAT typically uses ELU
                x = F.dropout(x, p=self.dropout, training=self.training)
            # For the last GAT layer, activation (like ELU) is often applied *before* the final linear layer
            elif i == self.num_layers - 1:
                 x = F.elu(x) # Apply activation after the last GAT layer as well

        # Apply output layer
        # No dropout typically applied right before the final classification layer
        x = self.output_layer(x)

        return F.log_softmax(x, dim=1)


class GNNModelBaseline(nn.Module):
    """A baseline GNN model without the MLA layer."""
    def __init__(self, in_features, hidden_channels, out_features, num_layers=2, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        current_dim = in_features
        for i in range(num_layers):
            # Use hidden_channels for all layers except potentially the last GNN layer
            # if we want the output layer to take hidden_channels as input.
            # Here, all GCN layers output hidden_channels.
            self.layers.append(GCNConv(current_dim, hidden_channels))
            current_dim = hidden_channels # Input dim for the next layer is the output dim of the current one

        # Output layer
        self.output_layer = nn.Linear(hidden_channels, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply GNN layers
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            x = F.relu(x)
            # Apply dropout after activation, except maybe for the last layer before output
            if i < self.num_layers - 1: # Optional: Don't dropout before final linear layer
                 x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply output layer
        x = self.output_layer(x)

        return F.log_softmax(x, dim=1)


# === QM9 Graph Regression Models ===

class QM9GraphRegGAT(nn.Module):
    """A GAT model adapted for QM9 graph regression."""
    # Reduced defaults for smaller size
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_channels=32, out_features=19,
                 num_layers=2, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads
        self.hidden_channels = hidden_channels

        # Node Embedding for atom types (QM9 has 11 features per node, typically categorical)
        # Assuming node_feature_dim is the embedding dimension needed, not the raw feature count.
        # Let's use a simple linear layer for initial projection if node_feature_dim is not embedding size.
        # Or better, assume QM9 loader gives categorical features that need embedding.
        # QM9 node features: x [num_nodes, 11] (atom type, charge, etc.) - needs embedding
        self.node_emb = Embedding(100, hidden_channels) # Embedding for atom types (adjust size if needed)

        self.layers = nn.ModuleList()
        current_dim = hidden_channels # After embedding

        # GAT layers
        for i in range(num_layers):
            # Output heads are averaged, not concatenated, except maybe the first layer?
            # Let's average for simplicity in regression.
            concat = False # Average heads for regression tasks
            # Input dim is hidden_channels, output dim is hidden_channels
            self.layers.append(GATConv(current_dim, hidden_channels, heads=heads, dropout=dropout, concat=concat))
            current_dim = hidden_channels # Output dim after averaging

        # Graph pooling layer
        self.pool = global_add_pool # Or global_mean_pool

        # Output linear layer for 19 targets
        self.output_layer = Linear(hidden_channels, out_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Node features in QM9: x[:, 0] is atom type. Embed only that.
        x_emb = self.node_emb(x[:, 0].long())
        # We might want to concatenate other node features if available/useful,
        # but for now, just use the embedding.
        x = x_emb # Use embedded features

        # Apply GAT layers
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            x = F.relu(x) # Use ReLU or ELU
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply graph pooling
        x = self.pool(x, batch)

        # Apply output layer (no activation for regression)
        x = self.output_layer(x)
        return x


class QM9GraphRegMLA(nn.Module):
    """An MLA model adapted for QM9 graph regression."""
    # Reduced defaults for smaller size
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_channels=32, out_features=19,
                 num_heads=2, kv_compression_dim=16, q_compression_dim=16,
                 num_base_layers=1, num_mla_layers=1, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.num_base_layers = num_base_layers
        self.num_mla_layers = num_mla_layers
        self.hidden_channels = hidden_channels

        # Node Embedding
        self.node_emb = Embedding(100, hidden_channels) # Embedding for atom types

        # Base GNN layers (optional, e.g., GCN)
        self.base_layers = nn.ModuleList()
        current_dim = hidden_channels # After embedding
        for _ in range(num_base_layers):
            self.base_layers.append(GCNConv(current_dim, hidden_channels))
            current_dim = hidden_channels

        # MLA layers
        self.mla_layers = nn.ModuleList()
        self.norms = nn.ModuleList() # Norms for residual connections
        for _ in range(num_mla_layers):
            self.mla_layers.append(MultiHeadLatentAttentionLayer(
                in_channels=current_dim,
                out_channels=hidden_channels,
                num_heads=num_heads,
                kv_compression_dim=kv_compression_dim,
                q_compression_dim=q_compression_dim,
                dropout=dropout
            ))
            self.norms.append(nn.LayerNorm(hidden_channels))
            current_dim = hidden_channels

        # Graph pooling layer
        self.pool = global_add_pool # Or global_mean_pool

        # Output linear layer for 19 targets
        self.output_layer = Linear(hidden_channels, out_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node Embedding - Use only the first feature (atom type)
        x_emb = self.node_emb(x[:, 0].long())
        x = x_emb # Use embedded features

        # Apply base GNN layers
        for i in range(self.num_base_layers):
            x = self.base_layers[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply MLA layers with residuals
        for i in range(self.num_mla_layers):
            residual = x
            x = self.mla_layers[i](x, edge_index)
            x = self.norms[i](x + residual) # Add residual connection and normalize
            x = F.relu(x) # Activation after norm
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply graph pooling
        x = self.pool(x, batch)

        # Apply output layer
        x = self.output_layer(x)
        return x


class QM9GraphRegMLA_NoBases(nn.Module):
    """An MLA model without base GCN layers for QM9 graph regression."""
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_channels=32, out_features=19,
                 num_heads=2, kv_compression_dim=16, q_compression_dim=16,
                 num_mla_layers=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.num_mla_layers = num_mla_layers
        self.hidden_channels = hidden_channels

        # Node Embedding
        self.node_emb = Embedding(100, hidden_channels)  # Embedding for atom types

        # Use MLA layers directly without base GCN layers
        self.mla_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # For the first layer, input is from node embedding
        current_dim = hidden_channels
        
        for _ in range(num_mla_layers):
            self.mla_layers.append(MultiHeadLatentAttentionLayer(
                in_channels=current_dim,
                out_channels=hidden_channels,
                num_heads=num_heads,
                kv_compression_dim=kv_compression_dim,
                q_compression_dim=q_compression_dim,
                dropout=dropout
            ))
            self.norms.append(nn.LayerNorm(hidden_channels))
            current_dim = hidden_channels

        # Graph pooling layer
        self.pool = global_add_pool

        # Output linear layer for targets
        self.output_layer = Linear(hidden_channels, out_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node Embedding - Use only the first feature (atom type)
        x = self.node_emb(x[:, 0].long())
        
        # Apply MLA layers with residuals
        for i in range(self.num_mla_layers):
            residual = x
            x = self.mla_layers[i](x, edge_index)
            x = self.norms[i](x + residual)  # Add residual and normalize
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply graph pooling
        x = self.pool(x, batch)

        # Apply output layer
        x = self.output_layer(x)
        return x
