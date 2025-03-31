import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv # Example base GNN layer
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

        # Apply MLA layer
        x = self.mla_layer(x, edge_index)

        # Apply output layer
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
