import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import LSTM, Linear, ModuleList
import pytorch_lightning as pl

from directed_gnn_layers import DirGCNConv, DirSAGEConv, DirGATConv

class TemporalGNN(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim,
        num_gnn_layers=2,
        lstm_layers=1,
        conv_type="dir-gcn",
        dropout=0.1,
        alpha=0.5,
        learn_alpha=False,
        bidirectional=False
    ):
        super(TemporalGNN, self).__init__()
        
        self.num_gnn_layers = num_gnn_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Parameter for directional convolutions
        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)
        
        # Initialize GNN layers
        self.gnn_layers = ModuleList()
        
        # First GNN layer (input -> hidden)
        self.gnn_layers.append(self._get_conv_layer(conv_type, num_features, hidden_dim))
        
        # Additional GNN layers (hidden -> hidden)
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(self._get_conv_layer(conv_type, hidden_dim, hidden_dim))
            
        # LSTM layer for temporal patterns
        self.lstm = LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Final prediction layer - predicts next timestep's features
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.predictor = Linear(lstm_output_dim, num_features)  # Predict same number of features as input
        
    def _get_conv_layer(self, conv_type, in_dim, out_dim):
        if conv_type == "dir-gcn":
            return DirGCNConv(in_dim, out_dim, self.alpha)
        elif conv_type == "dir-sage":
            return DirSAGEConv(in_dim, out_dim, self.alpha)
        elif conv_type == "dir-gat":
            return DirGATConv(in_dim, out_dim, heads=1, alpha=self.alpha)
        else:
            raise ValueError(f"Unsupported convolution type: {conv_type}")
            
    def _apply_gnn(self, x, edge_index):
        # Save the original device (e.g., mps)
        orig_device = x.device

        # Move x to CPU for the sparse operations.
        x_cpu = x.cpu()
        # Temporarily move all GNN layers to CPU.
        self.gnn_layers.to("cpu")

        for conv in self.gnn_layers:
            x_cpu = conv(x_cpu, edge_index)
            x_cpu = F.relu(x_cpu)
            x_cpu = F.dropout(x_cpu, p=self.dropout, training=self.training)

        # Move the output back to the original device.
        out = x_cpu.to(orig_device)

        # (Optional) Move the GNN layers back to orig_device for subsequent operations.
        self.gnn_layers.to(orig_device)

        return out
    
    def forward(self, x_sequence, edge_index_sequence):
        """
        Forward pass through the temporal GNN.
        
        Args:
            x_sequence: Tensor of shape (batch, sequence_length, num_nodes, num_features)
            edge_index_sequence: List of edge_index tensors for each time step,
                                 where each tensor is of shape (2, num_edges)
        Returns:
            predictions: Tensor of shape (batch, num_nodes, output_dim)
                         with per-node predictions (e.g. future prices).
        """
        batch_size, sequence_length, num_nodes, _ = x_sequence.size()
        gnn_outputs = []
        
        # Process each time step individually.
        for t in range(sequence_length):
            x_t = x_sequence[:, t]  # Shape: (batch, num_nodes, num_features)
            edge_index_t = edge_index_sequence[t]
            # Apply the GNN layers on the snapshot for time step t.
            # _apply_gnn should output a tensor of shape (batch, num_nodes, hidden_dim)
            x_t = self._apply_gnn(x_t, edge_index_t)
            gnn_outputs.append(x_t)
        
        # Stack over time to build the temporal dimension.
        # gnn_outputs now has shape: (batch, sequence_length, num_nodes, hidden_dim)
        gnn_outputs = torch.stack(gnn_outputs, dim=1)
        
        # Reshape the tensor so that each node's sequence is processed individually by the LSTM.
        B, T, N, H = gnn_outputs.size()
        node_sequences = gnn_outputs.view(B * N, T, H)
        
        # Process each node's temporal sequence using the LSTM.
        lstm_out, _ = self.lstm(node_sequences)
        # For prediction, we use the final hidden state from the LSTM for each node.
        final_hidden = lstm_out[:, -1, :]  # Shape: (B * N, H')
        
        # Pass through a predictor layer (e.g., a Linear layer) to output the predicted price.
        predictions = self.predictor(final_hidden)  # Expected shape: (B * N, output_dim)
        
        # Reshape predictions back to (batch, num_nodes, output_dim)
        predictions = predictions.view(B, N, -1)
        return predictions

def get_temporal_model(args):
    return TemporalGNN(
        num_features=args.num_features,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_layers,
        lstm_layers=args.lstm_layers,
        conv_type=args.conv_type,
        dropout=args.dropout,
        alpha=args.alpha,
        learn_alpha=args.learn_alpha,
        bidirectional=args.bidirectional if hasattr(args, 'bidirectional') else False
    )

class TemporalGNNLightning(pl.LightningModule):
    def __init__(
        self,
        model,
        lr,
        weight_decay,
        target_tickers=None,
        target_features="returns",   # Default now predicts only 'returns'
        metadata=None,
    ):
        """
        Args:
            model: An instance of TemporalGNN.
            lr: Learning rate.
            weight_decay: Weight decay.
            target_tickers: Optional list of tickers to restrict predictions.
            target_features: Feature name or list of feature names to select for prediction.
                            Default is "returns".
            metadata: Dictionary that includes ticker_to_index and feature_to_index.
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.metadata = metadata or {}
        
        # Set up ticker restriction if provided.
        if target_tickers is not None:
            if 'ticker_to_index' not in self.metadata:
                raise ValueError("metadata must include ticker_to_index if using target_tickers.")
            ticker_to_index = self.metadata['ticker_to_index']
            self.target_nodes = [ticker_to_index[ticker] for ticker in target_tickers if ticker in ticker_to_index]
            if len(self.target_nodes) == 0:
                raise ValueError("None of the provided target tickers were found.")
        else:
            self.target_nodes = None

        # Set up feature selection.
        if target_features is not None:
            if 'feature_to_index' not in self.metadata:
                raise ValueError("metadata must include feature_to_index if using target_features.")
            # Allow target_features as string or list.
            if isinstance(target_features, str):
                self.target_feature_indices = [self.metadata['feature_to_index'][target_features]]
            else:
                self.target_feature_indices = [self.metadata['feature_to_index'][feat] for feat in target_features]
        else:
            # Default to predicting all features (should not happen with our new default)
            self.target_feature_indices = None

    def forward(self, x_seq, edge_index_seq):
        # Simply pass through to your underlying model
        return self.model(x_seq, edge_index_seq)

    def on_before_batch_transfer(self, batch, device):
        # Ensure x_seq and y are moved to the training device,
        # but leave edge_index_seq on CPU (for sparse operations).
        batch["x_seq"] = batch["x_seq"].to(device)
        batch["y"] = batch["y"].to(device)
        return batch

    def training_step(self, batch, batch_idx):
        x_seq = batch['x_seq']                    # (batch, sequence_length, num_nodes, num_features)
        edge_index_seq = batch['edge_index_seq']    # remains on CPU (or as originally collated)
        y = batch['y']                            # (batch, label_node_count, output_dim)

        pred = self.model(x_seq, edge_index_seq)    # (batch, full_num_nodes, output_dim)
        pred, y = self._adjust_predictions_and_labels(pred, y)
        loss = F.mse_loss(pred, y)
        self.log('train_loss', loss, batch_size=x_seq.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x_seq = batch['x_seq']
        edge_index_seq = batch['edge_index_seq']
        y = batch['y']

        pred = self.model(x_seq, edge_index_seq)
        pred, y = self._adjust_predictions_and_labels(pred, y)
        val_loss = F.mse_loss(pred, y)
        self.log('val_loss', val_loss, batch_size=x_seq.size(0))
        return val_loss

    def test_step(self, batch, batch_idx):
        x_seq = batch['x_seq']
        edge_index_seq = batch['edge_index_seq']
        y = batch['y']

        pred = self.model(x_seq, edge_index_seq)
        pred, y = self._adjust_predictions_and_labels(pred, y)
        test_loss = F.mse_loss(pred, y)
        test_mae = F.l1_loss(pred, y)
        self.log("test_loss", test_loss, batch_size=x_seq.size(0))
        self.log("test_mae", test_mae, batch_size=x_seq.size(0))

    def _adjust_predictions_and_labels(self, pred, y):
        """
        Adjust predictions and labels by optionally selecting a subset of nodes and features.
        """
        # First, ensure y is on the same device as pred.
        y = y.to(pred.device)

        # Restrict to target nodes if specified.
        if self.target_nodes is not None:
            target_nodes = torch.tensor(self.target_nodes, device=pred.device)
            pred = pred.index_select(1, target_nodes)
            y = y.index_select(1, target_nodes)
        elif pred.size(1) != y.size(1):
            # self.print(f"Adjusting prediction tensor from shape {pred.shape} to match labels shape {y.shape}")
            pred = pred[:, :y.size(1), :]

        # Now, restrict to target feature(s) if specified.
        if self.target_feature_indices is not None:
            feature_idx_tensor = torch.tensor(self.target_feature_indices, device=pred.device)
            # Assume the feature dimension is the last one.
            pred = pred.index_select(2, feature_idx_tensor)
            y = y.index_select(2, feature_idx_tensor)
            
        return pred, y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer