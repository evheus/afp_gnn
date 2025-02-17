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
        """Forward pass through the temporal GNN
        
        Args:
            x_sequence: Tensor of shape (batch, sequence_length, num_nodes, num_features)
            edge_index_sequence: List of edge_index tensors of shape (2, num_edges)
        """
        batch_size, sequence_length = x_sequence.size(0), x_sequence.size(1)
        gnn_outputs = []
        
        # Temporary debugging in TemporalGNN.forward
        for t in range(sequence_length):
            x_t = x_sequence[:, t]  # (batch, num_nodes, num_features)
            edge_index_t = edge_index_sequence[t]
            # Debug: Check edge_index_t structure
            print(f"Time step {t}: type {type(edge_index_t)}, shape {edge_index_t.shape if torch.is_tensor(edge_index_t) else 'Not a tensor'}")
            assert torch.is_tensor(edge_index_t), "edge_index_t is not a tensor!"
            assert edge_index_t.size(0) == 2, f"Expected shape[0]==2, got {edge_index_t.size(0)} at t={t}"
            x_t = self._apply_gnn(x_t, edge_index_t)
            gnn_outputs.append(x_t)
        
        # Stack along sequence dimension
        gnn_outputs = torch.stack(gnn_outputs, dim=1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(gnn_outputs)
        final_hidden = lstm_out[:, -1, :]
        
        # Predict
        out = self.predictor(final_hidden)
        return out

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
    def __init__(self, model, lr, weight_decay, train_mask, val_mask, test_mask):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

    def training_step(self, batch, batch_idx):
        # Access as dictionary instead of attributes
        x_seq = batch['x_seq']
        edge_index_seq = batch['edge_index_seq']
        y = batch['y']
        
        # Forward pass
        out = self.model(x_seq, edge_index_seq)
        
        # Calculate loss using the training mask
        loss = F.mse_loss(out[self.train_mask], y[self.train_mask])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Access dictionary items
        x_seq = batch['x_seq']
        edge_index_seq = batch['edge_index_seq']
        y = batch['y']
        
        # Forward pass
        out = self.model(x_seq, edge_index_seq)
        
        # Calculate validation loss using the validation mask
        val_loss = F.mse_loss(out[self.val_mask], y[self.val_mask])
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Access as dictionary instead of attributes
        x_seq = batch['x_seq']
        edge_index_seq = batch['edge_index_seq']
        y = batch['y']
        
        out = self.model(x_seq, edge_index_seq)

        test_loss = F.mse_loss(out[self.test_mask], y[self.test_mask])
        test_mae = F.l1_loss(out[self.test_mask], y[self.test_mask])
        self.log("test_loss", test_loss)
        self.log("test_mae", test_mae)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Only move the tensors that should be on the GPU.
        batch["x_seq"] = batch["x_seq"].to(device)
        batch["y"] = batch["y"].to(device)
        # Do NOT move edge_index_seq; leave it on CPU.
        # Note: edge_index_seq is a list of tensors.
        return batch