import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import LSTM, Linear, ModuleList
import pytorch_lightning as pl

# Import the directional convolution layers
from .directed_gnn_layers import DirGCNConv, DirSAGEConv, DirGATConv

class TemporalGNN(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim,
        num_classes,
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
        
        # Final prediction layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.predictor = Linear(lstm_output_dim, num_classes)
        
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
        """Apply all GNN layers to a single time step"""
        for i, conv in enumerate(self.gnn_layers):
            x = conv(x, edge_index)
            if i != len(self.gnn_layers) - 1:  # Don't apply activation to final layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def forward(self, x_sequence, edge_index_sequence):
        """
        Forward pass through the temporal GNN
        
        Args:
            x_sequence: Tensor of shape (sequence_length, num_nodes, num_features)
            edge_index_sequence: List of edge_index tensors for each timestep
        """
        seq_length = x_sequence.size(0)
        
        # Process each time step with GNN layers
        gnn_outputs = []
        for t in range(seq_length):
            x_t = x_sequence[t]
            edge_index_t = edge_index_sequence[t]
            
            # Apply GNN layers directly with edge_index
            x_t = self._apply_gnn(x_t, edge_index_t)
            gnn_outputs.append(x_t)
            
        # Combine all timesteps
        gnn_outputs = torch.stack(gnn_outputs, dim=1)  # (batch_size, seq_length, hidden_dim)
        
        # Apply LSTM to the sequence of graph embeddings
        lstm_out, _ = self.lstm(gnn_outputs)  # (batch_size, seq_length, hidden_dim)
        
        # Use the final time step's output for prediction
        final_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Make prediction
        out = self.predictor(final_hidden)  # (batch_size, num_classes)
        
        return F.log_softmax(out, dim=1)

def get_temporal_model(args):
    return TemporalGNN(
        num_features=args.num_features,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        num_gnn_layers=args.num_layers,
        lstm_layers=args.lstm_layers,
        conv_type=args.conv_type,
        dropout=args.dropout,
        alpha=args.alpha,
        learn_alpha=args.learn_alpha,
        bidirectional=args.bidirectional if hasattr(args, 'bidirectional') else False
    )

class TemporalGNNLightning(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, train_mask, val_mask, test_mask, evaluator=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.evaluator = evaluator
        self.train_mask, self.val_mask, self.test_mask = train_mask, val_mask, test_mask

    def training_step(self, batch, batch_idx):
        x_seq, edge_index_seq = batch.x_seq, batch.edge_index_seq
        y = batch.y.long()
        out = self.model(x_seq, edge_index_seq)

        loss = nn.functional.nll_loss(out[self.train_mask], y[self.train_mask].squeeze())
        self.log("train_loss", loss)

        y_pred = out.max(1)[1]
        train_acc = self.evaluate(y_pred=y_pred[self.train_mask], y_true=y[self.train_mask])
        self.log("train_acc", train_acc)
        val_acc = self.evaluate(y_pred=y_pred[self.val_mask], y_true=y[self.val_mask])
        self.log("val_acc", val_acc)

        return loss

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred.unsqueeze(1)})["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]
        return acc

    def test_step(self, batch, batch_idx):
        x_seq, edge_index_seq = batch.x_seq, batch.edge_index_seq
        y = batch.y.long()
        out = self.model(x_seq, edge_index_seq)

        y_pred = out.max(1)[1]
        test_acc = self.evaluate(y_pred=y_pred[self.test_mask], y_true=y[self.test_mask])
        self.log("test_acc", test_acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer