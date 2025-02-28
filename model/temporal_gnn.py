import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import LSTM, Linear, ModuleList
import pytorch_lightning as pl
import traceback


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
        
        # Initialize device
        self.device = 'cpu'  # Default to CPU, will be updated during forward pass
        
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
            
    def _apply_gnn(self, x, edge_index, edge_weight):
        """Apply GNN layer while ensuring proper device management."""
        device = x.device
        # print(f"Input to GNN shape: {x.shape}")  # Add this debug print
        # halt = input("Press Enter to continue...")

        # Keep sparse tensors on CPU
        edge_index = edge_index.cpu()
        if edge_weight is not None:
            edge_weight = edge_weight.cpu()
        
        for layer in self.gnn_layers:
            # Dropout on current device
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Apply GNN layer (which internally handles device placement)
            x = layer(x, edge_index, edge_weight)

            # print(f"After GNN layer shape: {x.shape}")
            # halt = input("Enter to continue...")
            
            # Ensure output is on correct device
            x = x.to(device)
        
        return x  

    def forward(self, x_seq, edge_index_seq, edge_weight_seq):
        """Process a batch of temporal graph sequences through GNN and LSTM layers.
        
        Args:
            x_seq: Node features [batch_size, seq_len, num_nodes, feat_dim]
            edge_index_seq: List[batch_size][seq_len] of edge indices [2, num_edges]
            edge_weight_seq: List[batch_size][seq_len] of edge weights [num_edges]
        """
        # Extract dimensions
        batch_size, seq_len, num_nodes, feat_dim = x_seq.shape
        device = x_seq.device
        
        has_issues = False
        debug_info = []
        
        # Process each batch and timestep through GNN
        batch_outputs = []
        for b in range(batch_size):
            timestep_outputs = []
            for t in range(seq_len):
                # Get current timestep data
                x_t = x_seq[b, t]  # [num_nodes, feat_dim]
                edge_index_t = edge_index_seq[b][t]  # [2, num_edges]
                edge_weight_t = edge_weight_seq[b][t]  # [num_edges]
                
                # Check for numerical issues
                if torch.isnan(x_t).any() or torch.isnan(edge_weight_t).any():
                    has_issues = True
                    debug_info.append(f"NaN detected at batch {b}, timestep {t}")
                    debug_info.append(f"x_t NaNs: {torch.isnan(x_t).sum().item()}")
                    debug_info.append(f"edge_weights NaNs: {torch.isnan(edge_weight_t).sum().item()}")
                
                # Apply GNN (sparse ops on CPU)
                x_t = self._apply_gnn(x_t.unsqueeze(0), edge_index_t, edge_weight_t)
                # Check for NaNs after GNN
                if torch.isnan(x_t).any():
                    has_issues = True
                    debug_info.append(f"NaN detected after GNN at batch {b}, timestep {t}")

                timestep_outputs.append(x_t)
            
            # Stack timesteps for this batch [seq_len, num_nodes, hidden_dim]
            batch_out = torch.cat(timestep_outputs, dim=1)
            batch_outputs.append(batch_out)
        
        # Stack all batches [batch_size, seq_len, num_nodes, hidden_dim]
        x = torch.cat(batch_outputs, dim=0)
        
        # self._debug_print("After GNN Processing", {
        #     "Combined output": x.shape
        # })
        
        # Reshape for LSTM [batch_size * num_nodes, seq_len, hidden_dim]
        try:
            x = x.reshape(batch_size * num_nodes, seq_len, self.hidden_dim)
            # self._debug_print("LSTM Input", {"Reshaped": x.shape})
        except RuntimeError as e:
            self._debug_print("Reshape Error", {
                "Current shape": x.shape,
                "Current elements": x.numel(),
                "Target elements": batch_size * num_nodes * seq_len * self.hidden_dim
            })
            raise e
        
        # Process through LSTM
        x, _ = self.lstm(x)
        # self._debug_print("LSTM Output", {"Shape": x.shape})
        
        # Take final timestep and reshape [batch_size, num_nodes, hidden_dim]
        x = x[:, -1].reshape(batch_size, num_nodes, -1)
        # self._debug_print("Final LSTM", {"Shape": x.shape})
        
        # Final prediction
        x = self.predictor(x)
        # self._debug_print("Prediction", {"Shape": x.shape})
        # halt=input("Press Enter to continue...")
        return x.to(device)

    def _debug_print(self, title, shapes):
        """Helper method for consistent debug output formatting."""
        print(f"\n=== {title} ===")
        for name, shape in shapes.items():
            print(f"{name}: {shape}")
        # if self.training:  # Only pause for input during training
        #     input("Press Enter to continue...")

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
    def __init__(self, model, lr, weight_decay, target_tickers=None, target_features=None, metadata=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.target_tickers = target_tickers
        self.target_features = target_features or ["returns"]
        self.metadata = metadata
        
        # Initialize test storage as empty lists
        self.test_predictions = []
        self.test_labels = []
        
        # Pre-compute feature indices if possible
        if self.target_features and self.metadata:
            feature_to_index = self.metadata.get('feature_to_index', {})
            self.target_feature_indices = [
                idx for feat, idx in feature_to_index.items()
                if feat in self.target_features
            ]
            if not self.target_feature_indices:
                print(f"Warning: No valid features found. Available: {list(feature_to_index.keys())}")
    
    def on_test_start(self):
        """Reset test storage at start of testing"""
        self.test_predictions = []
        self.test_labels = []
    
    def test_step(self, batch, batch_idx):
        """Process one test batch and store results"""
        x_seq = batch['x_seq'].to(self.device)
        y = batch['y'].to(self.device)
        
        # Check input for issues
        if torch.isnan(x_seq).any() or torch.isnan(y).any():
            print(f"\n=== Warning: NaN detected in input (Batch {batch_idx}) ===")
            print(f"x_seq NaNs: {torch.isnan(x_seq).sum().item()}")
            print(f"y NaNs: {torch.isnan(y).sum().item()}")
        
        # Process edges
        edge_index_seq = [
            [edge_tensor.cpu() for edge_tensor in sample_seq] 
            for sample_seq in batch['edge_index_seq']
        ]
        edge_weight_seq = [
            [weight_tensor.cpu() for weight_tensor in sample_seq]
            for sample_seq in batch['edge_weight_seq']
        ]

        # Forward pass
        pred = self.model(x_seq, edge_index_seq, edge_weight_seq)
        
        # Check predictions for issues
        if torch.isnan(pred).any():
            print(f"\n=== Warning: NaN detected in predictions (Batch {batch_idx}) ===")
            print(f"Raw predictions NaNs: {torch.isnan(pred).sum().item()}")
            print(f"Value range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
        
        # Adjust predictions
        pred, y = self._adjust_predictions_and_labels(pred, y)
        
        # Check for post-adjustment issues
        if torch.isnan(pred).any():
            print(f"\n=== Warning: NaN detected after adjustment (Batch {batch_idx}) ===")
            print(f"Adjusted predictions NaNs: {torch.isnan(pred).sum().item()}")
        
        # Calculate metrics
        test_loss = F.mse_loss(pred, y)
        test_mae = F.l1_loss(pred, y)
        
        if torch.isnan(test_loss) or torch.isnan(test_mae):
            print(f"\n=== Warning: NaN detected in metrics (Batch {batch_idx}) ===")
            print(f"MSE: {test_loss.item()}")
            print(f"MAE: {test_mae.item()}")
        
        self.test_predictions.append(pred.detach().cpu())
        self.test_labels.append(y.detach().cpu())
        
        return test_loss

    def on_test_end(self):
        """Combine all test predictions"""
        if isinstance(self.test_predictions, list) and len(self.test_predictions) > 0:
            self.test_predictions = torch.cat(self.test_predictions)
            self.test_labels = torch.cat(self.test_labels)

    def forward(self, x_seq, edge_index_seq, edge_weight_seq):
        return self.model(x_seq, edge_index_seq, edge_weight_seq)

    def training_step(self, batch, batch_idx):
        # Batch supposed to contain:
        # - x_seq: [32, 5, 10, 3]         (batch, seq_len, nodes, features)
        # - edge_index_seq: list of 5 tensors, each [2, 1280]
        # - edge_weight_seq: list of 5 tensors, each [1280]
        # - y: [32, 10, 3]                (batch, nodes, features)

        # print(f"\n=== Training Step {batch_idx} ===")
        # for i in batch.keys() : print(f"{i} shape: {batch[i].shape}")
        # halt = input("Press Enter to continue...")  

        # Move dense tensors to GPU, keep sparse on CPU
        x_seq = batch['x_seq'].to(self.device)
        y = batch['y'].to(self.device)
        
        # Monitor input NaNs
        if torch.isnan(x_seq).any() or torch.isnan(y).any():
            print(f"\n=== Warning: NaN detected in training input (Batch {batch_idx}) ===")
            print(f"x_seq NaNs: {torch.isnan(x_seq).sum().item()}")
            print(f"y NaNs: {torch.isnan(y).sum().item()}")

        edge_index_seq = [
            [edge_tensor.cpu() for edge_tensor in sample_seq] 
            for sample_seq in batch['edge_index_seq']
        ]
        edge_weight_seq = [
            [weight_tensor.cpu() for weight_tensor in sample_seq]
            for sample_seq in batch['edge_weight_seq']
        ]
        
        pred = self(x_seq, edge_index_seq, edge_weight_seq)

        # Monitor prediction NaNs
        if torch.isnan(pred).any():
            print(f"\n=== Warning: NaN detected in training predictions (Batch {batch_idx}) ===")
            print(f"Predictions NaNs: {torch.isnan(pred).sum().item()}")

        loss = F.mse_loss(pred, y)

        # Monitor loss NaNs
        if torch.isnan(loss):
            print(f"\n=== Warning: NaN detected in training loss (Batch {batch_idx}) ===")
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # print(f"\n=== Validation Step {batch_idx} ===")
        # for key in batch.keys():
        #     if isinstance(batch[key], (list, tuple)):
        #         print(f"{key}: [List of {len(batch[key])} elements]")
        #         if isinstance(batch[key][0], (list, tuple)):
        #             # Handle nested lists
        #             print(f"First element: List of {len(batch[key][0])} elements")
        #             print(f"First inner element shape: {batch[key][0][0].shape}")
        #         else:
        #             # Handle list of tensors
        #             print(f"First element shape: {batch[key][0].shape}")
        #     else:
        #         print(f"{key} shape: {batch[key].shape}")
        # halt = input("Press Enter to continue...")

        x_seq = batch['x_seq'].to(self.device)

        # Handle nested lists for edge data
        # batch['edge_index_seq'] is [batch_size][seq_length][2, num_edges]
        edge_index_seq = [
            [edge_tensor.cpu() for edge_tensor in sample_seq] 
            for sample_seq in batch['edge_index_seq']
        ]
        edge_weight_seq = [
            [weight_tensor.cpu() for weight_tensor in sample_seq]
            for sample_seq in batch['edge_weight_seq']
        ]

        y = batch['y'].to(self.device)

        # # Debug shapes
        # print("\n=== Data Device Allocation ===")
        # print(f"x_seq device: {x_seq.device}")
        # print(f"First edge index device: {edge_index_seq[0][0].device}")
        # print(f"First edge weight device: {edge_weight_seq[0][0].device}")
        
        pred = self(x_seq, edge_index_seq, edge_weight_seq)
        loss = F.mse_loss(pred, y)

        self.log('val_loss', loss)
        return loss


    def on_test_epoch_end(self):
        # Concatenate all predictions and labels
        self.test_predictions = torch.cat(self.test_predictions, dim=0)
        self.test_labels = torch.cat(self.test_labels, dim=0)

    def _adjust_predictions_and_labels(self, pred, y):
        """Adjust predictions and labels by selecting subset of nodes and features."""
        import traceback
        
        try:
            # Check for NaNs in input
            has_nans = torch.isnan(pred).any() or torch.isnan(y).any()
            if has_nans:
                print("\n=== Input Validation ===")
                print(f"NaNs detected:")
                print(f"- pred NaNs: {torch.isnan(pred).sum().item()}")
                print(f"- y NaNs: {torch.isnan(y).sum().item()}")
                print(f"Shapes: pred {pred.shape}, y {y.shape}")
            
            # Ensure consistent device placement
            y = y.to(pred.device)
            
            # Handle target nodes and feature selection
            if hasattr(self, 'target_nodes') and self.target_nodes is not None:
                feature_to_index = self.metadata.get('feature_to_index', {})
                feature_indices = [
                    idx for feat, idx in feature_to_index.items()
                    if feat in self.target_features
                ]
                
                if not feature_indices:
                    print(f"\n=== Feature Selection Warning ===")
                    print(f"No valid features found.")
                    print(f"Target features: {self.target_features}")
                    print(f"Available features: {list(feature_to_index.keys())}")
                    return pred, y
                    
                feature_idx_tensor = torch.tensor(feature_indices, device=pred.device)
                pred = pred.index_select(2, feature_idx_tensor)
                y = y.index_select(2, feature_idx_tensor)
                
                # Check for NaNs after selection
                if torch.isnan(pred).any() or torch.isnan(y).any():
                    print("\n=== Post-Selection Warning ===")
                    print(f"NaNs after feature selection:")
                    print(f"- pred NaNs: {torch.isnan(pred).sum().item()}")
                    print(f"- y NaNs: {torch.isnan(y).sum().item()}")
            
            return pred, y
            
        except Exception as e:
            print("\n=== Error Traceback ===")
            print(traceback.format_exc())
            print("State at error:")
            print(f"- pred: shape {pred.shape}, device {pred.device}, NaNs {torch.isnan(pred).sum().item()}")
            print(f"- y: shape {y.shape}, device {y.device}, NaNs {torch.isnan(y).sum().item()}")
            raise

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer