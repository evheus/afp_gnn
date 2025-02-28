import argparse
import torch
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
import time

from temporal_gnn import TemporalGNN, TemporalGNNLightning, get_temporal_model

# def dense_to_sparse(adj_matrix):
#     """Convert dense adjacency matrix to edge indices.
    
#     Args:
#         adj_matrix: Dense adjacency matrix of shape (num_nodes, num_nodes)
        
#     Returns:
#         edge_index: Tuple of (row_indices, col_indices) each of shape (num_edges,)
#     """
#     indices = torch.nonzero(adj_matrix).t()
#     if indices.size(1) > 0:
#         row, col = indices[0], indices[1]
#         return row, col
#     return torch.zeros(2, 0, dtype=torch.long)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data parameters with default that will be overridden in main()
    parser.add_argument('--data_path', type=str,
                      default='data/processed/training_data.pt',
                      help='Path to the dataset')
    
    # Load metadata from saved tensors using the default path
    try:
        # First try with weights_only=False
        data = torch.load(parser.get_default('data_path'), weights_only=False)
    except Exception as e:
        print("Warning: Loading with weights_only=False failed. Adding safe globals and retrying...")
        # Add pandas timestamp unpickler to safe globals
        from pandas._libs.tslibs.timestamps import _unpickle_timestamp
        torch.serialization.add_safe_globals([_unpickle_timestamp])
        data = torch.load(parser.get_default('data_path'), weights_only=False)
    
    metadata = data['metadata']
    
    # Model parameters
    parser.add_argument('--num_features', type=int, 
                      default=metadata['num_features'],
                      help='Number of input features')
    parser.add_argument('--sequence_length', type=int,
                      default=metadata['sequence_length'],
                      help='Length of temporal sequences')
    parser.add_argument('--hidden_dim', type=int, default=12,
                      help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of GNN layers')
    parser.add_argument('--lstm_layers', type=int, default=1,
                      help='Number of LSTM layers')
    parser.add_argument('--conv_type', type=str, default='dir-gcn',
                      choices=['dir-gcn', 'dir-sage', 'dir-gat'],
                      help='Type of GNN convolution to use')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.5,
                      help='Alpha parameter for directional convolutions')
    parser.add_argument('--learn_alpha', action='store_true', default=False,
                      help='Whether to learn the alpha parameter')
    parser.add_argument('--bidirectional', default=False,
                      help='Whether to use bidirectional LSTM')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                      help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=10,
                      help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=20,
                      help='Patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU index to use')
    
    args = parser.parse_args()
    args.bidirectional = False  # Disable bidirectional LSTM for lookahead prevention
    return args

class TemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, node_features, edge_indices, edge_weights, next_step_features):
        self.node_features = node_features
        self.edge_indices = edge_indices
        self.edge_weights = edge_weights
        self.next_step_features = next_step_features

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        return {
            'x_seq': self.node_features[idx],  # Shape: (sequence_length, num_nodes, num_features)
            'edge_index_seq': [
                self.edge_indices[idx, t].long() for t in range(self.edge_indices.size(1))
            ],  # List of length sequence_length, each element shape: (2, num_edges)
            'edge_weight_seq': [
                self.edge_weights[idx, t] for t in range(self.edge_weights.size(1))
            ],  # List of length sequence_length, each element shape: (num_edges,)
            'y': self.next_step_features[idx]  # Shape: (num_nodes, num_features)
        }

def custom_collate(batch):
    """Maintain sparsity for edge data"""
    return {
        'x_seq': torch.stack([item['x_seq'] for item in batch]),
        # Keep edges as lists of sparse tensors
        'edge_index_seq': [b['edge_index_seq'] for b in batch],
        'edge_weight_seq': [b['edge_weight_seq'] for b in batch],
        'y': torch.stack([item['y'] for item in batch])
    }

def prepare_data(args):
    """Enhanced data preparation with chronological splitting"""
    try:
        data = torch.load(args.data_path, weights_only=False)
    except Exception as e:
        from pandas._libs.tslibs.timestamps import _unpickle_timestamp
        torch.serialization.add_safe_globals([_unpickle_timestamp])
        data = torch.load(args.data_path, weights_only=False)
    
    # Get all tensors
    node_features = data['features']
    edges_sparse = data['edges_sparse']
    edge_weights = data['edge_weights']
    next_step_features = data['labels']
    
    # Forward fill function for any tensor
    def forward_fill(tensor):
        if torch.isnan(tensor).any():
            mask = torch.isnan(tensor)
            tensor[mask] = 0.0  # Initialize NaNs to zero
            for i in range(1, len(tensor)):
                mask_i = mask[i]
                tensor[i][mask_i] = tensor[i-1][mask_i]
        return tensor
    
    # Handle NaNs in both features and labels
    if torch.isnan(node_features).any() or torch.isnan(next_step_features).any():
        print("\n=== Processing NaN Values ===")
        print(f"Features NaNs before: {torch.isnan(node_features).sum().item()}")
        print(f"Labels NaNs before: {torch.isnan(next_step_features).sum().item()}")
        
        # Forward fill both tensors
        node_features = forward_fill(node_features)
        next_step_features = forward_fill(next_step_features)
        
        print(f"Features NaNs after: {torch.isnan(node_features).sum().item()}")
        print(f"Labels NaNs after: {torch.isnan(next_step_features).sum().item()}")
    
    print(f"Loaded data with shapes: {node_features.shape}, {edges_sparse.shape}, {edge_weights.shape}, {next_step_features.shape}")
    
    # Modified splitting logic for chronological ordering
    num_samples = len(node_features)
    
    # Instead of random indices, use chronological order
    train_end = int(0.7 * num_samples)
    val_end = int(0.8 * num_samples)
    
    # Create chronological splits
    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, num_samples))
    
    print(f"\nChronological Split Info:")
    print(f"Training period: samples 0 to {train_end-1}")
    print(f"Validation period: samples {train_end} to {val_end-1}")
    print(f"Testing period: samples {val_end} to {num_samples-1}")
    
    dataset = TemporalGraphDataset(
        node_features=node_features,
        edge_indices=edges_sparse,
        edge_weights=edge_weights,
        next_step_features=next_step_features
    )
    
    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
        torch.utils.data.Subset(dataset, test_idx)
    )

# def compute_hit_rate(predictions, labels):
#     """
#     Calculate hit rate from stored predictions and true values.
#     Hit rate = proportion of correctly predicted directions
#     """
#     return (torch.sign(predictions) == torch.sign(labels)).float().mean().item()

# def calculate_trading_metrics(predictions, actual_returns, risk_free_rate=0.0):
#     """Calculate trading metrics with issue detection"""
#     # Check for data issues
#     has_issues = False
#     if torch.isnan(predictions).any() or torch.isnan(actual_returns).any():
#         has_issues = True
#         print("\n=== Warning: NaN Values Detected in Trading Metrics Input ===")
#         print(f"Predictions NaNs: {torch.isnan(predictions).sum().item()}")
#         print(f"Returns NaNs: {torch.isnan(actual_returns).sum().item()}")
    
#     # Convert tensors to float32 then numpy arrays
#     preds = predictions.float().cpu().numpy()
#     returns = actual_returns.float().cpu().numpy()

#     # Check for extreme values
#     if has_issues or np.any(np.abs(preds) > 10) or np.any(np.abs(returns) > 10):
#         print("\n=== Warning: Extreme Values Detected ===")
#         print(f"Predictions range: [{preds.min():.4f}, {preds.max():.4f}]")
#         print(f"Returns range: [{returns.min():.4f}, {returns.max():.4f}]")
    
#     # Generate trading signals (1 for long, -1 for short)
#     signals = np.sign(preds)
    
#     # Calculate strategy returns (signal * next period return)
#     strategy_returns = signals * returns
    
#     # Calculate cumulative returns
#     cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    
#     # Calculate annualized metrics (assuming daily data)
#     annual_factor = 252 * 390  # Trading days in a year
    
#     # Total return
#     total_return = cumulative_returns[-1]
    
#     # Annualized return
#     ann_return = (1 + total_return) ** (annual_factor / len(strategy_returns)) - 1
    
#     # Annualized volatility
#     ann_vol = strategy_returns.std() * np.sqrt(annual_factor)
    
#     # Sharpe ratio
#     excess_returns = strategy_returns - risk_free_rate / annual_factor
#     sharpe_ratio = np.sqrt(annual_factor) * excess_returns.mean() / strategy_returns.std()
    
#     # Maximum drawdown
#     cumulative = np.cumprod(1 + strategy_returns)
#     rolling_max = np.maximum.accumulate(cumulative)
#     drawdowns = cumulative / rolling_max - 1
#     max_drawdown = drawdowns.min()
    
#     # Plot cumulative returns
#     plt.figure(figsize=(12, 6))
#     plt.plot(cumulative_returns, label='Strategy Returns')
#     plt.title('Cumulative Strategy Returns')
#     plt.xlabel('Trading Days')
#     plt.ylabel('Cumulative Return')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('strategy_returns.png')
#     plt.close()
    
#     return {
#         'total_return': total_return,
#         'annualized_return': ann_return,
#         'annualized_volatility': ann_vol,
#         'sharpe_ratio': sharpe_ratio,
#         'max_drawdown': max_drawdown,
#         'strategy_returns': strategy_returns,
#         'cumulative_returns': cumulative_returns
#     }

def get_dataloaders(train_dataset, val_dataset, test_dataset, args):
    """Centralized dataloader configuration"""
    common_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': False,  # Prevent lookahead bias
        'persistent_workers': True,
        'collate_fn': custom_collate,
        'pin_memory': True,  # Enable for all loaders when using GPU
    }
    
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,  # Fewer workers for training due to frequent updates
        **common_kwargs
    )
    
    eval_loader_kwargs = {
        'num_workers': 12,  # More workers for validation/testing
        **common_kwargs
    }
    
    val_loader = DataLoader(val_dataset, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, **eval_loader_kwargs)
    
    return train_loader, val_loader, test_loader

from pytorch_lightning.callbacks import Callback
import time

class TimingCallback(Callback):
    def __init__(self):
        self.start_time = None
        self.last_epoch_time = None
    
    def on_sanity_check_end(self, trainer, pl_module):
        self.start_time = time.time()
        print(f"\nSanity check completed at: {time.strftime('%H:%M:%S')}")
        
    def on_train_epoch_end(self, trainer, pl_module):
        current_time = time.time()
        if self.last_epoch_time:
            epoch_duration = current_time - self.last_epoch_time
            total_duration = current_time - self.start_time
            print(f"\nEpoch {trainer.current_epoch} completed at: {time.strftime('%H:%M:%S')}")
            print(f"Epoch duration: {epoch_duration:.2f}s")
            print(f"Total training time: {total_duration:.2f}s")
        self.last_epoch_time = current_time

def main():
    torch.backends.cudnn.benchmark = True  # Enables CuDNN auto-tuner for faster runtime if input sizes are uniform

    # Set default data path
    default_data_path = 'data/processed/training_data.pt'
    
    # Parse arguments
    args = parse_args()
    
    # Override data_path with default
    args.data_path = default_data_path
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Print data path being used
    print(f"Using data from: {args.data_path}")
    
    # Load the processed data and extract metadata
    data = torch.load(args.data_path, weights_only=False)
    metadata = data['metadata']
    
    # Prepare data and split subsets
    train_dataset, val_dataset, test_dataset = prepare_data(args)
    
    # Create data loaders from the subsets
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, args)
        
    # Create model and ensure it's on CPU initially
    model = get_temporal_model(args)
    
    # Create the Lightning module with device handling
    target_tickers = None  
    # To restrict predictions, uncomment:
    # target_tickers = ['TSLA', 'AAPL']

    lightning_model = TemporalGNNLightning(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        target_tickers=target_tickers,  # if used
        # target_features=["returns", "volatility"]  # override if needed; default is "returns"
        metadata=metadata
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Changed from val_acc to val_loss
        dirpath='checkpoints/',
        filename='temporal-gnn-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min'  # Changed from max to min since we're monitoring loss
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Changed from val_acc to val_loss
        patience=args.patience,
        mode='min'  # Changed from max to min
    )
    
    # Set up logger
    logger = TensorBoardLogger('logs/', name='temporal_gnn')
    
    # Device configuration
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Modified trainer configuration
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=device,
        devices=1,
        callbacks=[checkpoint_callback, early_stopping, TimingCallback()],  # Add timing callback
        logger=logger,
        deterministic=True,
        precision="32"
    )

    trainer.fit(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    # Test the model and get predictions
    trainer.test(
        lightning_model,
        dataloaders=test_loader
    )

    # # Calculate hit rate using stored predictions
    # hit_rate = compute_hit_rate(
    #     lightning_model.test_predictions,
    #     lightning_model.test_labels
    # )
    # print(f"Out-of-sample hit rate: {hit_rate:.4f}")
    
    # Save predictions and actual returns
    predictions_path = 'results/predictions1.pt'
    os.makedirs('results', exist_ok=True)
    torch.save({
        'predictions': lightning_model.test_predictions.cpu(),
        'actual_returns': lightning_model.test_labels.cpu(),
        'metadata': {
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'model_args': vars(args),
            # 'hit_rate': hit_rate
        }
    }, predictions_path)
    print(f"\nSaved predictions and returns to: {predictions_path}")
    
    # # Calculate trading metrics
    # trading_metrics = calculate_trading_metrics(
    #     lightning_model.test_predictions,
    #     lightning_model.test_labels
    # )
    
    # # Print trading performance metrics
    # print("\nTrading Performance Metrics:")
    # print(f"Total Return: {trading_metrics['total_return']:.4f}")
    # print(f"Annualized Return: {trading_metrics['annualized_return']:.4f}")
    # print(f"Annualized Volatility: {trading_metrics['annualized_volatility']:.4f}")
    # print(f"Sharpe Ratio: {trading_metrics['sharpe_ratio']:.4f}")
    # print(f"Maximum Drawdown: {trading_metrics['max_drawdown']:.4f}")

if __name__ == "__main__":
    main()