import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from temporal_gnn import TemporalGNN, TemporalGNNLightning, get_temporal_model

def dense_to_sparse(adj_matrix):
    """Convert dense adjacency matrix to edge indices.
    
    Args:
        adj_matrix: Dense adjacency matrix of shape (num_nodes, num_nodes)
        
    Returns:
        edge_index: Tuple of (row_indices, col_indices) each of shape (num_edges,)
    """
    indices = torch.nonzero(adj_matrix).t()
    if indices.size(1) > 0:
        row, col = indices[0], indices[1]
        return row, col
    return torch.zeros(2, 0, dtype=torch.long)

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
    parser.add_argument('--hidden_dim', type=int, default=32,
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
    parser.add_argument('--bidirectional', action='store_true',
                      help='Whether to use bidirectional LSTM')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                      help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=200,
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
    return args

class TemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, node_features, adj_matrices, next_step_features):
        self.node_features = node_features
        # Convert dense adjacency matrices to sparse format and stack correctly
        self.edge_indices = []
        for sample in adj_matrices:
            sample_edges = []
            for adj in sample:
                row, col = dense_to_sparse(adj)
                # Create edge_index tensor of shape (2, num_edges)
                edge_index = torch.stack([row, col])  # correctly shapes to (2, num_edges)
                sample_edges.append(edge_index)
            self.edge_indices.append(sample_edges)
        self.next_step_features = next_step_features

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        return {
            'x_seq': self.node_features[idx],
            # Return the already correctly formatted list of edge_index tensors
            'edge_index_seq': self.edge_indices[idx],
            'y': self.next_step_features[idx]
        }

def custom_collate(batch):
    collated = {}
    collated["x_seq"] = torch.stack([item["x_seq"] for item in batch], dim=0)
    collated["y"] = torch.stack([item["y"] for item in batch], dim=0)
    
    sequence_length = len(batch[0]["edge_index_seq"])
    collated_edge_index_seq = []
    for t in range(sequence_length):
        edge_indices_t = [sample["edge_index_seq"][t] for sample in batch]
        aggregated_edge_index = torch.cat(edge_indices_t, dim=1).cpu()  # Force indices to CPU.
        collated_edge_index_seq.append(aggregated_edge_index)
    
    collated["edge_index_seq"] = collated_edge_index_seq
    return collated

def prepare_data(args):
    """
    Prepare the temporal graph data and masks.
    """
    # Load node features and adjacency matrices with appropriate settings
    try:
        data = torch.load(args.data_path, weights_only=False)
    except Exception as e:
        from pandas._libs.tslibs.timestamps import _unpickle_timestamp
        torch.serialization.add_safe_globals([_unpickle_timestamp])
        data = torch.load(args.data_path, weights_only=False)
    
    node_features = data['features']    # (num_samples, sequence_length, num_nodes, num_features)
    adj_matrices = data['adjacency']      # (num_samples, sequence_length, num_nodes, num_nodes)
    next_step_features = data['labels']   # (num_samples, num_nodes, num_features)

    # Create indices for splits
    num_samples = len(node_features)
    indices = torch.randperm(num_samples)

    train_idx = indices[:int(0.7 * num_samples)].tolist()
    val_idx = indices[int(0.7 * num_samples):int(0.8 * num_samples)].tolist()
    test_idx = indices[int(0.8 * num_samples):].tolist()

    # Create full dataset
    dataset = TemporalGraphDataset(node_features, adj_matrices, next_step_features)

    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    return train_dataset, val_dataset, test_dataset

def compute_hit_rate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            # Ensure batch is transferred properly
            batch = model.on_before_batch_transfer(batch, model.device) if hasattr(model, 'device') else batch
            x_seq = batch['x_seq']
            edge_index_seq = batch['edge_index_seq']
            y = batch['y']
            
            pred = model(x_seq, edge_index_seq)
            # Adjust predictions and labels using the same helper as during training.
            pred, y = model._adjust_predictions_and_labels(pred, y)
            all_preds.append(pred)
            all_labels.append(y)
        
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute hit rate: proportion where sign(pred) equals sign(true)
    hit_rate = (torch.sign(all_preds) == torch.sign(all_labels)).float().mean().item()
    return hit_rate

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
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=12,                 # Increase based on your CPU cores
        pin_memory=True,               # Only effective for GPU training
        persistent_workers=True,
        collate_fn=custom_collate
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate
    )
        
    # Create model
    model = get_temporal_model(args)
    
    # Create the Lightning module.
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
    
    # Determine accelerator and devices based on availability of CUDA.
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = [args.gpu] if args.gpu >= 0 else None
    else:
        accelerator = 'cpu'
        devices = 1  # Use 1 CPU core

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        deterministic=True,
        log_every_n_steps=30,  # Lower logging frequency to reduce overhead
        precision=16           # Enable mixed precision training (for CUDA only)
    )

    trainer.fit(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    trainer.test(
        lightning_model,
        dataloaders=test_loader
    )

    # After training and testing, evaluate hit rate.
    hit_rate = compute_hit_rate(lightning_model, test_loader)
    print(f"Out-of-sample hit rate: {hit_rate:.4f}")

if __name__ == "__main__":
    main()