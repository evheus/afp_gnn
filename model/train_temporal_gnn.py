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
    parser.add_argument('--hidden_dim', type=int, default=64,
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
    # Stack node feature sequences and targets normally.
    collated["x_seq"] = torch.stack([item["x_seq"] for item in batch], dim=0)
    collated["y"] = torch.stack([item["y"] for item in batch], dim=0)
    
    # Each sample's edge_index_seq is a list of tensors (one per time step)
    sequence_length = len(batch[0]["edge_index_seq"])
    collated_edge_index_seq = []
    
    # For each time step, concatenate edge_index tensors along dim=1, then move them to CPU.
    for t in range(sequence_length):
        edge_indices_t = [sample["edge_index_seq"][t] for sample in batch]
        aggregated_edge_index = torch.cat(edge_indices_t, dim=1).cpu()  # <-- fix here
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
    adj_matrices = data['adjacency']    # (num_samples, sequence_length, num_nodes, num_nodes)
    next_step_features = data['labels'] # (num_samples, num_nodes, num_features)
    
    # Create masks for samples
    num_samples = len(node_features)
    indices = torch.randperm(num_samples)
    
    train_idx = indices[:int(0.7 * num_samples)]
    val_idx = indices[int(0.7 * num_samples):int(0.8 * num_samples)]
    test_idx = indices[int(0.8 * num_samples):]
    
    train_mask = torch.zeros(num_samples, dtype=torch.bool)
    val_mask = torch.zeros(num_samples, dtype=torch.bool)
    test_mask = torch.zeros(num_samples, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # Create dataset
    dataset = TemporalGraphDataset(node_features, adj_matrices, next_step_features)
    
    return dataset, train_mask, val_mask, test_mask

def main():
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
    
    # Prepare data
    dataset, train_mask, val_mask, test_mask = prepare_data(args)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate
    )
    
    # Create model
    model = get_temporal_model(args)
    
    # Create Lightning module
    lightning_model = TemporalGNNLightning(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
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
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpu >= 0 else 'cpu',
        devices=[args.gpu] if args.gpu >= 0 else None,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        deterministic=True
    )
    
    # Train model
    trainer.fit(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Test model
    trainer.test(
        lightning_model,
        dataloaders=test_loader
    )

if __name__ == "__main__":
    main()