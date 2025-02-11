import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from temporal_gnn import TemporalGNN, TemporalGNNLightning, get_temporal_model

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument('--num_features', type=int, required=True,
                      help='Number of input features')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Hidden dimension size')
    parser.add_argument('--num_classes', type=int, required=True,
                      help='Number of output classes')
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
    parser.add_argument('--learn_alpha', action='store_true',
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
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the dataset')
    parser.add_argument('--sequence_length', type=int, required=True,
                      help='Length of temporal sequences')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU index to use')
    
    return parser.parse_args()

class TemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, node_embeddings, adj_matrices, labels=None):
        """
        Dataset for temporal graph data.
        
        Args:
            node_embeddings: List of tensors, each of shape (sequence_length, num_nodes, num_features)
            adj_matrices: List of tensors, each of shape (sequence_length, num_nodes, num_nodes)
            labels: Tensor of shape (num_nodes,) containing node labels
        """
        self.node_embeddings = node_embeddings
        self.adj_matrices = adj_matrices
        self.labels = labels
        
    def __len__(self):
        return len(self.node_embeddings)
        
    def __getitem__(self, idx):
        item = {
            'x_seq': self.node_embeddings[idx],
            'adj_seq': self.adj_matrices[idx],
        }
        if self.labels is not None:
            item['y'] = self.labels[idx]
        return item

def prepare_data(args):
    """
    Prepare the temporal graph data and masks.
    """
    # Load node embeddings and adjacency matrices
    # Assuming data is stored as a dictionary with 'embeddings', 'adjacency', and 'labels' keys
    data = torch.load(args.data_path)
    
    node_embeddings = data['embeddings']  # (num_samples, sequence_length, num_nodes, num_features)
    adj_matrices = data['adjacency']      # (num_samples, sequence_length, num_nodes, num_nodes)
    labels = data['labels']               # (num_nodes,)
    
    # Create masks for nodes
    num_nodes = labels.size(0)
    indices = torch.randperm(num_nodes)
    
    train_idx = indices[:int(0.7 * num_nodes)]
    val_idx = indices[int(0.7 * num_nodes):int(0.8 * num_nodes)]
    test_idx = indices[int(0.8 * num_nodes):]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # Create dataset
    dataset = TemporalGraphDataset(node_embeddings, adj_matrices, labels)
    
    return dataset, train_mask, val_mask, test_mask

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Prepare data
    data, train_mask, val_mask, test_mask = prepare_data(args)
    
    # Create data loaders
    # Note: Implement your DataLoader according to your data format
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
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
        monitor='val_acc',
        dirpath='checkpoints/',
        filename='temporal-gnn-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_acc',
        patience=args.patience,
        mode='max'
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