import torch
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from data.construct_tensors.construct_lead_lag_matrices import generate_leadlag_tensor
from data.construct_tensors.construct_nodes_and_labels import generate_node_features_with_features

def prepare_training_data(
    tickers,
    start_date,
    end_date,
    matrix_lookback='5min',
    sequence_length=5,
    freq='1min',
    data_folder='data/ohlcv',
    output_path='data/processed/training_data.pt'
):
    # Generate adjacency matrices
    adj_matrices, adj_timestamps, adj_nodes = generate_leadlag_tensor(
        tickers,
        start_date,
        end_date,
        matrix_lookback,
        sequence_length,
        freq,
        data_folder
    )

    # Generate node features and labels, and also capture feature names
    node_features, label_tensor, feat_timestamps, feat_nodes, feature_names = generate_node_features_with_features(
        tickers,
        start_date,
        end_date,
        matrix_lookback,
        sequence_length,
        freq,
        data_folder
    )

    # Verify alignment
    assert len(adj_timestamps) == len(feat_timestamps), "Timestamp mismatch"
    assert adj_nodes == feat_nodes, "Node list mismatch"

    # Create ticker to index mapping using feat_nodes (the ordered node list)
    ticker_to_index = {ticker: idx for idx, ticker in enumerate(feat_nodes)}
    # Create feature_to_index mapping using the feature names from construct_nodes_and_labels.py
    feature_to_index = {feature: idx for idx, feature in enumerate(feature_names)}

    # Convert to PyTorch tensors and include mapping in metadata
    data_dict = {
        'features': torch.FloatTensor(node_features),
        'adjacency': torch.FloatTensor(adj_matrices),
        'labels': torch.FloatTensor(label_tensor),
        'timestamps': feat_timestamps,
        'nodes': feat_nodes,
        'metadata': {
            'num_features': node_features.shape[-1],
            'sequence_length': sequence_length,
            'num_nodes': len(feat_nodes),
            'num_samples': node_features.shape[0],
            'ticker_to_index': ticker_to_index,
            'feature_to_index': feature_to_index  # New mapping added here.
        }
    }

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to file
    torch.save(data_dict, output_path)
    print(f"Data saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Example usage
    tickers = {'AAPL', 'AMZN', 'TSLA'}
    data_path = prepare_training_data(
        tickers=tickers,
        start_date='2024-11-28',
        end_date='2024-11-29',
        matrix_lookback='5min',
        sequence_length=5,
        freq='1min',
        data_folder='data/ohlcv',
        output_path='data/processed/training_data.pt'
    )