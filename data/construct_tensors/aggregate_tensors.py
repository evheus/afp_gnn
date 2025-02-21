import torch
import os
import sys
import pandas as pd
from datetime import datetime
import pytz
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from data.construct_tensors.construct_lead_lag_matrices import generate_leadlag_tensor_from_data
from data.construct_tensors.construct_nodes_and_labels import generate_node_features_with_features_from_data
from data.utils.data_utils import load_and_preprocess_all_data

def prepare_training_data(
    tickers: set,  # Note: tickers should be an ordered set or list
    start_date,
    end_date,
    matrix_lookback='5min',
    sequence_length=5,
    freq='1min',
    data_folder='data/ohlcv',
    output_path='data/processed/training_data.pt'
):
    # Convert tickers to sorted list for consistent ordering
    ordered_tickers = sorted(list(tickers))
    
    print("Loading and preprocessing data...")
    # Load and preprocess data only once, including SPY
    all_tickers = ordered_tickers + ['SPY']
    processed_data = load_and_preprocess_all_data(all_tickers, [start_date, end_date], data_folder)
    
    print("Generating lead-lag matrices...")
    # Generate adjacency matrices using the shared processed_data
    adj_matrices, adj_timestamps, adj_nodes = generate_leadlag_tensor_from_data(
        processed_data, ordered_tickers, matrix_lookback, sequence_length, freq
    )
    
    print("Generating node features and labels...")
    # Generate node features/labels using the same processed_data
    node_features, label_tensor, feat_timestamps, feat_nodes, feature_names = generate_node_features_with_features_from_data(
        processed_data, ordered_tickers, matrix_lookback, sequence_length, freq
    )
    
    # Verify alignment between lead-lag and feature modules
    assert len(adj_timestamps) == len(feat_timestamps), "Timestamp mismatch between lead-lag and features"
    assert adj_nodes == feat_nodes, "Node list mismatch between lead-lag and features"
    assert len(adj_timestamps) > 0, "No valid timestamps were generated"

    # Debug prints for timestamp lists
    print("DEBUG: Number of adj_timestamps (lead-lag):", len(adj_timestamps))
    print("DEBUG: Number of feat_timestamps (features):", len(feat_timestamps))
    if adj_timestamps and feat_timestamps:
        print("DEBUG: First adj_timestamp:", adj_timestamps[0])
        print("DEBUG: First feat_timestamp:", feat_timestamps[0])
    
    # Create mappings from tickers and feature names to indices
    ticker_to_index = {ticker: idx for idx, ticker in enumerate(feat_nodes)}
    feature_to_index = {feature: idx for idx, feature in enumerate(feature_names)}
    
    # Check tensor shapes:
    # node_features: (num_samples, sequence_length, num_nodes, num_features)
    # label_tensor: (num_samples, num_nodes, num_features)
    num_samples = node_features.shape[0]
    num_nodes = node_features.shape[2]
    num_features = node_features.shape[3]
    
    assert label_tensor.shape[0] == num_samples, "Mismatch in number of samples between features and labels"
    assert label_tensor.shape[1] == num_nodes, "Mismatch in number of nodes between features and labels"
    assert label_tensor.shape[2] == num_features, "Mismatch in number of features between features and labels"
    
    # Ensure there are no NaN values in the tensors.
    assert not np.isnan(node_features).any(), "NaN values detected in node_features tensor"
    assert not np.isnan(label_tensor).any(), "NaN values detected in label_tensor"
    
    # Build data dictionary with metadata
    data_dict = {
        'features': torch.FloatTensor(node_features),
        'adjacency': torch.FloatTensor(adj_matrices),
        'labels': torch.FloatTensor(label_tensor),
        'timestamps': feat_timestamps,
        'nodes': feat_nodes,
        'metadata': {
            'num_features': num_features,
            'sequence_length': sequence_length,
            'num_nodes': len(feat_nodes),
            'num_samples': num_samples,
            'ticker_to_index': ticker_to_index,
            'feature_to_index': feature_to_index
        }
    }
    
    # Final checks on mappings
    assert ticker_to_index, "Ticker mapping is empty"
    assert feature_to_index, "Feature mapping is empty"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data_dict, output_path)
    print(f"Data saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Use an ordered list instead of a set for consistent ordering
    tickers = ['AAPL', 'AMZN', 'TSLA']  # or sorted(['AAPL', 'AMZN', 'TSLA'])
    start_date = '2024-11-25'
    end_date   = '2024-11-29'
    
    data_path = prepare_training_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        matrix_lookback='5min',
        sequence_length=5,
        freq='1min',
        data_folder='data/ohlcv',
        output_path='data/processed/training_data.pt'
    )