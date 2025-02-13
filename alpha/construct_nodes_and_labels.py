import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

project_root = os.getcwd()
sys.path.append(project_root)

from data.data_utils import load_data_for_tickers

def calculate_volatility(data: pd.DataFrame, window: str = '30min') -> pd.Series:
    """
    Calculate rolling volatility for price data.
    """
    returns = data.pct_change()
    volatility = returns.rolling(window=window).std()
    return volatility

def create_feature_matrices(
    tickers: set,
    start_date: str,
    end_date: str,
    matrix_lookback: str = '30min',
    freq: str = '1min',
    data_folder: str = 'data/ohlcv'
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Create feature matrices for each timestamp.
    """
    # Load data
    dfs = load_data_for_tickers(tickers, [start_date, end_date], data_folder)
    
    # Create multi-index DataFrame
    processed_data = pd.concat(dfs, axis=1)
    processed_data.columns = pd.MultiIndex.from_product([dfs.keys(), dfs[list(dfs.keys())[0]].columns])
    
    # Initialize dictionary for feature matrices
    feature_matrices = {}
    
    # Create calculation points
    calc_start = processed_data.index.min() + pd.Timedelta(matrix_lookback)
    calc_points = pd.date_range(start=calc_start,
                              end=processed_data.index.max(),
                              freq=freq)
    
    for ticker in tickers:
        # Calculate returns and volatility for each ticker
        price_data = processed_data[ticker]['close']
        processed_data[(ticker, 'returns')] = price_data.pct_change()
        processed_data[(ticker, 'volatility')] = calculate_volatility(price_data, matrix_lookback)
    
    # Select features for each timestamp
    for current_time in calc_points:
        features = []
        
        for ticker in tickers:
            # Get returns, volume, and volatility
            returns = processed_data.loc[current_time, (ticker, 'returns')]
            volume = processed_data.loc[current_time, (ticker, 'volume')]
            volatility = processed_data.loc[current_time, (ticker, 'volatility')]
            
            features.append([returns, volume, volatility])
        
        # Create feature matrix for current timestamp
        feature_matrices[current_time] = pd.DataFrame(
            features,
            index=list(tickers),
            columns=['returns', 'volume', 'volatility']
        )
    
    return feature_matrices

def create_feature_and_label_tensors(
    matrices: Dict[pd.Timestamp, pd.DataFrame],
    sequence_length: int,
    frequency: str = '1min'
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Create rolling windows tensor and corresponding label tensor.
    
    Args:
        matrices: Dictionary mapping timestamps to feature matrices
        sequence_length: Number of sequential matrices to include in each sample
        frequency: Sampling frequency for the windows
        
    Returns:
        Tuple containing:
        - node_features: Shape (num_samples, sequence_length, num_nodes, num_features)
        - label_tensor: Shape (num_samples, num_nodes, num_features)
        - window_timestamps: List of timestamps for each window
    """
    timestamps = sorted(matrices.keys())
    num_nodes = matrices[timestamps[0]].shape[0]
    num_features = matrices[timestamps[0]].shape[1]
    
    # Create frequency-based sampling points
    start_time = timestamps[sequence_length - 1]
    end_time = timestamps[-2]  # Use second-to-last timestamp to ensure labels exist
    sampling_points = pd.date_range(start=start_time, end=end_time, freq=frequency)
    
    # Filter sampling points
    valid_sampling_points = [t for t in sampling_points if t in timestamps]
    
    # Initialize tensors
    num_samples = len(valid_sampling_points)
    node_features = np.zeros((num_samples, sequence_length, num_nodes, num_features))
    label_tensor = np.zeros((num_samples, num_nodes, num_features))
    window_timestamps = []
    
    # Fill tensors with rolling windows and labels
    for i, current_time in enumerate(valid_sampling_points):
        current_idx = timestamps.index(current_time)
        
        # Fill feature tensor
        for j in range(sequence_length):
            matrix_time = timestamps[current_idx - (sequence_length - 1) + j]
            node_features[i, j] = matrices[matrix_time].values
        
        # Fill label tensor with next timestep's features
        next_time = timestamps[current_idx + 1]
        label_tensor[i] = matrices[next_time].values
        
        window_timestamps.append(current_time)
    
    return node_features, label_tensor, window_timestamps

def generate_node_features(
    tickers: set,
    start_date: str,
    end_date: str,
    matrix_lookback: str = '30min',
    sequence_length: int = 5,
    freq: str = '1min',
    data_folder: str = 'data/ohlcv'
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
    """
    Generate feature and label tensors from raw data.
    
    Returns:
        Tuple containing:
        - node_features: Shape (num_samples, sequence_length, num_nodes, num_features)
        - label_tensor: Shape (num_samples, num_nodes, num_features)
        - window_timestamps: List of timestamps for each window
        - node_list: List of node identifiers in order
    """
    # Generate feature matrices
    matrices = create_feature_matrices(
        tickers,
        start_date,
        end_date,
        matrix_lookback,
        freq,
        data_folder
    )
    
    # Create feature and label tensors
    node_features, label_tensor, window_timestamps = create_feature_and_label_tensors(
        matrices,
        sequence_length,
        freq
    )
    
    # Get ordered list of nodes
    node_list = list(matrices[window_timestamps[0]].index)
    
    return node_features, label_tensor, window_timestamps, node_list

def display_tensor_info(
    node_features: np.ndarray,
    label_tensor: np.ndarray,
    window_timestamps: List[pd.Timestamp],
    node_list: List[str]
) -> None:
    """Display information about the created tensors"""
    print(f"Node features tensor shape: {node_features.shape}")
    print(f"Label tensor shape: {label_tensor.shape}")
    print(f"\nInterpretation:")
    print(f"- Number of samples: {node_features.shape[0]}")
    print(f"- Sequence length: {node_features.shape[1]}")
    print(f"- Number of nodes: {node_features.shape[2]}")
    print(f"- Number of features: {node_features.shape[3]}")
    print(f"\nNode list: {node_list}")
    print(f"\nFirst window ends at: {window_timestamps[0]}")
    print(f"Last window ends at: {window_timestamps[-1]}")
    
    # Show example window and its label
    print(f"\nExample - First window feature matrices and label:")
    feature_names = ['returns', 'volume', 'volatility']
    
    print("\nFeature sequence:")
    for i in range(node_features.shape[1]):
        print(f"\nTime point {i+1} of {node_features.shape[1]}:")
        for j, feature in enumerate(feature_names):
            print(f"\n{feature.capitalize()} values:")
            df = pd.DataFrame(
                node_features[0, i, :, j],
                index=node_list,
                columns=[feature]
            )
            print(df.round(4))
    
    print("\nLabel (next timestep features):")
    for j, feature in enumerate(feature_names):
        print(f"\n{feature.capitalize()} values:")
        df = pd.DataFrame(
            label_tensor[0, :, j],
            index=node_list,
            columns=[feature]
        )
        print(df.round(4))

if __name__ == "__main__":
    # Set parameters
    tickers = {'AAPL', 'AMZN', 'TSLA'}
    start_date = '2024-11-28'
    end_date = '2024-11-29'
    matrix_lookback = '5min'
    sequence_length = 5
    freq = '1min'
    data_folder = os.path.join(project_root, 'data', 'ohlcv')
    
    # Generate tensors
    node_features, label_tensor, window_timestamps, node_list = generate_node_features(
        tickers,
        start_date,
        end_date,
        matrix_lookback,
        sequence_length,
        freq,
        data_folder
    )
    
    # Display information
    display_tensor_info(node_features, label_tensor, window_timestamps, node_list)