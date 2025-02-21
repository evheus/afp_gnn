import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple
from datetime import datetime, timedelta, time

project_root = os.getcwd()
sys.path.append(project_root)

from data.utils.data_utils import load_data_for_tickers

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
) -> Tuple[Dict[pd.Timestamp, pd.DataFrame], List[str]]:
    """
    Create feature matrices for each timestamp and return the column names.
    """
    # Use tickers as is - don't add SPY here since it's already in the input set
    ordered_tickers = sorted(list(tickers))
    
    # Load data
    dfs = load_data_for_tickers(ordered_tickers, [start_date, end_date], data_folder)
    
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
    
    # Use all tickers including SPY
    all_tickers = list(dfs.keys())  # Changed from tickers to dfs.keys()
    
    for ticker in all_tickers:  # Changed from tickers to all_tickers
        # Calculate returns and volatility for each ticker
        price_data = processed_data[ticker]['close']
        processed_data[(ticker, 'returns')] = price_data.pct_change()
        processed_data[(ticker, 'volatility')] = calculate_volatility(price_data, matrix_lookback)
    
    # # Add this in create_feature_matrices after volatility calculation
    # for ticker in all_tickers:
    #     vol = processed_data[(ticker, 'volatility')].mean()
    #     print(f"Average volatility for {ticker}: {vol:.4f}")
    
    # Calculate sampling points
    calc_start = processed_data.index.min() + pd.Timedelta(matrix_lookback)
    calc_points = pd.date_range(start=calc_start,
                                end=processed_data.index.max(),
                                freq=freq)

    # Define market open and close times.
    market_open = time(9, 30)
    market_close = time(16, 0)

    # Filter calc_points to only include times within trading hours and that exist in processed_data.index
    valid_calc_points = [
        t for t in calc_points 
        if market_open <= t.time() <= market_close and t in processed_data.index
    ]

    # Debug: print valid calculation points for features
    print("Valid calc points for features:", valid_calc_points)

    feature_matrices = {}
    feature_names = ['returns', 'log_volume', 'volatility']

    for current_time in valid_calc_points:
        features = []
        # Change this line to use ordered_tickers instead of tickers
        for ticker in ordered_tickers:  # <- This is the key change
            returns = processed_data.loc[current_time, (ticker, 'returns')]
            volume = processed_data.loc[current_time, (ticker, 'volume')]
            log_volume = np.log(volume + 1e-8)
            volatility = processed_data.loc[current_time, (ticker, 'volatility')]
            features.append([returns, log_volume, volatility])
        
        feature_matrices[current_time] = pd.DataFrame(
            features,
            index=ordered_tickers,  # <- Also change this
            columns=feature_names
        )

    return feature_matrices, feature_names

def create_feature_and_label_tensors(
    matrices: Dict[pd.Timestamp, pd.DataFrame],
    sequence_length: int,
    frequency: str = '1min'
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Create rolling windows tensor and corresponding label tensor.
    """
    timestamps = sorted(matrices.keys())
    num_nodes = matrices[timestamps[0]].shape[0]
    num_features = matrices[timestamps[0]].shape[1]
    
    # Create frequency-based sampling points
    start_time = timestamps[sequence_length - 1]
    end_time = timestamps[-2]  # Ensure labels exist
    
    sampling_points = pd.date_range(start=start_time, end=end_time, freq=frequency)
    
    # Define market open and close times.
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    # Filter sampling points to only include times within trading hours
    valid_sampling_points = [
        t for t in sampling_points
        if market_open <= t.time() <= market_close and t in timestamps
    ]
    
    # Initialize tensors
    num_samples = len(valid_sampling_points)
    node_features = np.zeros((num_samples, sequence_length, num_nodes, num_features))
    label_tensor = np.zeros((num_samples, num_nodes, num_features))
    window_timestamps = []
    
    # Fill tensors with rolling windows and labels
    for i, current_time in enumerate(valid_sampling_points):
        current_idx = timestamps.index(current_time)
        
        # Fill feature tensor for each window
        for j in range(sequence_length):
            window_time = timestamps[current_idx - (sequence_length - 1) + j]
            node_features[i, j] = matrices[window_time].values
        
        # Fill label tensor with the next timestep's features
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
    matrices, feature_names = create_feature_matrices(
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

def generate_node_features_with_features(
    tickers: set,
    start_date: str,
    end_date: str,
    matrix_lookback: str = '30min',
    sequence_length: int = 5,
    freq: str = '1min',
    data_folder: str = 'data/ohlcv'
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str], List[str]]:
    """
    Generate feature and label tensors from raw data along with feature names.
    
    Returns:
        node_features: (num_samples, sequence_length, num_nodes, num_features)
        label_tensor: (num_samples, num_nodes, num_features)
        window_timestamps: list of timestamps for each window
        node_list: list of node identifiers in order
        feature_names: list of feature names (e.g., ['returns', 'volume', 'volatility'])
    """
    matrices, feature_names = create_feature_matrices(
        tickers,
        start_date,
        end_date,
        matrix_lookback,
        freq,
        data_folder
    )
    
    node_features, label_tensor, window_timestamps = create_feature_and_label_tensors(
        matrices,
        sequence_length,
        freq
    )
    
    node_list = list(matrices[window_timestamps[0]].index)
    
    return node_features, label_tensor, window_timestamps, node_list, feature_names

def generate_node_features_with_features_from_data(
    processed_data: pd.DataFrame,
    tickers: set,
    matrix_lookback: str,
    sequence_length: int,
    freq: str
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str], List[str]]:
    """
    Generate node features and labels using the given processed_data.
    """
    # First build the feature matrices.
    feature_matrices, feature_names = create_feature_matrices_from_data(
        processed_data, tickers, matrix_lookback, freq
    )
    # Then create tensors with rolling windows.
    node_features, label_tensor, feat_timestamps = create_feature_and_label_tensors(
        feature_matrices, sequence_length, freq
    )
    # Suppose the ordered node list is the DataFrame index of one feature matrix.
    feat_nodes = list(feature_matrices[next(iter(feature_matrices))].index)
    return node_features, label_tensor, feat_timestamps, feat_nodes, feature_names

def create_feature_matrices_from_data(
    processed_data: pd.DataFrame,
    tickers: set,
    matrix_lookback: str,
    freq: str
) -> Tuple[Dict[pd.Timestamp, pd.DataFrame], List[str]]:
    from datetime import time as dt_time
    calc_start = processed_data.index.min() + pd.Timedelta(matrix_lookback)
    calc_points = pd.date_range(start=calc_start, end=processed_data.index.max(), freq=freq)
    
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    
    valid_calc_points = [
        t for t in calc_points
        if market_open <= t.time() <= market_close and t in processed_data.index
    ]
    print("DEBUG: Valid calc points for features:", valid_calc_points)
    
    feature_matrices = {}
    feature_names = ['returns', 'log_volume', 'volatility']
    for current_time in valid_calc_points:
        features = []
        for ticker in tickers:
            returns = processed_data.loc[current_time, (ticker, 'returns')]
            volume = processed_data.loc[current_time, (ticker, 'volume')]
            log_volume = np.log(volume + 1e-8)
            volatility = processed_data.loc[current_time, (ticker, 'volatility')]
            features.append([returns, log_volume, volatility])
        
        feature_matrices[current_time] = pd.DataFrame(
            features,
            index=list(tickers),
            columns=feature_names
        )
    
    return feature_matrices, feature_names

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
    # Use an ordered list instead of a set for consistent ordering
    tickers = ['AAPL', 'AMZN', 'TSLA']  # Ensure consistent order
    all_tickers = sorted(tickers + ['SPY'])  # Add SPY to the list, and sort once
    start_date = '2024-11-21'
    end_date = '2024-11-22'
    matrix_lookback = '5min'
    sequence_length = 5
    freq = '1min'
    data_folder = os.path.join(project_root, 'data', 'ohlcv')
    
    # Generate tensors with SPY included in processing
    node_features, label_tensor, window_timestamps, node_list = generate_node_features(
        set(all_tickers),  # Pass all_tickers including SPY
        start_date,
        end_date,
        matrix_lookback,
        sequence_length,
        freq,
        data_folder
    )
    
    # Display information
    display_tensor_info(node_features, label_tensor, window_timestamps, node_list)
    #print(node_features.shape, label_tensor.shape)