import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple
from datetime import datetime, timedelta, time
import torch

project_root = os.getcwd()
sys.path.append(project_root)

from data.utils.data_utils import load_and_preprocess_all_data
from data.utils.lead_lag import construct_lead_lag_matrix





def create_model_tensors(
    tickers: list,
    start_date: str,
    end_date: str,
    matrix_lookback: str = '5min',
    sequence_length: int = 5,
    freq: str = '1min',
    data_folder: str = 'data/ohlcv'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
    """
    Create all tensors needed for the model in a single pipeline:
    1. Load and preprocess data (including SPY)
    2. Generate lead-lag tensors
    3. Generate node feature tensors
    4. Generate label tensors
    
    Returns:
        adj_tensor: Shape (num_samples, sequence_length, num_nodes, num_nodes)
        node_features: Shape (num_samples, sequence_length, num_nodes, num_features)
        label_tensor: Shape (num_samples, num_nodes, num_features)
        timestamps: List of timestamps for each window
        node_list: List of nodes in consistent order ['AAPL', 'AMZN', 'SPY', 'TSLA']
    """
    # 1. Ensure consistent ordering of tickers
    ordered_tickers = sorted(tickers) + ['SPY']
    print(f"Processing tickers in order: {ordered_tickers}")
    
    # 2. Load and preprocess data once for all calculations
    print("Loading and preprocessing data...")
    processed_data = load_and_preprocess_all_data(ordered_tickers, [start_date, end_date], data_folder)
    
    # print("\nStep 1: Verify data loading")
    # print("Processed data columns:", processed_data.columns)
    # print("Processed data head:\n", processed_data['AAPL']['log_return'][processed_data.index.date == pd.to_datetime('2024-11-22').date()])
    # # # print("Unique dates:", pd.unique(processed_data.index.date))
    # halt = input("Press Enter to continue...")
    
    # 3. Generate lead-lag tensors
    print("Generating lead-lag matrices...")
    edges_sparse, edge_weights, adj_timestamps, adj_nodes = generate_leadlag_tensor_from_data(
        processed_data,
        ordered_tickers,
        matrix_lookback,
        sequence_length,
        freq
    )

    
    # 4. Generate node features and labels
    print("Generating node features and labels...")
    node_features, label_tensor, feat_timestamps, feat_nodes, feature_names = generate_node_features_from_data(
        processed_data,
        ordered_tickers,
        matrix_lookback,
        sequence_length,
        freq
    )
    #halt = input("Press Enter to continue...")
      
    
    # 5. Verify alignment
    assert len(adj_timestamps) == len(feat_timestamps), "Timestamp mismatch between lead-lag and features"
    assert adj_nodes == feat_nodes, "Node list mismatch between lead-lag and features"
    assert len(adj_timestamps) > 0, "No valid timestamps were generated"
    
    print(f"\nTensor shapes:")
    print(f"Adjacency edges tensor: {edges_sparse.shape}")
    print(f"Edge weights tensor: {edge_weights.shape}")
    print(f"Node features tensor: {node_features.shape}")
    print(f"Label tensor: {label_tensor.shape}")
    print(f"\nNode list: {feat_nodes}")
    print(f"Number of timestamps: {len(feat_timestamps)}")
    
    return edges_sparse, edge_weights, node_features, label_tensor, feat_timestamps, feat_nodes

# def generate_leadlag_tensor_from_data(
#     processed_data: pd.DataFrame,
#     ordered_tickers: list,
#     matrix_lookback: str,
#     sequence_length: int,
#     freq: str
# ) -> Tuple[np.ndarray, List[pd.Timestamp], List[str]]:
    
#     """Generate lead-lag tensor from preprocessed data."""
#     lead_lag_matrices = {}
    
#     # Define market hours
#     market_open = time(9, 30)
#     market_close = time(16, 0)

#     # try:
#     #     # Extract all digits from the beginning of the string
#     #     c1_max_lag = int(''.join(filter(str.isdigit, matrix_lookback)))
#     #     c1_max_lag = (c1_max_lag // 2) + 1  # Using integer division for clean result
#     # except ValueError:
#     #     raise ValueError(f"Invalid time string format: {matrix_lookback}")
    
#     # # Add debug prints for input validation
#     # print("\nInput Validation:")
#     # print(f"Total date range: {processed_data.index.min()} to {processed_data.index.max()}")
#     # print(f"Number of unique dates: {len(pd.unique(processed_data.index.date))}")
#     # print(f"Matrix lookback: {matrix_lookback}")
#     # print(f"Sampling frequency: {freq}")
    
#     # Modified approach
#     valid_calc_points = []
#     for date in pd.unique(processed_data.index.date):
#         # Get data for this date
#         date_data = processed_data[processed_data.index.date == date]
        
#         # Calculate start point for this day
#         daily_calc_start = date_data.index.min() + pd.Timedelta(matrix_lookback) #+ pd.Timedelta(freq)
        
#         # # Add debug prints for each day
#         # print(f"\nProcessing date: {date}")
#         # print(f"Day start: {date_data.index.min()}")
#         # print(f"First calc point: {daily_calc_start}")
#         # print(f"Day end: {date_data.index.max()}")
        
#         # Generate calc points for this day
#         daily_calc_points = pd.date_range(
#             start=daily_calc_start,
#             end=date_data.index.max(),
#             freq=freq
#         )
        
#         # Add assertions to verify daily calculations
#         assert daily_calc_start.date() == date, "Calculation start should be on the same day"
#         assert len(daily_calc_points) > 0, f"No calculation points generated for {date}"
        
#         # Filter for market hours and existing data points
#         daily_valid_points = [
#             t for t in daily_calc_points 
#             if market_open <= t.time() < market_close and t in processed_data.index  # Changed <= to <
#         ]
        
#         # # Debug print for valid points
#         # print(f"Number of valid calc points for {date}: {len(daily_valid_points)}")
#         # if len(daily_valid_points) > 0:
#         #     print(f"First valid point: {daily_valid_points[0]}")
#         #     print(f"Last valid point: {daily_valid_points[-1]}")
        
#         # print(daily_valid_points[:1], daily_valid_points[-1:])
#         # halt = input("Press Enter to continue...")

#         valid_calc_points.extend(daily_valid_points)
    
#     # Add verification for window data
#     for current_time in valid_calc_points:
#         window_start = current_time - pd.Timedelta(matrix_lookback) + pd.Timedelta(freq)
#         # # Debug print for window boundaries
#         # if current_time == valid_calc_points[0]:  # Only print for first window
#         #     print(f"\nFirst window analysis:")
#         #     print(f"Window start: {window_start}")
#         #     print(f"Window end: {current_time}")
#         #     print(f"Window duration: {current_time - window_start}")
        
#         # Create a MultiIndex DataFrame with normalized returns
#         window_data = pd.DataFrame()
#         for ticker in ordered_tickers:
#             window_data[(ticker, 'normalized_return')] = processed_data.loc[
#                 window_start:current_time, 
#                 (ticker, 'normalized_return')
#             ]
#         window_data.columns = pd.MultiIndex.from_tuples(window_data.columns)

#         # # Filter for market hours when creating window_data
#         # window_data = window_data[
#         #     (window_data.index.time > market_open) & 
#         #     (window_data.index.time <= market_close)
#         # ]
        
#         # Add assertions for window data integrity
#         assert not window_data.empty, f"Empty window data for {current_time}"
#         assert len(window_data.columns) == len(ordered_tickers), "Missing tickers in window"
#         assert (window_data.index >= window_start).all(), "Data before window start"
#         assert (window_data.index <= current_time).all(), "Data after window end"
        
#         if len(window_data) < 2:
#             print(f"Skipping {current_time}: window size ({len(window_data)}) is too small.")
#             continue
        

#         try:
#             lead_lag_matrix = construct_lead_lag_matrix(window_data, method='C1')#, max_lag=c1_max_lag - 1)
#             # print(window_data.shape, window_data, lead_lag_matrix)
#             lead_lag_matrices[current_time] = lead_lag_matrix

#         except Exception as e:
#             print(f"Error calculating lead-lag for {current_time}: {e}")
#             print("Window data shape:", window_data.shape)
#             print("Window data columns:", window_data.columns)
#             print("\nWindow data head:\n", window_data)
#             halt = input("Press Enter to continue...")

#     # Create rolling windows tensor with verification
#     if not lead_lag_matrices:
#         raise ValueError("No valid lead-lag matrices were generated")
    
#     tensor, window_timestamps = create_rolling_windows(lead_lag_matrices, sequence_length, freq)
    
#     # # Add verification assertions and debug prints
#     # print("\nRolling windows tensor verification:")
#     # print(f"Tensor shape: {tensor.shape}")
#     # print(f"Number of window timestamps: {len(window_timestamps)}")
    
#     # Verify tensor dimensions
#     num_samples, seq_len, num_nodes, _ = tensor.shape
#     assert num_samples == len(window_timestamps), (
#         f"Mismatch between tensor samples ({num_samples}) and timestamps ({len(window_timestamps)})"
#     )
#     assert seq_len == sequence_length, (
#         f"Mismatch between tensor sequence length ({seq_len}) and requested length ({sequence_length})"
#     )
#     assert num_nodes == len(ordered_tickers), (
#         f"Mismatch between tensor nodes ({num_nodes}) and tickers ({len(ordered_tickers)})"
#     )
    
#     # Verify timestamp ordering
#     assert all(window_timestamps[i] <= window_timestamps[i+1] 
#               for i in range(len(window_timestamps)-1)), "Window timestamps not in order"
    
#     # # Verify frequency between timestamps
#     # if len(window_timestamps) > 1:
#     #     time_diff = window_timestamps[1] - window_timestamps[0]
#     #     print(f"Time difference between windows: {time_diff}")
#     print("Length of lead_lag tensor:", len(window_timestamps))
#     # print(window_timestamps[:2])
#     # print(window_timestamps[-2:])

#     # Add verification prints
#     for date in pd.unique(pd.DatetimeIndex(valid_calc_points).date):
#         day_points = [t for t in valid_calc_points if t.date() == date]
#         # print(f"\nVerification for {date}:")
#         # print(f"Last feature time: {day_points[-1].time()}")  # Should be 15:59

#     print("Lead-lag tensor shape: ", tensor.shape)
#     return tensor, window_timestamps, ordered_tickers








def generate_leadlag_tensor_from_data(
    processed_data: pd.DataFrame,
    ordered_tickers: list,
    matrix_lookback: str,
    sequence_length: int,
    freq: str
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
    """Generate lead-lag sparse tensor from preprocessed data.
    
    Returns:
        edge_index_tensor: Shape (num_samples, sequence_length, 2, num_edges)
        edge_weight_tensor: Shape (num_samples, sequence_length, num_edges)
        timestamps: List of timestamps for each window
        node_list: List of nodes in consistent order
    """
    # Keep existing setup code until after lead-lag matrix construction
    lead_lag_matrices = {}
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    # Generate valid calculation points (same as before)
    valid_calc_points = []
    for date in pd.unique(processed_data.index.date):
        date_data = processed_data[processed_data.index.date == date]
        daily_calc_start = date_data.index.min() + pd.Timedelta(matrix_lookback)
        
        daily_calc_points = pd.date_range(
            start=daily_calc_start,
            end=date_data.index.max(),
            freq=freq
        )
        
        daily_valid_points = [
            t for t in daily_calc_points 
            if market_open <= t.time() < market_close and t in processed_data.index
        ]
        valid_calc_points.extend(daily_valid_points)

    # Dictionary to store sparse representations
    sparse_matrices = {}
    
    # Process each time window
    for current_time in valid_calc_points:
        window_start = current_time - pd.Timedelta(matrix_lookback) + pd.Timedelta(freq)
        
        # Create window data (same as before)
        window_data = pd.DataFrame()
        for ticker in ordered_tickers:
            window_data[(ticker, 'normalized_return')] = processed_data.loc[
                window_start:current_time, 
                (ticker, 'normalized_return')
            ]
        window_data.columns = pd.MultiIndex.from_tuples(window_data.columns)
        
        if len(window_data) < 2:
            continue

        try:
            # Get dense lead-lag matrix with validation and fallback values
            window_size = len(window_data)
            max_lag = (window_size // 2) - 1  # Ensure proper lag for correlation
            
            # # Validate data before calculation
            # std_values = window_data.std()
            # min_std_threshold = 1e-6
            
            # # Create a mask for problematic stocks (low variance)
            # problem_stocks = std_values < min_std_threshold
            
            # if problem_stocks.any():
            #     # Add small noise to problematic series
            #     for ticker in window_data.columns[problem_stocks]:
            #         noise = np.random.normal(0, min_std_threshold, len(window_data))
            #         window_data[ticker] += noise
            
            dense_matrix = construct_lead_lag_matrix(
                df=window_data,
                method='Levy',
                max_lag=max_lag
            )
            # print(dense_matrix)
            # halt = input("Press Enter to continue...")

            # Replace NaN/Inf values with zeros in the matrix
            dense_matrix = np.nan_to_num(dense_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # # Ensure matrix is skew-symmetric even after replacing NaNs
            # dense_matrix = (dense_matrix - dense_matrix.T) / 2
            
            # Convert to sparse with threshold
            threshold = 1.5
            rows, cols = np.where(dense_matrix > threshold)  # (np.abs(dense_matrix) > threshold)
            
            if len(rows) == 0:
                # If no edges meet threshold, add minimal connectivity
                # Connect each node to its next neighbor with small weights
                rows = np.arange(len(ordered_tickers) - 1)
                cols = rows + 1
                dense_matrix[rows, cols] = 0.1  # Small positive weight
                dense_matrix[cols, rows] = -0.1  # Maintain skew-symmetry
            
            edge_index = np.stack([rows, cols], axis=0)
            edge_weights = dense_matrix[rows, cols]
            
            sparse_matrices[current_time] = {
                'edge_index': edge_index,
                'edge_weights': edge_weights
            }

        except Exception as e:
            print(f"Warning: Error in lead-lag calculation at {current_time}: {e}")
            # Create fallback sparse representation with minimal connectivity
            rows = np.arange(len(ordered_tickers) - 1)
            cols = rows + 1
            edge_index = np.stack([rows, cols], axis=0)
            edge_weights = np.ones(len(rows)) * 0.1  # Small positive weights
            
            sparse_matrices[current_time] = {
                'edge_index': edge_index,
                'edge_weights': edge_weights
            }
            continue

    # Create rolling windows with sparse format
    edge_index_tensor, edge_weight_tensor, window_timestamps = create_rolling_windows_sparse(
        sparse_matrices, sequence_length, freq
    )
    
    # Verify tensor dimensions
    num_samples = edge_index_tensor.shape[0]
    assert num_samples == len(window_timestamps), (
        f"Mismatch between tensor samples ({num_samples}) and timestamps ({len(window_timestamps)})"
    )
    assert edge_index_tensor.shape[1] == sequence_length, (
        f"Mismatch between tensor sequence length ({edge_index_tensor.shape[1]}) "
        f"and requested length ({sequence_length})"
    )
    
    print("Sparse tensors shapes:")
    print(f"Edge index tensor: {edge_index_tensor.shape}")
    print(f"Edge weight tensor: {edge_weight_tensor.shape}")
    print(f"Number of timestamps: {len(window_timestamps)}")
    
    return edge_index_tensor, edge_weight_tensor, window_timestamps, ordered_tickers







def generate_node_features_from_data(
    processed_data: pd.DataFrame,
    ordered_tickers: list,
    matrix_lookback: str,
    sequence_length: int,
    freq: str
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str], List[str]]:
    """
    Generate node features and labels from preprocessed data.
    
    Returns:
        node_features: Shape (num_samples, sequence_length, num_nodes, num_features)
        label_tensor: Shape (num_samples, num_nodes, num_features)
        feat_timestamps: List of timestamps for each window
        feat_nodes: List of nodes in consistent order
        feature_names: List of feature names
    """
    # Initialize feature matrices
    feature_matrices = {}
    feature_names = ['returns', 'log_volume', 'volatility']
    
    # Define market hours
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    # Process each day separately
    valid_calc_points = []
    feature_timestamps = []
    label_timestamps = []
    for date in pd.unique(processed_data.index.date):
        date_data = processed_data[processed_data.index.date == date]
        daily_calc_start = date_data.index.min() + pd.Timedelta(matrix_lookback) #+ pd.Timedelta(freq)
        
        # Generate calc points including 16:00 for labels
        daily_calc_points = pd.date_range(
            start=daily_calc_start,
            end=date_data.index.max(),
            freq=freq
        )
        
        daily_valid_points = [
            t for t in daily_calc_points 
            if market_open <= t.time() <= market_close and t in processed_data.index 
        ]

        # print(daily_valid_points[:1], daily_valid_points[-1:])
        # halt = input("Press Enter to continue...")
        
        feature_timestamps.extend(daily_valid_points[:-1])
        label_timestamps.extend(daily_valid_points[1 + sequence_length - 1:]) 
        valid_calc_points.extend(daily_valid_points)
        
        # Calculate features for each valid point in this day
        for current_time in daily_valid_points:
            window_start = current_time - pd.Timedelta(matrix_lookback) #+ pd.Timedelta(freq)
            window_data = date_data.loc[window_start:current_time]
            # print(window_data['AAPL'][['close', 'log_return']])
            # print('here')
            # halt = input("Press Enter to continue...")

            features = []
            for ticker in ordered_tickers:
                # Calculate rolling features using only this window
                returns = window_data[(ticker, 'close')].pct_change().iloc[-1]
                # print(current_time, returns)
                # halt = input("Press Enter to continue...")
                volume = window_data[(ticker, 'volume')].iloc[-1]
                log_volume = np.log(volume + 1e-8)
                volatility = window_data[(ticker, 'close')].pct_change().std()
                features.append([returns, log_volume, volatility])
            
        
            feature_matrices[current_time] = pd.DataFrame(
                features,
                index=ordered_tickers,
                columns=feature_names
            )
    # Create intersection for label matrices
    valid_label_times = sorted(set(feature_matrices.keys()).intersection(label_timestamps))
    label_feature_matrices = {t: feature_matrices[t] for t in valid_label_times}
    
    # Redefine feature matrices to only include feature timestamps
    valid_feature_times = sorted(set(feature_matrices.keys()).intersection(feature_timestamps))
    feature_matrices = {t: feature_matrices[t] for t in valid_feature_times}
    
    # print(f"Feature matrices timestamps: {len(feature_matrices)}")
    # print(f"Label matrices timestamps: {len(label_feature_matrices)}")
    # print(f"First feature time: {valid_feature_times[0]}")
    # print(f"Last feature time: {valid_feature_times[-1]}")
    # print(f"First feature time: {valid_label_times[0]}")
    # print(f"Last feature time: {valid_label_times[-1]}")
    # halt = input("Press Enter to continue...")

    # Create rolling windows from feature matrices
    node_features, feat_timestamps = create_rolling_windows(
        matrices=feature_matrices,
        sequence_length=sequence_length,
        freq=freq
    )
    
    # Create rolling windows from label matrices
    label_tensor, label_timestamps = create_rolling_windows(
        matrices=label_feature_matrices,
        sequence_length=1,  # Labels are single timestep
        freq=freq
    )
    
    # Reshape label tensor to remove sequence dimension since it's 1
    label_tensor = label_tensor.squeeze(1)

    # print(feat_timestamps[-1], label_timestamps[-2])
    # print("-----------------")
    
    # Verify shapes and alignment
    # print(f"Node features shape: {node_features.shape}")
    # print(f"Label tensor shape: {label_tensor.shape}")
    assert len(feat_timestamps) == len(label_timestamps), "Timestamp mismatch between features and labels"
    assert feat_timestamps[-1] == label_timestamps[-2], "Last feature time should match first label time"
    assert feat_timestamps[1] == label_timestamps[0], "Second feature timestamp should match first label timestamp"
    print(f"Node features tensor shape: {node_features.shape}")
    print(f"Label tensor shape: {label_tensor.shape}")
    # halt = input("Press Enter to continue...")
    return node_features, label_tensor, feat_timestamps, ordered_tickers, feature_names








def create_rolling_windows(
    matrices: Dict[pd.Timestamp, np.ndarray],
    sequence_length: int,
    freq: str
) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    """Helper function to create rolling windows from a dictionary of matrices."""
    timestamps = sorted(matrices.keys())
    if len(timestamps) < sequence_length:
        raise ValueError(f"Not enough timestamps ({len(timestamps)}) for sequence length {sequence_length}")
    
    # Group timestamps by date
    dates = pd.unique([ts.date() for ts in timestamps])
    
    all_tensors = []
    all_window_timestamps = []
    
    for date in dates:
        # Get timestamps for this day only
        day_timestamps = [ts for ts in timestamps if ts.date() == date]
        if len(day_timestamps) < sequence_length:
            continue
            
        num_samples = len(day_timestamps) - sequence_length + 1
        matrix_shape = matrices[timestamps[0]].shape
        day_tensor = np.zeros((num_samples, sequence_length) + matrix_shape)
        
        for i in range(num_samples):
            window_end = day_timestamps[i + sequence_length - 1]
            all_window_timestamps.append(window_end)
            
            for j in range(sequence_length):
                current_time = day_timestamps[i + j]
                day_tensor[i, j] = matrices[current_time]
                
        all_tensors.append(day_tensor)
    
    if not all_tensors:
        raise ValueError("No valid windows were created")
        
    final_tensor = np.concatenate(all_tensors, axis=0)
    return final_tensor, all_window_timestamps






# Modified rolling windows creation for sparse format
def create_rolling_windows_sparse(
    matrices: Dict[pd.Timestamp, Dict[str, np.ndarray]],
    sequence_length: int,
    freq: str
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Create rolling windows from sparse matrices, handling varying numbers of edges.
    Now pads each timestep to the maximum number of edges found in the dataset.
    
    Args:
        matrices: Dictionary mapping timestamps to dictionaries containing 'edge_index' and 'edge_weights'
        sequence_length: Number of timesteps in each sequence
        freq: Frequency of timestamps
        
    Returns:
        edge_index_tensor: Shape (num_samples, sequence_length, 2, max_edges)
        edge_weight_tensor: Shape (num_samples, sequence_length, max_edges)
        window_timestamps: List of timestamps for each window
    """
    timestamps = sorted(matrices.keys())
    if len(timestamps) < sequence_length:
        raise ValueError(f"Not enough timestamps ({len(timestamps)}) for sequence length {sequence_length}")
    
    # First pass: determine maximum number of edges
    max_edges = 0
    for ts in timestamps:
        num_edges = matrices[ts]['edge_index'].shape[1]
        max_edges = max(max_edges, num_edges)
    
    # Group timestamps by date
    dates = pd.unique([ts.date() for ts in timestamps])
    
    all_tensors_edge_index = []
    all_tensors_edge_weights = []
    all_window_timestamps = []
    
    for date in dates:
        # Get timestamps for this day only
        day_timestamps = [ts for ts in timestamps if ts.date() == date]
        if len(day_timestamps) < sequence_length:
            continue
            
        num_samples = len(day_timestamps) - sequence_length + 1
        
        # Pre-allocate day tensors with max_edges
        day_tensor_edge_index = np.zeros((num_samples, sequence_length, 2, max_edges))
        day_tensor_edge_weights = np.zeros((num_samples, sequence_length, max_edges))
        
        for i in range(num_samples):
            window_end = day_timestamps[i + sequence_length - 1]
            all_window_timestamps.append(window_end)
            
            for j in range(sequence_length):
                current_time = day_timestamps[i + j]
                curr_edges = matrices[current_time]['edge_index'].shape[1]
                
                # Pad or truncate edges to max_edges
                if curr_edges > 0:
                    day_tensor_edge_index[i, j, :, :curr_edges] = matrices[current_time]['edge_index']
                    day_tensor_edge_weights[i, j, :curr_edges] = matrices[current_time]['edge_weights']
                
        all_tensors_edge_index.append(day_tensor_edge_index)
        all_tensors_edge_weights.append(day_tensor_edge_weights)
    
    if not all_tensors_edge_index:
        raise ValueError("No valid windows were created")
        
    final_tensor_edge_index = np.concatenate(all_tensors_edge_index, axis=0)
    final_tensor_edge_weights = np.concatenate(all_tensors_edge_weights, axis=0)
    
    return final_tensor_edge_index, final_tensor_edge_weights, all_window_timestamps










def display_tensor_info(
    edges_sparse: np.ndarray,
    edge_weights: np.ndarray,
    node_features: np.ndarray,
    label_tensor: np.ndarray,
    timestamps: List[pd.Timestamp],
    node_list: List[str]
) -> None:
    """Display information about the created tensors"""
    print(f"\nTensor shapes:")
    print(f"Edges sparse tensor: {edges_sparse.shape}")
    print(f"Edge weights tensor: {edge_weights.shape}")
    print(f"Node features tensor: {node_features.shape}")
    print(f"Label tensor: {label_tensor.shape}")
    print(f"\nNode list: {node_list}")
    print(f"Number of timestamps: {len(timestamps)}")
    print(f"First timestamp: {timestamps[0]}")
    print(f"Last timestamp: {timestamps[-1]}")

def create_data_dict(
        tickers:list,
        start_date:str,
        end_date:str,
        matrix_lookback:str='5min',
        sequence_length:int=5,
        freq:str='1min',
        data_folder:str='data/ohlcv',
        output_path:str='data/processed/training_data.pt'
) -> Dict:
    """
    Create all tensors needed for the model, augment with necessary metadata, and return them in a structured dictionary.
    
    Returns:
        data_dict: Dictionary containing all tensors and metadata needed for the model
    """
    edges_sparse, edge_weights, node_features, label_tensor, timestamps, node_list = create_model_tensors(
        tickers, start_date, end_date, matrix_lookback, sequence_length, freq, data_folder
    )

    # Get shapes and create mappings
    num_samples = node_features.shape[0]
    num_nodes = node_features.shape[2]
    num_features = node_features.shape[3]

    # Create mappings
    ticker_to_index = {ticker: idx for idx, ticker in enumerate(node_list)}
    feature_names = ['returns', 'log_volume', 'volatility']
    feature_to_index = {feature: idx for idx, feature in enumerate(feature_names)}

    print(ticker_to_index, feature_to_index)

    # Create the data dictionary
    data_dict = {
        'features': torch.FloatTensor(node_features),
        'edges_sparse': torch.FloatTensor(edges_sparse),
        'edge_weights': torch.FloatTensor(edge_weights),
        'labels': torch.FloatTensor(label_tensor),
        'feature_timestamps': timestamps,
        'nodes': node_list,
        'metadata': {
            'num_features': num_features,
            'sequence_length': sequence_length,
            'num_nodes': len(node_list),
            'num_samples': num_samples,
            'ticker_to_index': ticker_to_index,
            'feature_to_index': feature_to_index
        }
    }

    assert data_dict['features'].shape == (num_samples, sequence_length, num_nodes, num_features)
    assert data_dict['edges_sparse'].shape == (num_samples, sequence_length, 2, edges_sparse.shape[-1])
    assert data_dict['edge_weights'].shape == (num_samples, sequence_length, edge_weights.shape[-1])
    assert data_dict['labels'].shape == (num_samples, num_nodes, num_features)
    assert len(data_dict['feature_timestamps']) == num_samples
    assert len(data_dict['nodes']) == num_nodes

    # Save the data dictionary
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data_dict, output_path)
    print(f"Data saved to {output_path}")
    
    return data_dict










# Example usage in __main__
if __name__ == "__main__":
    tickers = ['AAPL', 'AMZN', 'TSLA', 'NVDA', 'GOOGL', 'DDOG', 'MSFT', 'ROKU', 'ZM']  # SPY will be added automatically
    start_date = '2024-10-01' 
    end_date = '2024-11-30'
    
    data_dict = create_data_dict(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        matrix_lookback='5min',
        sequence_length=5,
        freq='1min',
        data_folder='data/ohlcv'
    )
    # Print verification info
    print("\nTensor shapes:")
    print(f"Edges Sparse: {data_dict['edges_sparse'].shape}")
    print(f"Edge Weights: {data_dict['edge_weights'].shape}")
    print(f"Features: {data_dict['features'].shape}")
    print(f"Labels: {data_dict['labels'].shape}")
    print(f"\nMetadata:")
    for key, value in data_dict['metadata'].items():
        print(f"{key}: {value}")


    # display_tensor_info(adj_tensor, node_features, label_tensor, timestamps, node_list)