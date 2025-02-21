import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple
from datetime import datetime, timedelta, time

project_root = os.getcwd()
sys.path.append(project_root)

from data.utils.data_utils import load_data_for_tickers
from data.utils.lead_lag import construct_lead_lag_matrix

def load_and_process_data(
    tickers: set,
    start_date: str,
    end_date: str,
    matrix_lookback: str = '30min',
    freq: str = '1min',
    data_folder: str = 'data/ohlcv',
    max_lag: int = 5
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Load data and create initial lead-lag matrices.
    
    Returns:
    --------
    Dict[pd.Timestamp, pd.DataFrame]
        Dictionary of timestamp-indexed lead-lag matrices
    """
    # Load and preprocess data
    dfs = load_data_for_tickers(tickers, [start_date, end_date], data_folder)
    
    # Create multi-index DataFrame
    processed_data = pd.concat(dfs, axis=1)
    processed_data.columns = pd.MultiIndex.from_product([dfs.keys(), dfs[list(dfs.keys())[0]].columns])
    
    # Generate lead-lag matrices
    lead_lag_matrices = {}
    
    # Create calculation points
    calc_start = processed_data.index.min() + pd.Timedelta(matrix_lookback)
    calc_points = pd.date_range(start=calc_start,
                              end=processed_data.index.max(),
                              freq=freq)
    
    for current_time in calc_points:
        window_start = current_time - pd.Timedelta(matrix_lookback)
        window_data = processed_data.loc[window_start:current_time]
        
        if len(window_data) > 0:
            try:
                lead_lag_matrix = construct_lead_lag_matrix(
                    window_data,
                    method='C1',
                    max_lag=max_lag
                )
                lead_lag_matrices[current_time] = lead_lag_matrix
                
            except Exception as e:
                print(f"Error calculating lead-lag matrix for time {current_time}: {str(e)}")
    
    return lead_lag_matrices

def create_rolling_windows(
    matrices: Dict[pd.Timestamp, pd.DataFrame],
    sequence_length: int,
    frequency: str = '1min'
) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    timestamps = sorted(matrices.keys())
    num_nodes = matrices[timestamps[0]].shape[0]
    
    # Create frequency-based sampling points
    start_time_point = timestamps[sequence_length - 1]
    end_time_point = timestamps[-2]  # ensure next label exists
    sampling_points = pd.date_range(start=start_time_point, end=end_time_point, freq=frequency)
    
    # Define market hours
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    # Filter sampling points to include only timestamps within trading hours and that exist in data.
    valid_sampling_points = [
        t for t in sampling_points
        if market_open <= t.time() <= market_close and t in timestamps
    ]
    
    # Debug: print valid sampling points for lead-lag tensor
    print("Valid sampling points for lead-lag:", valid_sampling_points)
    
    num_samples = len(valid_sampling_points)
    tensor = np.zeros((num_samples, sequence_length, num_nodes, num_nodes))
    window_timestamps = []
    
    for i, cur_time in enumerate(valid_sampling_points):
        current_idx = timestamps.index(cur_time)
        for j in range(sequence_length):
            window_time = timestamps[current_idx - (sequence_length - 1) + j]
            tensor[i, j] = matrices[window_time].values
        window_timestamps.append(cur_time)
    
    return tensor, window_timestamps

def generate_leadlag_tensor(
    tickers: set,
    start_date: str,
    end_date: str,
    matrix_lookback: str = '30min',
    sequence_length: int = 5,
    freq: str = '1min',
    data_folder: str = 'data/ohlcv',
    max_lag: int = 5
) -> Tuple[np.ndarray, List[pd.Timestamp], List[str]]:
    """
    Generate the final 4D lead-lag tensor from raw data.
    
    Parameters:
    -----------
    tickers : set
        Set of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    matrix_lookback : str
        Lookback window for individual lead-lag matrix calculation
    sequence_length : int
        Number of historical matrices to include in each tensor window
    freq : str
        Frequency for sampling windows
    data_folder : str
        Folder containing OHLCV data
    max_lag : int
        Maximum lag for lead-lag calculation
        
    Returns:
    --------
    Tuple containing:
        - numpy.ndarray: 4D tensor of shape (num_samples, sequence_length, num_nodes, num_nodes)
        - List[pd.Timestamp]: Timestamps for each window
        - List[str]: List of node identifiers in order
    """
    # Generate initial matrices
    matrices = load_and_process_data(
        tickers,
        start_date,
        end_date,
        matrix_lookback,
        freq,
        data_folder,
        max_lag
    )
    
    # Create rolling windows tensor
    tensor, window_timestamps = create_rolling_windows(
        matrices,
        sequence_length,
        freq
    )
    
    # Get ordered list of nodes
    node_list = list(matrices[window_timestamps[0]].index)
    
    return tensor, window_timestamps, node_list

def display_tensor_info(
    tensor: np.ndarray,
    window_timestamps: List[pd.Timestamp],
    node_list: List[str]
) -> None:
    """Display information about the created tensor"""
    print(f"Tensor shape: {tensor.shape}")
    print(f"Interpretation:")
    print(f"- Number of samples: {tensor.shape[0]}")
    print(f"- Sequence length: {tensor.shape[1]}")
    print(f"- Matrix dimensions: {tensor.shape[2]}x{tensor.shape[3]}")
    print(f"\nNode list: {node_list}")
    print(f"\nFirst window ends at: {window_timestamps[0]}")
    print(f"Last window ends at: {window_timestamps[-1]}")
    
    # Show example window
    print(f"\nExample - First window matrices:")
    for i in range(tensor.shape[1]):
        print(f"\nMatrix {i+1} of {tensor.shape[1]}:")
        df = pd.DataFrame(
            tensor[0, i],
            index=node_list,
            columns=node_list
        )
        print(df.round(4))

def generate_leadlag_tensor_from_data(
    processed_data: pd.DataFrame,
    tickers: set,
    matrix_lookback: str,
    sequence_length: int,
    freq: str
) -> Tuple[np.ndarray, List[pd.Timestamp], List[str]]:
    """
    Generate lead-lag tensor from preprocessed data. Only windows with sufficient data
    are used. In this implementation we filter out calc points before market open.
    """
    lead_lag_matrices = {}

    # Define a market open time used as an additional filter.
    market_open_time = time(9, 30)
    
    # Calculate start of calculation points. We ensure that the earliest calc point
    # is at least matrix_lookback after data starts.
    calc_start = processed_data.index.min() + pd.Timedelta(matrix_lookback)
    # Generate calc points over the entire data range...
    calc_points = pd.date_range(start=calc_start, end=processed_data.index.max(), freq=freq)
    
    # Filter calc points to only include those at or after market open.
    calc_points = [t for t in calc_points if t.time() >= market_open_time]
    
    # Now compute matrices for each calc point. Skip if window has fewer than 2 data points.
    for current_time in calc_points:
        window_start = current_time - pd.Timedelta(matrix_lookback)
        # Check if the window start is before our data starts; if so, skip.
        if window_start < processed_data.index.min():
            continue
        
        window_data = processed_data.loc[window_start:current_time]
        
        if len(window_data) < 2:
            print(f"Skipping {current_time}: window size ({len(window_data)}) is too small to compute covariance.")
            continue
        
        try:
            lead_lag_matrix = construct_lead_lag_matrix(window_data, method='C1')
            lead_lag_matrices[current_time] = lead_lag_matrix
        except Exception as e:
            print(f"Error calculating lead-lag for {current_time}: {e}")
    
    # Create rolling windows tensor
    tensor, window_timestamps = create_rolling_windows(lead_lag_matrices, sequence_length, freq)
    if not window_timestamps:
        raise ValueError("No valid rolling windows were generated from lead-lag matrices.")
    # Assume ordering of nodes is determined by the index of the first valid matrix.
    node_list = list(lead_lag_matrices[window_timestamps[0]].index)
    return tensor, window_timestamps, node_list

if __name__ == "__main__":
    # Use an ordered list instead of a set for consistent ordering
    tickers = ['AAPL', 'AMZN', 'TSLA']  # or sorted(['AAPL', 'AMZN', 'TSLA'])
    start_date = '2024-11-28'
    end_date = '2024-11-29'
    matrix_lookback = '5min'
    sequence_length = 5
    freq = '1min'
    data_folder = os.path.join(project_root, 'data', 'ohlcv')
    
    # Generate tensor
    tensor, window_timestamps, node_list = generate_leadlag_tensor(
        tickers,
        start_date,
        end_date,
        matrix_lookback,
        sequence_length,
        freq,
        data_folder
    )
    
    # Display information
    display_tensor_info(tensor, window_timestamps, node_list)

    print(tensor.shape)