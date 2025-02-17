import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

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
    """
    Create rolling windows tensor from lead-lag matrices.
    """
    timestamps = sorted(matrices.keys())
    num_nodes = matrices[timestamps[0]].shape[0]
    
    # Create frequency-based sampling points
    start_time = timestamps[sequence_length - 1]
    end_time = timestamps[-2] # so num_samples can match the feature/label generation.
    sampling_points = pd.date_range(start=start_time, end=end_time, freq=frequency)
    
    # Filter sampling points
    valid_sampling_points = [t for t in sampling_points if t in timestamps]
    
    # Initialize tensor
    num_samples = len(valid_sampling_points)
    tensor = np.zeros((num_samples, sequence_length, num_nodes, num_nodes))
    window_timestamps = []
    
    # Fill tensor with rolling windows
    for i, end_time in enumerate(valid_sampling_points):
        end_idx = timestamps.index(end_time)
        
        for j in range(sequence_length):
            matrix_time = timestamps[end_idx - (sequence_length - 1) + j]
            tensor[i, j] = matrices[matrix_time].values
        
        window_timestamps.append(end_time)
    
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

if __name__ == "__main__":
    # Set parameters
    tickers = {'AAPL', 'AMZN', 'TSLA'}
    start_date = '2024-11-28'
    end_date = '2024-11-29'
    matrix_lookback = '5min'
    sequence_length = 5  # Changed from tensor_lookback
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