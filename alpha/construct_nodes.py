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
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    window : str
        Rolling window size for volatility calculation
        
    Returns:
    --------
    pd.Series
        Rolling volatility values
    """
    # Calculate returns
    returns = data.pct_change()
    
    # Calculate rolling standard deviation (volatility)
    volatility = returns.rolling(window=window).std()
    
    return volatility

def create_feature_matrices(
    tickers: set,
    start_date: str,
    end_date: str,
    lookback: str = '30min',
    freq: str = '1min',
    data_folder: str = 'data/ohlcv'
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Create feature matrices for each timestamp.
    
    Parameters:
    -----------
    tickers : set
        Set of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    lookback : str
        Lookback window for volatility calculation
    freq : str
        Frequency for sampling
    data_folder : str
        Folder containing OHLCV data
        
    Returns:
    --------
    Dict[pd.Timestamp, pd.DataFrame]
        Dictionary of timestamp-indexed feature matrices
    """
    # Load data
    dfs = load_data_for_tickers(tickers, [start_date, end_date], data_folder)
    
    # Create multi-index DataFrame
    processed_data = pd.concat(dfs, axis=1)
    processed_data.columns = pd.MultiIndex.from_product([dfs.keys(), dfs[list(dfs.keys())[0]].columns])
    
    # Initialize dictionary for feature matrices
    feature_matrices = {}
    
    # Create calculation points
    calc_start = processed_data.index.min() + pd.Timedelta(lookback)
    calc_points = pd.date_range(start=calc_start,
                              end=processed_data.index.max(),
                              freq=freq)
    
    for ticker in tickers:
        # Calculate returns and volatility for each ticker
        price_data = processed_data[ticker]['close']
        processed_data[(ticker, 'returns')] = price_data.pct_change()
        processed_data[(ticker, 'volatility')] = calculate_volatility(price_data, lookback)
    
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

def create_rolling_windows(
    matrices: Dict[pd.Timestamp, pd.DataFrame],
    lookback: int,
    frequency: str = '1min'
) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    """
    Create rolling windows tensor from feature matrices.
    
    Parameters:
    -----------
    matrices : Dict[pd.Timestamp, pd.DataFrame]
        Dictionary of timestamp-indexed feature matrices
    lookback : int
        Number of historical matrices to include in each window
    frequency : str
        Frequency for sampling windows
        
    Returns:
    --------
    Tuple containing:
        - numpy.ndarray: 4D tensor of shape (num_windows, lookback, num_stocks, num_features)
        - List[pd.Timestamp]: Timestamps for each window
    """
    timestamps = sorted(matrices.keys())
    num_stocks = matrices[timestamps[0]].shape[0]
    num_features = matrices[timestamps[0]].shape[1]
    
    # Create frequency-based sampling points
    start_time = timestamps[lookback - 1]
    end_time = timestamps[-1]
    sampling_points = pd.date_range(start=start_time, end=end_time, freq=frequency)
    
    # Filter sampling points
    valid_sampling_points = [t for t in sampling_points if t in timestamps]
    
    # Initialize tensor
    num_windows = len(valid_sampling_points)
    tensor = np.zeros((num_windows, lookback, num_stocks, num_features))
    window_timestamps = []
    
    # Fill tensor with rolling windows
    for i, end_time in enumerate(valid_sampling_points):
        end_idx = timestamps.index(end_time)
        
        for j in range(lookback):
            matrix_time = timestamps[end_idx - (lookback - 1) + j]
            tensor[i, j] = matrices[matrix_time].values
        
        window_timestamps.append(end_time)
    
    return tensor, window_timestamps

def generate_node_embeddings(
    tickers: set,
    start_date: str,
    end_date: str,
    matrix_lookback: str = '30min',
    tensor_lookback: int = 5,
    freq: str = '1min',
    data_folder: str = 'data/ohlcv'
) -> Tuple[np.ndarray, List[pd.Timestamp], List[str]]:
    """
    Generate the final 4D node embeddings tensor from raw data.
    
    Parameters:
    -----------
    tickers : set
        Set of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    matrix_lookback : str
        Lookback window for volatility calculation
    tensor_lookback : int
        Number of historical matrices to include in each tensor window
    freq : str
        Frequency for sampling windows
    data_folder : str
        Folder containing OHLCV data
        
    Returns:
    --------
    Tuple containing:
        - numpy.ndarray: 4D tensor of shape (num_windows, lookback, num_stocks, num_features)
        - List[pd.Timestamp]: Timestamps for each window
        - List[str]: List of stock tickers in order
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
    
    # Create rolling windows tensor
    tensor, window_timestamps = create_rolling_windows(
        matrices,
        tensor_lookback,
        freq
    )
    
    # Get ordered list of tickers
    stock_universe = list(matrices[window_timestamps[0]].index)
    
    return tensor, window_timestamps, stock_universe

def display_tensor_info(
    tensor: np.ndarray,
    window_timestamps: List[pd.Timestamp],
    stock_universe: List[str]
) -> None:
    """Display information about the created tensor"""
    print(f"Tensor shape: {tensor.shape}")
    print(f"Interpretation:")
    print(f"- Number of windows: {tensor.shape[0]}")
    print(f"- Matrices per window (lookback): {tensor.shape[1]}")
    print(f"- Number of stocks: {tensor.shape[2]}")
    print(f"- Number of features: {tensor.shape[3]}")
    print(f"\nStock universe: {stock_universe}")
    print(f"\nFirst window ends at: {window_timestamps[0]}")
    print(f"Last window ends at: {window_timestamps[-1]}")
    
    # Show example window
    print(f"\nExample - First window feature matrices:")
    feature_names = ['returns', 'volume', 'volatility']
    for i in range(tensor.shape[1]):
        print(f"\nTime point {i+1} of {tensor.shape[1]}:")
        for j, feature in enumerate(feature_names):
            print(f"\n{feature.capitalize()} values:")
            df = pd.DataFrame(
                tensor[0, i, :, j],
                index=stock_universe,
                columns=[feature]
            )
            print(df.round(4))

if __name__ == "__main__":
    # Set parameters
    tickers = {'AMZN', 'AAPL', 'TSLA'}
    start_date = '2024-11-28'
    end_date = '2024-11-29'
    matrix_lookback = '5min'
    tensor_lookback = 5
    freq = '1min'
    data_folder = os.path.join(project_root, 'data', 'ohlcv')
    
    # Generate tensor
    tensor, window_timestamps, stock_universe = generate_node_embeddings(
        tickers,
        start_date,
        end_date,
        matrix_lookback,
        tensor_lookback,
        freq,
        data_folder
    )
    
    # Display information
    display_tensor_info(tensor, window_timestamps, stock_universe)
    print(tensor)