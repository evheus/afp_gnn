import pandas as pd
import numpy as np
import os
import sys

project_root = os.getcwd()
sys.path.append(project_root)

from data.data_utils import load_data_for_tickers, preprocess_data
from lead_lag import construct_lead_lag_matrix
from datetime import datetime, timedelta

data_folder = os.path.join(project_root, 'data', 'ohlcv')

def process_and_generate_leadlag(tickers, start_date, end_date, lookback='30min', freq='1min', data_folder=data_folder, max_lag=5):
    """
    Process OHLCV data and generate lead-lag matrices using rolling windows.
    
    Parameters:
    -----------
    tickers : set
        Set of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    lookback : str
        Size of the rolling window used for lead-lag calculation (default: '30min')
    freq : str
        Frequency at which to calculate new matrices (default: '1min')
    data_folder : str
        Folder containing the OHLCV parquet files
    max_lag : int
        Maximum lag for lead-lag calculation
        
    Returns:
    --------
    tuple
        (processed_data, lead_lag_matrices)
    """
    # Load and preprocess data
    dfs = load_data_for_tickers(tickers, [start_date, end_date], data_folder)
    
    # Create multi-index DataFrame
    processed_data = pd.concat(dfs, axis=1)
    processed_data.columns = pd.MultiIndex.from_product([dfs.keys(), dfs[list(dfs.keys())[0]].columns])
    
    # Generate lead-lag matrices for each window
    lead_lag_matrices = {}
    
    # Create calculation points based on frequency
    # Add lookback to start_date to ensure we have enough data for first calculation
    calc_start = processed_data.index.min() + pd.Timedelta(lookback)
    calc_points = pd.date_range(start=calc_start,
                              end=processed_data.index.max(),
                              freq=freq)
    
    for current_time in calc_points:
        # Define the window of data to use for calculation
        window_start = current_time - pd.Timedelta(lookback)
        window_data = processed_data.loc[window_start:current_time]
        
        if len(window_data) > 0:
            # Calculate lead-lag matrices using different methods
            for method in ['C1']:#, 'C2', 'Levy', 'Linear']:
                try:
                    lead_lag_matrix = construct_lead_lag_matrix(
                        window_data,
                        method=method,
                        max_lag=max_lag
                    )
                    
                    if method not in lead_lag_matrices:
                        lead_lag_matrices[method] = {}
                    
                    # Store matrix with the current_time as key (end of window)
                    lead_lag_matrices[method][current_time] = lead_lag_matrix
                    
                except Exception as e:
                    print(f"Error calculating {method} lead-lag matrix for time {current_time}: {str(e)}")
    
    return processed_data, lead_lag_matrices

def save_leadlag_matrices(lead_lag_matrices, output_file):
    """
    Save lead-lag matrices to a parquet file.
    
    Parameters:
    -----------
    lead_lag_matrices : dict
        Dictionary of lead-lag matrices by method and timestamp
    output_file : str
        Output parquet file path
    """
    # Convert the nested dictionary to a DataFrame
    data = []
    for method in lead_lag_matrices:
        for timestamp, matrix in lead_lag_matrices[method].items():
            # Flatten the matrix and add metadata
            for i in matrix.index:
                for j in matrix.columns:
                    data.append({
                        'timestamp': timestamp,
                        'method': method,
                        'stock1': i,
                        'stock2': j,
                        'value': matrix.loc[i, j]
                    })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to parquet
    df.to_parquet(output_file)

# Example usage
if __name__ == "__main__":
    # Set parameters
    tickers = {'AMZN', 'AAPL', 'TSLA'}
    start_date = '2024-11-28'
    end_date = '2024-11-29'
    lookback = '5min'
    freq = '1min'
    data_folder = data_folder
    output_file = "lead_lag_matrices.parquet"
    
    # Process data and generate lead-lag matrices
    processed_data, lead_lag_matrices = process_and_generate_leadlag(
        tickers,
        start_date,
        end_date,
        lookback,
        freq,
        data_folder
    )
    
    # Save lead-lag matrices
    save_leadlag_matrices(lead_lag_matrices, output_file)
    
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Lead-lag matrices saved to: {output_file}")