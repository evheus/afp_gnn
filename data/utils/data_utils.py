import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pyarrow
import fastparquet

def load_data(ticker, folder="ohlcv"):
    """
    Load the data file for a given ticker.
    """
    file_path = os.path.join(folder, f"{ticker}_ohlcv.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for {ticker} not found in {folder}.")

    data = pd.read_parquet(file_path)
    if 'timestamp' in data.columns:
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
        except Exception as e:
            raise ValueError(f"Failed to restore DatetimeIndex for {ticker}: {e}")
    else:
        raise ValueError(f"No timestamp column found in the data for {ticker}.")

    return data

def load_and_preprocess_spy(data_folder="ohlcv"):
    """Load and preprocess SPY data."""
    spy = load_data("SPY", folder=data_folder)
    spy = preprocess_data(spy)
    return spy

def forward_fill_within_day(day_df):
    """
    Forward fill for a given trading day
    """
    # Index for each minute of the trading day
    day_index = pd.date_range(
        start=day_df.index.min().replace(hour=9, minute=30),
        end=day_df.index.min().replace(hour=16, minute=0),
        freq='min', tz='US/Eastern'
    )
    day_df = day_df.reindex(day_index)

    # If the first entry of the day is missing, backfill with the next available value
    if pd.Timestamp('09:30', tz='US/Eastern') in day_index and day_df.loc[pd.Timestamp('09:30', tz='US/Eastern')].isnull().all():
        next_available_value = day_df['close'].bfill().iloc[0]
        day_df.loc[pd.Timestamp('09:30', tz='US/Eastern'), 'close'] = next_available_value

    day_df.ffill(inplace=True)

    return day_df


def preprocess_data(df):
    """
    Preprocess the input dataframe by:
    - UTC to EST.
    - Only trading hours (excluding open and close).
    - Forward-filling missing minutes.
    - Computing log returns based on the close price.
    - Normalizing log returns using a rolling window.
    """
    if 'close' not in df.columns:
        raise ValueError("The dataframe must contain a 'close' column.")

    # Convert timestamps from UTC to EST
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('US/Eastern')
    else:
        df.index = df.index.tz_convert('US/Eastern')

    # Forward-fill missing minutes within each day
    df['date'] = df.index.date
    df['time'] = df.index.time

    # Apply forward fill within each day
    df = df.groupby('date').apply(forward_fill_within_day)

    # remove multi index introduced in groupby
    df.index = df.index.get_level_values(-1)

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))   # return over the previous minute

    # removing open and close (15 min buffer)
    df = df.between_time('09:0', '16:00')

    # removing half day
    #df = df[~((df.index.month == 11) & (df.index.day == 29))]

    rolling_mean = df['log_return'].rolling(window=60, min_periods=1).mean()
    rolling_std = df['log_return'].rolling(window=60, min_periods=1).std()

    # Normalize
    df['normalized_return'] = (df['log_return'] - rolling_mean) / rolling_std

    df.dropna(subset=['log_return', 'normalized_return'], inplace=True)

    return df


def load_data_for_tickers(tickers, date_range, data_folder="ohlcv"):
    # Assume date_range items are naive and represent UTC times,
    # then convert them to US Eastern.
    start_date = pd.Timestamp(date_range[0], tz='UTC').tz_convert('US/Eastern')
    end_date = pd.Timestamp(date_range[1], tz='UTC').tz_convert('US/Eastern')
    
    dfs = {}
    for ticker in tickers:
        print(f"Loading data for {ticker}...")
        df = load_data(ticker, folder=data_folder)
        df = preprocess_data(df)
        # Ensure that the DataFrame index is in US Eastern time.
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('US/Eastern')
        else:
            df.index = df.index.tz_convert('US/Eastern')
        df = df[(df.index.date >= start_date.date()) & (df.index.date <= end_date.date())]
        dfs[ticker] = df

    spy = load_and_preprocess_spy(data_folder)
    spy = spy[(spy.index.date >= start_date.date()) & (spy.index.date <= end_date.date())]
    dfs['SPY'] = spy

    return dfs


def perform_eda(df, ticker=None, show_plots=False):
    """
    Perform exploratory data analysis on a DataFrame.
    """
    if not isinstance(df.index, pd.DatetimeIndex):  
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(f"Failed to convert index to DatetimeIndex for {ticker}: {e}")


    # overall
    print("Structure and Columns:")
    print(df.info())

    # summary
    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing or Bad Values:")
    missing = df.isnull().sum()
    print(missing)

    # Check for duplicate minutes
    print("\nChecking for Duplicate Minutes...")
    df['minute'] = df.index.floor('min')  # Floor to the nearest minute
    duplicate_minutes = df['minute'].duplicated(keep=False)
    if duplicate_minutes.any():
        print(f"Duplicate minutes found: {duplicate_minutes.sum()} occurrences")
        print(df.loc[duplicate_minutes, ['minute']])
    else:
        print("No duplicate minutes found.")

    # price ranges
    invalid_rows = df[(df['open'] < df['low']) | 
                      (df['open'] > df['high']) | 
                      (df['close'] < df['low']) | 
                      (df['close'] > df['high'])]
    print("\nRows with logical range issues:")
    print(invalid_rows)

    if show_plots:
        # boxplots
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df['volume'])
        plt.title(f"Boxplot of volume ({ticker if ticker else ''})")
        plt.show()

        price_metrics = ['open', 'high', 'low', 'close']
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[price_metrics])
        plt.title(f"Boxplots for Price Metrics ({ticker if ticker else ''})")
        plt.xlabel("Price Metric")
        plt.ylabel("Value")
        plt.show()

    # number of entries
    df['hour'] = df.index.hour
    df['date'] = df.index.date
    hourly_counts = df.groupby(['date', 'hour']).size()

    trading_hours = range(14, 21)
    expected_entries_per_hour = 60
    missing_hours = []

    for (date, hour), count in hourly_counts.items():
        if hour in trading_hours and count != expected_entries_per_hour:
            missing_hours.append((date, hour, count))

    print("\nMissing Entries per Hour (Trading Hours Only):")
    if missing_hours:
        for date, hour, count in missing_hours:
            print(f"Date: {date}, Hour: {hour}, Entries: {count}")
    else:
        print("All trading hours have the expected number of entries.")

    # Check daily counts
    daily_counts = df.groupby('date').size()
    full_trading_day_minutes = 390  # 6.5 hours * 60 minutes
    incomplete_days = daily_counts[daily_counts < full_trading_day_minutes]

    print("\nEntries per Day:")
    if not incomplete_days.empty:
        for date, count in incomplete_days.items():
            day_data = df[df['date'] == date]
            earliest_entry = day_data.index.min()
            latest_entry = day_data.index.max()
            print(f"Date: {date}, Entries: {count}, Earliest: {earliest_entry}, Latest: {latest_entry}")

        # Analyze consecutive missing minutes for each incomplete day
        print("\nConsecutive Missing Minutes for Each Incomplete Day:")
        for date in incomplete_days.index:
            day_data = df[df['date'] == date].reindex(
                pd.date_range(start=date, end=date, freq='min')
            )
            missing_streaks = day_data.isnull().all(axis=1).astype(int).groupby(
                (day_data.isnull().all(axis=1) != day_data.isnull().all(axis=1).shift()).cumsum()
            ).sum()
            missing_streaks = missing_streaks[missing_streaks > 0]
            for streak, count in missing_streaks.items():
                if count > 1:
                    print(f"  Streak: {count} consecutive minutes missing for {date}")
    else:
        print("All days have the expected number of entries.")
    
    # Ensure all trading days are present
    print("\nChecking for Missing and Additional Trading Days...")
    date_range = pd.date_range(start="2024-09-01", end="2024-11-30", freq='B')
    dataset_days = df['date'].unique()

    # Missing trading days
    missing_days = np.setdiff1d(date_range.date, dataset_days)
    if len(missing_days) > 0:
        print("\nMissing Trading Days:")
        for day in missing_days:
            print(day)
    else:
        print("No missing trading days from September to November 2024.")

    # Additional days
    additional_days = np.setdiff1d(dataset_days, date_range.date)
    if len(additional_days) > 0:
        print("\nAdditional Days in the Dataset:")
        for day in additional_days:
            print(day)
    else:
        print("No additional days in the dataset.")

    # outlier price detection
    print("\nOutlier Price Detection (MA):")
    for col in ['open', 'high', 'low', 'close']:
        df[f'{col}_ma5'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{col}_diff'] = abs(df[col] - df[f'{col}_ma5'])
        # threshold as 5x the rolling standard deviation
        threshold = 5 * df[col].rolling(window=5, min_periods=1).std()
        outliers = df[df[f'{col}_diff'] > threshold]
        
        print(f"\nOutliers detected for {col}:")
        if not outliers.empty:
            print(outliers[[col, f'{col}_ma5', f'{col}_diff']])
        else:
            print(f"No significant outliers detected for {col}.")

    df.drop(columns=[f'{col}_ma5' for col in ['open', 'high', 'low', 'close']] + 
                    [f'{col}_diff' for col in ['open', 'high', 'low', 'close']] + ['minute'], inplace=True)

def load_and_preprocess_all_data(tickers: list, date_range: list, data_folder="ohlcv") -> pd.DataFrame:
    """
    Load and preprocess data for all tickers once and return a concatenated
    multi-index DataFrame within the specified date range.
    """
    dfs = {}
    for ticker in tickers:
        print(f"Loading data for {ticker}...")
        df = load_data(ticker, folder=data_folder)
        df = preprocess_data(df)
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('US/Eastern')
        else:
            df.index = df.index.tz_convert('US/Eastern')
        dfs[ticker] = df
    
    processed_data = pd.concat(dfs, axis=1)
    processed_data.columns = pd.MultiIndex.from_product([list(dfs.keys()), dfs[list(dfs.keys())[0]].columns])
    
    # Filter data within date range (inclusive) assuming date_range is in Eastern time
    start_date = pd.Timestamp(date_range[0], tz='US/Eastern')
    end_date = pd.Timestamp(date_range[1], tz='US/Eastern')
    processed_data = processed_data.loc[start_date:end_date + pd.Timedelta(days=1)]
    
    return processed_data