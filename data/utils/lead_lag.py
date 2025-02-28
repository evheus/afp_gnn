import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from collections import defaultdict
import time


def compute_c2(stock1_returns, stock2_returns, max_lag, weights=None):
    """
    Compute the C2 lead-lag score for two stocks using the weighted sum of lagged correlations.
    """
    if len(stock1_returns) != len(stock2_returns):
        raise ValueError("Both return series must have the same length.")

    if weights is None:
        weights = np.ones(2 * max_lag + 1) / (2 * max_lag + 1)
    elif len(weights) != 2 * max_lag + 1:
        raise ValueError("Weights must have a length of 2 * max_lag + 1.")

    lagged_correlations = []

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Stock 1 leads Stock 2
            corr = stock1_returns[:lag].corr(stock2_returns[-lag:])
        elif lag > 0:
            # Stock 2 leads Stock 1
            corr = stock1_returns[lag:].corr(stock2_returns[:-lag])
        else:
            # No lag
            corr = stock1_returns.corr(stock2_returns)

        lagged_correlations.append(corr)

    lagged_correlations = np.array(lagged_correlations)
    weighted_sum = np.dot(lagged_correlations, weights)

    return weighted_sum


# Cache dictionary
c1_cache = defaultdict(lambda: defaultdict(dict))

def compute_c1_cache(stock1_name, stock2_name, stock1_returns, stock2_returns, max_lag, validate=True):
    """
    Compute the C1 lead-lag score for two stocks using cached max-lagged Pearson correlation.
    Optionally validates results against the original function.
    """
    start = time.time()
    if len(stock1_returns) != len(stock2_returns):
        raise ValueError("Both return series must have the same length.")

    cache_key = tuple(sorted([stock1_name, stock2_name]))
    cache = c1_cache[cache_key]

    max_corr = -np.inf
    n = len(stock1_returns)

    for lag in range(-max_lag, max_lag + 1):
        lag_key = f"lag_{lag}"

        if lag_key in cache and len(cache[lag_key]["series1"]) == n:
            # Retrieve cached values
            prev_corr = cache[lag_key]["correlation"]
            prev_mean_s1, prev_mean_s2 = cache[lag_key]["mean_s1"], cache[lag_key]["mean_s2"]
            prev_std_s1, prev_std_s2 = cache[lag_key]["std_s1"], cache[lag_key]["std_s2"]
            prev_cov = cache[lag_key]["covariance"]
            prev_s1, prev_s2 = cache[lag_key]["series1"], cache[lag_key]["series2"]

            # Remove the oldest point and add the newest point
            if lag < 0:
                new_s1 = stock1_returns[:lag]
                new_s2 = stock2_returns[-lag:]
            elif lag > 0:
                new_s1 = stock1_returns[lag:]
                new_s2 = stock2_returns[:-lag]
            else:
                new_s1 = stock1_returns
                new_s2 = stock2_returns

            # Update correlation using cached stats
            new_corr, new_mean_s1, new_mean_s2, new_std_s1, new_std_s2, new_cov = update_correlation(
                prev_corr, prev_s1, prev_s2, new_s1, new_s2, prev_mean_s1, prev_mean_s2, prev_std_s1, prev_std_s2, prev_cov
            )
        else:
            # Compute from scratch
            if lag < 0:
                new_s1 = stock1_returns[:lag]
                new_s2 = stock2_returns[-lag:]
            elif lag > 0:
                new_s1 = stock1_returns[lag:]
                new_s2 = stock2_returns[:-lag]
            else:
                new_s1 = stock1_returns
                new_s2 = stock2_returns

            new_mean_s1, new_mean_s2 = new_s1.mean(), new_s2.mean()
            new_std_s1, new_std_s2 = new_s1.std(), new_s2.std()
            new_cov = ((new_s1 - new_mean_s1) * (new_s2 - new_mean_s2)).sum()

            if new_std_s1 > 0 and new_std_s2 > 0:
                new_corr = new_cov / ((len(new_s1) - 1) * new_std_s1 * new_std_s2)
            else:
                new_corr = 0  # Avoid division by zero

            # Store in cache
            cache[lag_key] = {
                "correlation": new_corr,
                "mean_s1": new_mean_s1,
                "mean_s2": new_mean_s2,
                "std_s1": new_std_s1,
                "std_s2": new_std_s2,
                "covariance": new_cov,
                "series1": new_s1.copy(),
                "series2": new_s2.copy()
            }

        if new_corr > max_corr:
            max_corr = new_corr

    print(f"Cached version took : {time.time() - start} sec")

    if validate:
        start = time.time()
        original_result = compute_c1(stock1_returns, stock2_returns, max_lag)
        print(f"Original version took {time.time() - start} sec")
        if not np.isclose(max_corr, original_result, atol=1e-6):
            print(f"Validation failed for {stock1_name}-{stock2_name} at max_lag={max_lag}")
            print(f"Cached Result: {max_corr}, Original Result: {original_result}")
        else:
            print(f"Validation passed for {stock1_name}-{stock2_name} at max_lag={max_lag}")

    return max_corr

def update_correlation(prev_corr, prev_s1, prev_s2, new_s1, new_s2, mean_s1, mean_s2, std_s1, std_s2, prev_cov):
    """
    Efficiently updates Pearson correlation by removing the oldest value and adding the newest.
    """
    if len(prev_s1) != len(new_s1) or len(prev_s2) != len(new_s2):
        return new_s1.corr(new_s2), new_s1.mean(), new_s2.mean(), new_s1.std(), new_s2.std(), ((new_s1 - new_s1.mean()) * (new_s2 - new_s2.mean())).sum()

    old_value_s1, old_value_s2 = prev_s1.iloc[0], prev_s2.iloc[0]
    new_value_s1, new_value_s2 = new_s1.iloc[-1], new_s2.iloc[-1]

    # Update rolling means
    new_mean_s1 = mean_s1 + (new_value_s1 - old_value_s1) / len(prev_s1)
    new_mean_s2 = mean_s2 + (new_value_s2 - old_value_s2) / len(prev_s2)

    # Update rolling covariance
    new_cov = prev_cov - ((old_value_s1 - mean_s1) * (old_value_s2 - mean_s2)) + ((new_value_s1 - new_mean_s1) * (new_value_s2 - new_mean_s2))

    # Update rolling standard deviations
    new_std_s1 = np.sqrt(((std_s1**2 * len(prev_s1)) - (old_value_s1 - mean_s1)**2 + (new_value_s1 - new_mean_s1)**2) / len(prev_s1))
    new_std_s2 = np.sqrt(((std_s2**2 * len(prev_s2)) - (old_value_s2 - mean_s2)**2 + (new_value_s2 - new_mean_s2)**2) / len(prev_s2))

    if new_std_s1 > 0 and new_std_s2 > 0:
        new_corr = new_cov / ((len(prev_s1) - 1) * new_std_s1 * new_std_s2)
    else:
        new_corr = 0  # Avoid division by zero

    return new_corr, new_mean_s1, new_mean_s2, new_std_s1, new_std_s2, new_cov

# Original function for validation
def compute_c1(stock1_returns, stock2_returns, max_lag):
    """
    Compute the C1 lead-lag score for two stocks using the maximum lagged Pearson correlation.
    """
    if len(stock1_returns) != len(stock2_returns):
        raise ValueError("Both return series must have the same length.")
    max_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            shifted_corr = stock1_returns[:lag].corr(stock2_returns[-lag:])
        elif lag > 0:
            shifted_corr = stock1_returns[lag:].corr(stock2_returns[:-lag])
        else:
            shifted_corr = stock1_returns.corr(stock2_returns)

        if shifted_corr > max_corr:
            max_corr = shifted_corr

    return max(max_corr, 0)

def compute_direction_score(stock1_returns, stock2_returns, max_lag, weighted=True, handle_negative="discard"):
    """
    Compute the Direction Score metric for two stocks.
    
    Parameters:
    -----------
    stock1_returns : pandas.Series
        Returns for the first stock
    stock2_returns : pandas.Series
        Returns for the second stock
    max_lag : int
        Maximum lag to consider in both directions
    weighted : bool, default=False
        Whether to apply time weights (1/(|l|+1)) to the correlations
    handle_negative : str, default="discard"
        How to handle negative correlations:
        - "discard": Treat negative correlations as zero
        - "penalize": Include negative correlations but ensure sums remain nonnegative
    
    Returns:
    --------
    float
        The Direction Score indicating the lead-lag relationship
    """
    if len(stock1_returns) != len(stock2_returns):
        raise ValueError("Both return series must have the same length.")
    
    # Calculate correlations for all lags
    correlations = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            continue  # Skip lag 0 as it's not used in the formula
        
        if lag < 0:
            # Y leads X
            shifted_corr = stock1_returns[:lag].corr(stock2_returns[-lag:])
        else:  # lag > 0
            # X leads Y
            shifted_corr = stock1_returns[lag:].corr(stock2_returns[:-lag])
            
        correlations[lag] = shifted_corr
    
    # Process positive lags (X leads Y)
    positive_sum = 0
    for lag in range(1, max_lag + 1):
        corr = correlations[lag]
        
        if handle_negative == "discard":
            corr = max(corr, 0)
        
        if weighted:
            corr *= 1 / (lag + 1)
            
        positive_sum += corr
    
    # Process negative lags (Y leads X)
    negative_sum = 0
    for lag in range(-max_lag, 0):
        corr = correlations[lag]
        
        if handle_negative == "discard":
            corr = max(corr, 0)
            
        if weighted:
            corr *= 1 / (abs(lag) + 1)
            
        negative_sum += corr
    
    # Apply different handling for negative correlations
    if handle_negative == "penalize":
        positive_sum = max(positive_sum, 0)
        negative_sum = max(negative_sum, 0)
    
    # Calculate the direction score
    direction_score = positive_sum - negative_sum
    
    return direction_score


def compute_levy_area(stock1_returns, stock2_returns):
    """
    Compute the Levy score for two stocks to capture nonlinear dependencies.
    """

    if len(stock1_returns) != len(stock2_returns):
        raise ValueError("Both return series must have the same length.")
    
    stock1_returns = np.asarray(stock1_returns)
    stock2_returns = np.asarray(stock2_returns)

    path1 = np.cumsum(stock1_returns)
    path2 = np.cumsum(stock2_returns)

    dx = np.diff(path1)
    dy = np.diff(path2)

    levy_area = 0.5 * np.sum(dx[:-1] * dy[1:] - dx[1:] * dy[:-1])
    
    return -levy_area


def construct_ols_lead_lag_matrix(df, max_lag=5, alpha=0.01):
    """
    Construct a lead-lag matrix using regression.
    """
    stocks = df.columns.get_level_values(0).unique()
    n_stocks = len(stocks)
    lead_lag_matrix = np.zeros((n_stocks, n_stocks))

    for i, target_stock in enumerate(stocks):
        # Target: next-minute return of the current stock
        y = df[target_stock]['log_return'].shift(-1).iloc[max_lag:-1].dropna()

        # Predictor: past returns of all other stocks
        X = []
        for lag in range(1, max_lag + 1):
            lagged_returns = df.loc[:, pd.IndexSlice[:, 'log_return']].shift(lag).iloc[max_lag:-1]
            X.append(lagged_returns)

        X = pd.concat(X, axis=1).dropna()
        y = y.loc[X.index]

        # Exclude the target stock in predictors
        predictors = [col for col in X.columns if col[0] != target_stock]
        X = X[predictors]

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Add constant for OLS
        X_scaled = sm.add_constant(X_scaled)

        # OLS
        model = sm.OLS(y, X_scaled).fit()
        
        # Get betas
        for j, other_stock in enumerate(stocks):
            if other_stock != target_stock:
                # Sum of all betas
                coefficients = [
                    model.params.iloc[k+1] for k, col in enumerate(predictors) if col[0] == other_stock
                ]
                lead_lag_matrix[i, j] = np.sum(coefficients)

    # Ensure skew-symmetry
    for i in range(n_stocks):
        for j in range(i + 1, n_stocks):
            avg_value = (lead_lag_matrix[i, j] - lead_lag_matrix[j, i]) / 2
            lead_lag_matrix[i, j] = avg_value
            lead_lag_matrix[j, i] = -avg_value

    lead_lag_df = pd.DataFrame(lead_lag_matrix, index=stocks, columns=stocks)
    return lead_lag_df


def construct_lead_lag_matrix(df, method, max_lag=5, weights=None, alpha=0.01):
    """
    Construct a skew-symmetric lead-lag matrix for stock returns based on the method.
    """

    window_size = df.index.nunique()  # Gets number of unique timestamps
    
    # Add validation for max_lag
    max_possible_lag = (window_size // 2) - 1  # Ensure at least 2 points for correlation
    max_lag = min(max_lag, max_possible_lag)
    
    if max_lag < 1:
        raise ValueError(f"Window size {window_size} too small for lead-lag calculation. Need at least 4 points.")

    stocks = df.columns.get_level_values(0).unique()
    n_stocks = len(stocks)
    lead_lag_matrix = np.zeros((n_stocks, n_stocks))

    if method == 'C1':
        #compute_score = lambda stock1_name, stock2_name, stock1_returns, stock2_returns: compute_c1_cache(
        #    stock1_name, stock2_name, stock1_returns, stock2_returns, max_lag
        #)
        compute_score = lambda stock1, stock2: compute_c1(stock1_returns, stock2_returns, max_lag)
    elif method == 'C2':
        compute_score = lambda stock1, stock2: compute_c2(stock1, stock2, max_lag, weights)
    elif method == 'Levy':
        compute_score = lambda stock1, stock2: compute_levy_area(stock1, stock2)
    elif method == 'Linear':
        return construct_ols_lead_lag_matrix(df, max_lag, alpha)
    elif method == 'Direction':
        compute_score = lambda stock1, stock2: compute_direction_score(stock1, stock2, max_lag)
    else:
        raise ValueError("Invalid method. Choose from 'C1', 'C2', 'Levy', 'Direction' or 'Linear'.")

    # Iterate over stock pairs
    for i, stock1 in enumerate(stocks):
        stock1_returns = df[stock1]['normalized_return']
        for j, stock2 in enumerate(stocks):
            if i != j:
                stock2_returns = df[stock2]['normalized_return']
                # if method == 'C1':
                #    score = compute_score(stock1, stock2, stock1_returns, stock2_returns)
                # else:
                score = compute_score(stock1_returns, stock2_returns)

                lead_lag_matrix[i, j] = score
                lead_lag_matrix[j, i] = -score

    lead_lag_df = pd.DataFrame(lead_lag_matrix, index=stocks, columns=stocks)
    return lead_lag_df



def rank_stocks(window_data, method, max_lag):
    """
    Rank stocks based on the average of their column values in the lead-lag matrix.
    If the method requires a lead-lag matrix, it computes it first.
    """
    if method in ['C1', 'C2', 'Levy', 'Direction' 'Linear']:
        # Compute lead-lag matrix 
        lead_lag_matrix = construct_lead_lag_matrix(window_data, method, max_lag)
        scores = lead_lag_matrix.mean(axis=0)
    else:
        # Space for alternative ranking method
        scores = window_data.loc[:, pd.IndexSlice[:, 'log_return']].mean(axis=0)

    ranked_stocks = pd.DataFrame({'Stock': scores.index, 'Score': scores.values})
    ranked_stocks = ranked_stocks.sort_values(by='Score', ascending=False).reset_index(drop=True)

    return ranked_stocks

