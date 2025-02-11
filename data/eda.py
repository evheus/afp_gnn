from data_utils import load_data, perform_eda
import os


stocks = [
    # Consumer Discretionary - Large-Cap
    "AMZN", "TSLA", "HD", "MCD",
    # Consumer Discretionary - Mid-Cap
    "ROKU", "YUM", "DLTR", "BURL",
    # Consumer Discretionary - Small-Cap
    "CROX", "PLNT", "AEO", "PLAY",
    # Technology - Large-Cap
    "AAPL", "MSFT", "NVDA", "GOOGL",
    # Technology - Mid-Cap
    "PANW", "ZM", "DDOG", "NET",
    # Technology - Small-Cap
    "SMAR", "RPD", "ZS", "PATH"
]

# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# data_folder = os.path.join(project_root, "data", "ohlcv")

data_folder = "ohlcv"

for ticker in stocks:
    print(f"\nPerforming EDA for {ticker}...")
    try:
        df = load_data(ticker, folder=data_folder)
        perform_eda(df, ticker=ticker)
    except FileNotFoundError as e:
        print(f"Data for {ticker} not found: {e}")
    except Exception as e:
        print(f"An error occurred while processing {ticker}: {e}")
        
    input("\nPress Enter to continue to the next stock...")
