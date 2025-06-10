from alpaca.data import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import os

def verify_symbol(trading_client, symbol):
    """Verify if the symbol is valid and tradable"""
    try:
        asset = trading_client.get_asset(symbol)
        print(f"Symbol {symbol} is valid:")
        print(f"Name: {asset.name}")
        print(f"Exchange: {asset.exchange}")
        print(f"Status: {asset.status}")
        print(f"Tradable: {asset.tradable}")
        return True
    except Exception as e:
        print(f"Error verifying symbol {symbol}: {str(e)}")
        return False

def download_historical_data(symbol, start_date, end_date, timeframe='5Min'):
    """Download historical data and save to CSV"""
    # Load environment variables
    load_dotenv()
    
    # Initialize the clients
    trading_client = TradingClient(
        os.getenv("ALPACA_PAPER_API_KEY"),
        os.getenv("ALPACA_PAPER_API_SECRET"),
        paper=True
    )
    
    historical_client = StockHistoricalDataClient(
        os.getenv("ALPACA_PAPER_API_KEY"),
        os.getenv("ALPACA_PAPER_API_SECRET")
    )
    
    # First verify the symbol
    if not verify_symbol(trading_client, symbol):
        return None
    
    # Create request parameters
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],  # Note: API expects a list
        timeframe=TimeFrame.Minute,
        timeframe_value=5,
        start=start_date,
        end=end_date,
        feed='iex'
    )
    
    print(f"\nDownloading historical data for {symbol}")
    print(f"Start: {start_date}")
    print(f"End: {end_date}")
    print(f"Timeframe: {timeframe}")
    
    try:
        # Get the bars
        print("Requesting data from Alpaca...")
        bars = historical_client.get_stock_bars(request_params)
        
        print(f"Response type: {type(bars)}")
        
        # Convert BarSet to list of bars
        bars_list = list(bars)
        if not bars_list:
            print("No bars returned in response")
            return None
            
        print(f"Received {len(bars_list)} bars")
        
        # Debug the first few bar structures
        for i, bar in enumerate(bars_list[:3]):
            print(f"\nBAR {i}:")
            print(f"  type(bar): {type(bar)}")
            print(f"  bar[0]: {bar[0]} (type: {type(bar[0])})")
            print(f"  bar[1]: {bar[1]} (type: {type(bar[1])})")
            if isinstance(bar[1], dict):
                print(f"  bar[1].keys(): {list(bar[1].keys())}")
        
        # Flatten all bars for the symbol into a single list
        all_bars = []
        for bar in bars_list:
            if isinstance(bar[1], dict):
                for symbol, bar_list in bar[1].items():
                    all_bars.extend(bar_list)

        # Convert to pandas DataFrame
        df = pd.DataFrame([{
            'timestamp': b.timestamp,
            'open': float(b.open),
            'high': float(b.high),
            'low': float(b.low),
            'close': float(b.close),
            'volume': float(b.volume)
        } for b in all_bars])
        
        # Save to CSV
        filename = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    # Test with a shorter time period first
    symbol = "SPY"
    # Set the date range for May 2025
    start_date = datetime(2025, 5, 1)
    end_date = datetime(2025, 5, 31)
    
    print("Testing with historical data...")
    print(f"Using date range: {start_date} to {end_date}")
    df = download_historical_data(symbol, start_date, end_date)
    
    if df is not None:
        print("\nFirst few rows of data:")
        print(df.head())
        print(f"\nTotal bars: {len(df)}")
        
        # Print time range of data
        print("\nData time range:")
        print(f"Start: {df['timestamp'].min()}")
        print(f"End: {df['timestamp'].max()}") 