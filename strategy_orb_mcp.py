# strategy_orb_mcp.py
from alpaca.trading.client import TradingClient
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from datetime import datetime, timedelta
import pandas as pd
import time

class ORBStrategy:
    def __init__(self, api_key, secret_key):
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.historical_client = StockHistoricalDataClient(api_key, secret_key)
        self.orb_high = None
        self.orb_low = None
        self.breakout_direction = None
        self.entry_price = None
        self.position_size = None
        
    def calculate_opening_range(self, bars_df, start_idx):
        """Calculate 15-minute opening range using historical data"""
        # Get first 3 bars (15 minutes) of the day
        day_bars = bars_df.iloc[start_idx:start_idx + 3]
        if len(day_bars) < 3:
            return False
            
        self.orb_high = day_bars['high'].max()
        self.orb_low = day_bars['low'].min()
        return True

    def analyze_performance(self, trades):
        """Analyze strategy performance in detail"""
        if not trades:
            print("No trades to analyze")
            return

        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get('exit_price', 0) > 0 and 
                           ((t['direction'] == 'LONG' and t['exit_price'] > t['entry_price']) or
                            (t['direction'] == 'SHORT' and t['exit_price'] < t['entry_price'])))
        losing_trades = sum(1 for t in trades if t.get('exit_price', 0) > 0 and 
                          ((t['direction'] == 'LONG' and t['exit_price'] < t['entry_price']) or
                           (t['direction'] == 'SHORT' and t['exit_price'] > t['entry_price'])))
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate P&L metrics
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        avg_win = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0) / losing_trades if losing_trades > 0 else 0
        profit_factor = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0) / 
                          sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0)) if losing_trades > 0 else float('inf')
        
        # Calculate time-based metrics
        trade_durations = []
        for t in trades:
            if 'exit_time' in t and 'entry_time' in t:
                duration = t['exit_time'] - t['entry_time']
                trade_durations.append(duration.total_seconds() / 60)  # Convert to minutes
        
        avg_duration = sum(trade_durations) / len(trade_durations) if trade_durations else 0
        
        # Calculate drawdown metrics
        cumulative_pnl = 0
        peak = 0
        max_drawdown = 0
        for t in trades:
            pnl = t.get('pnl', 0)
            cumulative_pnl += pnl
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        # Print detailed analysis
        print("\nDetailed Performance Analysis:")
        print("=" * 50)
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Trade Duration: {avg_duration:.1f} minutes")
        print(f"Maximum Drawdown: ${max_drawdown:.2f}")
        
        # Analyze trade timing
        print("\nTrade Timing Analysis:")
        print("=" * 50)
        morning_trades = sum(1 for t in trades if t['entry_time'].hour < 12)
        afternoon_trades = sum(1 for t in trades if t['entry_time'].hour >= 12)
        print(f"Morning Trades (before 12:00): {morning_trades}")
        print(f"Afternoon Trades (after 12:00): {afternoon_trades}")
        
        # Analyze trade direction
        long_trades = sum(1 for t in trades if t['direction'] == 'LONG')
        short_trades = sum(1 for t in trades if t['direction'] == 'SHORT')
        print("\nTrade Direction Analysis:")
        print("=" * 50)
        print(f"Long Trades: {long_trades}")
        print(f"Short Trades: {short_trades}")
        
        # Analyze consecutive wins/losses
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        for t in trades:
            if t.get('exit_price', 0) > 0:
                is_win = ((t['direction'] == 'LONG' and t['exit_price'] > t['entry_price']) or
                         (t['direction'] == 'SHORT' and t['exit_price'] < t['entry_price']))
                if is_win:
                    if current_streak > 0:
                        current_streak += 1
                    else:
                        current_streak = 1
                else:
                    if current_streak < 0:
                        current_streak -= 1
                    else:
                        current_streak = -1
                max_win_streak = max(max_win_streak, current_streak)
                max_loss_streak = min(max_loss_streak, current_streak)
        
        print("\nStreak Analysis:")
        print("=" * 50)
        print(f"Longest Winning Streak: {max_win_streak}")
        print(f"Longest Losing Streak: {abs(max_loss_streak)}")

    def backtest(self, csv_file):
        """Backtest the strategy using downloaded CSV data"""
        # Read the CSV file
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Initialize results
        trades = []
        current_position = None
        pending_long = False
        pending_short = False
        long_breakout_price = None
        short_breakout_price = None
        
        # Process each day
        current_date = None
        for i in range(len(df)):
            bar = df.iloc[i]
            bar_date = bar['timestamp'].date()
            
            # New day
            if current_date != bar_date:
                current_date = bar_date
                # Calculate opening range for the new day
                if not self.calculate_opening_range(df, i):
                    continue
                current_position = None
                pending_long = False
                pending_short = False
                long_breakout_price = None
                short_breakout_price = None
            
            # Check for breakout and set pending flags
            if current_position is None:
                if not pending_long and bar['close'] > self.orb_high:
                    pending_long = True
                    long_breakout_price = self.orb_high
                if not pending_short and bar['close'] < self.orb_low:
                    pending_short = True
                    short_breakout_price = self.orb_low
            
            # Check for retest and enter trade
            if current_position is None:
                # Retest for LONG
                if pending_long and bar['low'] <= long_breakout_price:
                    trades.append({
                        'date': bar['timestamp'],
                        'direction': 'LONG',
                        'entry_price': bar['close'],
                        'entry_time': bar['timestamp']
                    })
                    current_position = 'LONG'
                    pending_long = False
                # Retest for SHORT
                elif pending_short and bar['high'] >= short_breakout_price:
                    trades.append({
                        'date': bar['timestamp'],
                        'direction': 'SHORT',
                        'entry_price': bar['close'],
                        'entry_time': bar['timestamp']
                    })
                    current_position = 'SHORT'
                    pending_short = False
            else:  # In position
                # Simple exit at end of day
                if bar['timestamp'].hour == 16 and bar['timestamp'].minute == 0:
                    trades[-1]['exit_price'] = bar['close']
                    trades[-1]['exit_time'] = bar['timestamp']
                    trades[-1]['pnl'] = (bar['close'] - trades[-1]['entry_price']) if current_position == 'LONG' else (trades[-1]['entry_price'] - bar['close'])
                    current_position = None
        
        # Add performance analysis
        self.analyze_performance(trades)
        
        return trades

# MCP Server Integration
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    strategy = ORBStrategy(
        os.getenv("ALPACA_PAPER_API_KEY"),
        os.getenv("ALPACA_PAPER_API_SECRET")
    )
    
    # Use the downloaded CSV file
    csv_file = "SPY_20250501_20250531.csv"  # Updated to use May 2025 data
    trades = strategy.backtest(csv_file)
    
    # Print results
    print("\nBacktest Results:")
    print(f"Total Trades: {len(trades)}")
    if trades:
        total_pnl = sum(trade['pnl'] for trade in trades if 'pnl' in trade)
        winning_trades = sum(1 for trade in trades if 'pnl' in trade and trade['pnl'] > 0)
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {(winning_trades/len(trades))*100:.1f}%")
        
        print("\nTrade Details:")
        for trade in trades:
            print(f"Date: {trade['date']}")
            print(f"Direction: {trade['direction']}")
            print(f"Entry Price: ${trade['entry_price']:.2f}")
            if 'exit_price' in trade:
                print(f"Exit Price: ${trade['exit_price']:.2f}")
                print(f"P&L: ${trade['pnl']:.2f}")
            print("---")
