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
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from alpaca_trade_api import REST
from matplotlib.gridspec import GridSpec
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ORBStrategy:
    def __init__(self, data, entry_mode='immediate', risk_per_trade=0.02, 
                 min_stop_distance=0.005, profit_target_multiplier=1.5,
                 orb_duration=30, use_volume_filter=True,
                 trailing_stop=True, trailing_stop_activation=0.4,
                 use_volatility_filter=True, scale_out_levels=[0.5, 0.75, 1.0],
                 ticker=None):
        self.data = data
        self.entry_mode = entry_mode
        self.risk_per_trade = risk_per_trade
        self.min_stop_distance = min_stop_distance
        self.profit_target_multiplier = profit_target_multiplier
        self.orb_duration = orb_duration
        self.use_volume_filter = use_volume_filter
        self.trailing_stop = trailing_stop
        self.trailing_stop_activation = trailing_stop_activation
        self.use_volatility_filter = use_volatility_filter
        self.scale_out_levels = scale_out_levels
        self.ticker = ticker
        self.trades = []
        self.orb_cache = {}
        
        # Calculate additional indicators
        if self.use_volume_filter:
            self.data['volume_sma'] = self.data['volume'].rolling(window=20).mean()
            self.data['volume_trend'] = self.data['volume'].rolling(window=20).mean() > self.data['volume'].rolling(window=40).mean()
            self.data['volume_momentum'] = self.data['volume'] / self.data['volume'].rolling(window=20).mean()
        
        # Calculate volatility metrics
        self.data['atr'] = self.calculate_atr(14)
        self.data['volatility'] = self.data['atr'] / self.data['close']
        self.data['volatility_sma'] = self.data['volatility'].rolling(window=20).mean()
        
        # Calculate volatility thresholds
        self.data['volatility_ratio'] = self.data['volatility'] / self.data['volatility_sma']
        self.data['is_volatile'] = self.data['volatility_ratio'] > 1.5  # 50% above average volatility
        
        # Calculate additional indicators for entry conditions
        self.data['sma_10'] = self.data['close'].rolling(window=10).mean()
        self.data['sma_20'] = self.data['close'].rolling(window=20).mean()
        self.data['rsi'] = self.calculate_rsi(14)

    def calculate_atr(self, period):
        """Calculate Average True Range for volatility measurement"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr

    def calculate_dynamic_trailing_stop(self, current_price, current_volatility):
        """Calculate dynamic trailing stop activation based on volatility"""
        base_activation = self.trailing_stop_activation
        volatility_factor = current_volatility * 100  # Scale volatility to percentage
        
        # Adjust activation based on volatility
        if volatility_factor > 2.0:  # High volatility
            return base_activation * 0.7  # More aggressive trailing stop
        elif volatility_factor < 0.5:  # Low volatility
            return base_activation * 1.3  # Less aggressive trailing stop
        return base_activation

    def calculate_orb(self, date):
        """Calculate Opening Range Breakout levels with caching"""
        if date in self.orb_cache:
            return self.orb_cache[date]
            
        # Get first N minutes of data for the given date
        date_data = self.data[self.data.index.date == date].iloc[:self.orb_duration]
        
        if len(date_data) < 3:  # Need at least 3 bars for valid ORB
            return None, None, None
            
        orb_high = date_data['high'].max()
        orb_low = date_data['low'].min()
        
        # Calculate average volume for the opening range
        avg_volume = date_data['volume'].mean() if self.use_volume_filter else None
        
        self.orb_cache[date] = (orb_high, orb_low, avg_volume)
        return orb_high, orb_low, avg_volume

    def check_volatility_conditions(self, current_volatility, current_volatility_ratio, ticker):
        """Check if volatility conditions are suitable for trading"""
        if not self.use_volatility_filter:
            return True
            
        # Different volatility thresholds for different tickers
        if ticker == 'SMH':
            # More conservative for SMH
            return current_volatility_ratio < 1.3  # Only trade in lower volatility
        else:
            # Standard threshold for other tickers
            return current_volatility_ratio < 1.5

    def calculate_scale_out_levels(self, entry_price, stop_loss, direction):
        """Calculate price levels for scaling out of positions"""
        if not self.scale_out_levels:
            return None
        
        risk = abs(entry_price - stop_loss)
        if direction == 'LONG':
            return [
                entry_price + (risk * self.profit_target_multiplier * level)
                for level in self.scale_out_levels
            ]
        else:
            return [
                entry_price - (risk * self.profit_target_multiplier * level)
                for level in self.scale_out_levels
            ]

    def update_position_size(self, current_trade, current_price):
        """Update position size based on scale-out levels"""
        if not current_trade.get('scale_out_levels'):
            return current_trade['position_size']
            
        remaining_size = current_trade['position_size']
        for level in current_trade['scale_out_levels']:
            if current_trade['direction'] == 'LONG' and current_price >= level:
                remaining_size = int(remaining_size * 0.5)  # Reduce by 50% at each level
            elif current_trade['direction'] == 'SHORT' and current_price <= level:
                remaining_size = int(remaining_size * 0.5)
                
        return remaining_size

    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index"""
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def check_volume_conditions(self, current_volume, avg_volume, i):
        """Enhanced volume analysis with trend and momentum"""
        if i < 20:  # Need enough data for calculations
            return True
        
        # Calculate volume trend
        volume_trend = self.data['volume_trend'].iloc[i]
        
        # Calculate volume momentum
        volume_momentum = self.data['volume_momentum'].iloc[i] > 1.2
        
        # Check for increasing volume
        volume_increasing = all(self.data['volume'].iloc[i-j] < current_volume 
                              for j in range(1, 4))
        
        # Calculate recent volume average
        recent_volume_avg = self.data['volume'].iloc[i-20:i].mean()
        volume_above_recent = current_volume > recent_volume_avg
        
        return (volume_trend and volume_momentum and 
                volume_increasing and volume_above_recent)

    def calculate_dynamic_profit_target(self, entry_price, stop_loss, current_volatility):
        """Calculate profit target based on volatility and market conditions"""
        base_risk = abs(entry_price - stop_loss)
        
        # Calculate volatility-based multiplier
        volatility_multiplier = 1.0
        if current_volatility < 0.01:  # Low volatility
            volatility_multiplier = 1.5
        elif current_volatility < 0.02:  # Medium volatility
            volatility_multiplier = 2.0
        else:  # High volatility
            volatility_multiplier = 2.5
        
        # Calculate market trend
        if len(self.data) >= 20:
            short_ma = self.data['close'].rolling(window=10).mean().iloc[-1]
            long_ma = self.data['close'].rolling(window=20).mean().iloc[-1]
            market_trend = 'strong' if short_ma > long_ma * 1.02 else 'weak' if short_ma < long_ma * 0.98 else 'neutral'
            
            # Adjust multiplier based on market trend
            if market_trend == 'strong':
                volatility_multiplier *= 1.2
            elif market_trend == 'weak':
                volatility_multiplier *= 0.8
        
        return base_risk * volatility_multiplier

    def check_entry_conditions(self, current_price, current_volume, orb_high, orb_low, avg_volume, i, ticker):
        """Check all entry conditions including enhanced volume filter"""
        # Basic price breakout conditions
        long_breakout = current_price > orb_high
        short_breakout = current_price < orb_low
        
        # Enhanced volume confirmation
        volume_confirmed = True
        if self.use_volume_filter and avg_volume is not None:
            volume_confirmed = self.check_volume_conditions(current_volume, avg_volume, i)
        
        # Volatility check
        current_volatility = self.data['volatility'].iloc[i]
        current_volatility_ratio = self.data['volatility_ratio'].iloc[i]
        volatility_confirmed = self.check_volatility_conditions(
            current_volatility, current_volatility_ratio, ticker
        )
        
        # For breakout_hold mode, require price to stay above/below breakout level for 3 bars
        if self.entry_mode == 'breakout_hold':
            if i >= 3:
                if long_breakout:
                    long_breakout = all(self.data['close'].iloc[i-j] > orb_high for j in range(1, 4))
                if short_breakout:
                    short_breakout = all(self.data['close'].iloc[i-j] < orb_low for j in range(1, 4))
        
        return (long_breakout and volume_confirmed and volatility_confirmed,
                short_breakout and volume_confirmed and volatility_confirmed)

    def update_trailing_stop(self, current_trade, current_price, current_volatility):
        """Update trailing stop if conditions are met"""
        if not self.trailing_stop:
            return current_trade['stop_loss']
            
        # Calculate dynamic trailing stop activation
        dynamic_activation = self.calculate_dynamic_trailing_stop(current_price, current_volatility)
            
        if current_trade['direction'] == 'LONG':
            profit_target = current_trade['profit_target']
            entry_price = current_trade['entry_price']
            current_profit_pct = (current_price - entry_price) / entry_price
            target_profit_pct = (profit_target - entry_price) / entry_price
            
            if current_profit_pct >= target_profit_pct * dynamic_activation:
                new_stop = current_price * (1 - self.min_stop_distance)
                return max(new_stop, current_trade['stop_loss'])
                
        elif current_trade['direction'] == 'SHORT':
            profit_target = current_trade['profit_target']
            entry_price = current_trade['entry_price']
            current_profit_pct = (entry_price - current_price) / entry_price
            target_profit_pct = (entry_price - profit_target) / entry_price
            
            if current_profit_pct >= target_profit_pct * dynamic_activation:
                new_stop = current_price * (1 + self.min_stop_distance)
                return min(new_stop, current_trade['stop_loss'])
                
        return current_trade['stop_loss']

    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on risk parameters"""
        account_value = 100000  # Starting account value
        risk_amount = account_value * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        position_size = int(risk_amount / stop_distance)
        return position_size

    def backtest(self):
        """Run backtest with optimized performance"""
        trades = []
        current_trade = None
        cumulative_equity = [100000]
        equity_timestamps = [self.data.index[0]]
        
        # Vectorized operations for price data
        highs = self.data['high'].values
        lows = self.data['low'].values
        closes = self.data['close'].values
        volumes = self.data['volume'].values
        timestamps = self.data.index
        
        for i in range(1, len(self.data)):
            current_date = timestamps[i].date()
            current_time = timestamps[i].time()
            
            # Skip if before market open
            if current_time < datetime.strptime('09:30', '%H:%M').time():
                continue
                
            orb_high, orb_low, avg_volume = self.calculate_orb(current_date)
            if orb_high is None or orb_low is None:
                continue
                
            current_price = closes[i]
            current_volume = volumes[i]
            
            # Check for existing trade
            if current_trade:
                # Update trailing stop if enabled
                if self.trailing_stop:
                    current_trade['stop_loss'] = self.update_trailing_stop(
                        current_trade, current_price, self.data['volatility'].iloc[i]
                    )
                
                # Update position size based on scale-out levels
                current_trade['position_size'] = self.update_position_size(
                    current_trade, current_price
                )
                
                # Check exit conditions
                if (current_trade['direction'] == 'LONG' and 
                    (current_price <= current_trade['stop_loss'] or 
                     current_price >= current_trade['profit_target'])):
                    current_trade['exit_price'] = current_price
                    current_trade['exit_time'] = timestamps[i]
                    trades.append(current_trade)
                    current_trade = None
                elif (current_trade['direction'] == 'SHORT' and 
                      (current_price >= current_trade['stop_loss'] or 
                       current_price <= current_trade['profit_target'])):
                    current_trade['exit_price'] = current_price
                    current_trade['exit_time'] = timestamps[i]
                    trades.append(current_trade)
                    current_trade = None
                    
                # Update equity curve
                if current_trade:
                    pnl = (current_price - current_trade['entry_price']) * current_trade['position_size']
                    if current_trade['direction'] == 'SHORT':
                        pnl = -pnl
                    cumulative_equity.append(cumulative_equity[-1] + pnl)
                    equity_timestamps.append(timestamps[i])
            
            # Check for new trade entries
            if not current_trade:
                long_entry, short_entry = self.check_entry_conditions(
                    current_price, current_volume, orb_high, orb_low, avg_volume, i, self.ticker
                )
                
                if long_entry:
                    stop_loss = max(orb_low, current_price * (1 - self.min_stop_distance))
                    position_size = self.calculate_position_size(current_price, stop_loss)
                    profit_target = current_price + self.calculate_dynamic_profit_target(
                        current_price, stop_loss, self.data['volatility'].iloc[i]
                    )
                    scale_out_levels = self.calculate_scale_out_levels(
                        current_price, stop_loss, 'LONG'
                    )
                    current_trade = {
                        'direction': 'LONG',
                        'entry_time': timestamps[i],
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'profit_target': profit_target,
                        'position_size': position_size,
                        'scale_out_levels': scale_out_levels
                    }
                    logger.info(f"[TRADE] LONG Entry at {timestamps[i]}")
                    logger.info(f"Entry Price: ${current_price:.2f}")
                    logger.info(f"Stop Loss: ${stop_loss:.2f}")
                    logger.info(f"Position Size: {position_size} shares")
                    logger.info(f"Profit Target: ${profit_target:.2f}\n")
                    
                elif short_entry:
                    stop_loss = min(orb_high, current_price * (1 + self.min_stop_distance))
                    position_size = self.calculate_position_size(current_price, stop_loss)
                    profit_target = current_price - self.calculate_dynamic_profit_target(
                        current_price, stop_loss, self.data['volatility'].iloc[i]
                    )
                    scale_out_levels = self.calculate_scale_out_levels(
                        current_price, stop_loss, 'SHORT'
                    )
                    current_trade = {
                        'direction': 'SHORT',
                        'entry_time': timestamps[i],
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'profit_target': profit_target,
                        'position_size': position_size,
                        'scale_out_levels': scale_out_levels
                    }
                    logger.info(f"[TRADE] SHORT Entry at {timestamps[i]}")
                    logger.info(f"Entry Price: ${current_price:.2f}")
                    logger.info(f"Stop Loss: ${stop_loss:.2f}")
                    logger.info(f"Position Size: {position_size} shares")
                    logger.info(f"Profit Target: ${profit_target:.2f}\n")
        
        # Close any open trade at the end
        if current_trade:
            current_trade['exit_price'] = closes[-1]
            current_trade['exit_time'] = timestamps[-1]
            trades.append(current_trade)
        
        return trades, cumulative_equity, equity_timestamps

    def analyze_performance(self, trades):
        """Analyze strategy performance with vectorized operations"""
        if not trades:
            return 0, 0, 0, 0, 0, 0, 0, 0
            
        # Convert trades to DataFrame for vectorized operations
        trades_df = pd.DataFrame(trades)
        trades_df['pnl'] = (trades_df['exit_price'] - trades_df['entry_price']) * trades_df['position_size']
        trades_df.loc[trades_df['direction'] == 'SHORT', 'pnl'] *= -1
        
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                          trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) if losing_trades > 0 else float('inf')
        
        return (total_trades, winning_trades, losing_trades, win_rate, 
                total_pnl, avg_win, avg_loss, profit_factor)

def download_data(ticker, output_file):
    """Download historical data for a given ticker"""
    print(f"Downloading data for {ticker}...")
    
    # Initialize Alpaca API client
    api = REST(
        os.getenv("ALPACA_PAPER_API_KEY"),
        os.getenv("ALPACA_PAPER_API_SECRET"),
        base_url='https://paper-api.alpaca.markets'
    )
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=150)
    
    # Get historical data
    data = api.get_bars(
        ticker,
        '1Min',
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d')
    ).df
    
    # Save to CSV
    data.to_csv(output_file)
    print(f"Data saved to {output_file}")

def run_backtest(ticker, entry_mode, orb_duration=30, use_volume_filter=True,
                trailing_stop=True, trailing_stop_activation=0.5,
                use_volatility_filter=True, scale_out_levels=None,
                profit_target_multiplier=1.5):
    """Run backtest for a single ticker and mode with specified parameters"""
    data_file = f"{ticker}_data.csv"
    if not os.path.exists(data_file):
        download_data(ticker, data_file)
    
    data = pd.read_csv(data_file, index_col='timestamp', parse_dates=True)
    strategy = ORBStrategy(
        data, 
        entry_mode=entry_mode,
        orb_duration=orb_duration,
        use_volume_filter=use_volume_filter,
        trailing_stop=trailing_stop,
        trailing_stop_activation=trailing_stop_activation,
        use_volatility_filter=use_volatility_filter,
        scale_out_levels=scale_out_levels,
        profit_target_multiplier=profit_target_multiplier,
        ticker=ticker
    )
    trades, equity, timestamps = strategy.backtest()
    stats = strategy.analyze_performance(trades)
    
    return {
        'ticker': ticker,
        'entry_mode': entry_mode,
        'trades': trades,
        'equity': equity,
        'timestamps': timestamps,
        'stats': stats,
        'orb_duration': orb_duration,
        'use_volume_filter': use_volume_filter,
        'trailing_stop': trailing_stop,
        'trailing_stop_activation': trailing_stop_activation,
        'use_volatility_filter': use_volatility_filter,
        'scale_out_levels': scale_out_levels,
        'profit_target_multiplier': profit_target_multiplier
    }

def create_report_directory():
    """Create a directory for backtest reports with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = f"backtest_reports/report_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    return report_dir

def create_consolidated_report(results, report_dir):
    """Create a single report with all charts"""
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    n_strategies = len(results)
    fig = plt.figure(figsize=(15, 5 * n_strategies))
    gs = GridSpec(n_strategies, 2)
    
    for i, result in enumerate(results):
        # Equity curve
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(result['timestamps'], result['equity'], label='Equity Curve')
        ax1.set_title(f"{result['ticker']} - {result['entry_mode']} Entry Mode")
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True)
        
        # Trade markers
        ax2 = fig.add_subplot(gs[i, 1])
        data = pd.read_csv(f"{result['ticker']}_data.csv", index_col='timestamp', parse_dates=True)
        ax2.plot(data.index, data['close'], label='Price', alpha=0.5)
        
        for trade in result['trades']:
            if trade['direction'] == 'LONG':
                ax2.scatter(trade['entry_time'], trade['entry_price'], 
                          color='green', marker='^', s=100)
                ax2.scatter(trade['exit_time'], trade['exit_price'], 
                          color='red', marker='v', s=100)
            else:
                ax2.scatter(trade['entry_time'], trade['entry_price'], 
                          color='red', marker='v', s=100)
                ax2.scatter(trade['exit_time'], trade['exit_price'], 
                          color='green', marker='^', s=100)
        
        ax2.set_title(f"{result['ticker']} - Trade Markers")
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price ($)')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'backtest_report.pdf'))
    plt.close()

def main():
    # Create report directory
    report_dir = create_report_directory()
    
    # Focus only on XLK
    tickers = ['XLK']
    entry_modes = ['immediate']  # Focus on immediate entry mode
    
    # Use only the optimal configuration for XLK
    param_combinations = [
        {
            'orb_duration': 30,
            'use_volume_filter': True,
            'trailing_stop': True,
            'trailing_stop_activation': 0.4,
            'use_volatility_filter': False,
            'scale_out_levels': None,
            'profit_target_multiplier': 2.0
        }
    ]
    
    all_results = []
    
    # Run backtests in parallel for each parameter combination
    with ProcessPoolExecutor() as executor:
        futures = []
        for ticker in tickers:
            for mode in entry_modes:
                for params in param_combinations:
                    futures.append(executor.submit(
                        run_backtest, 
                        ticker, 
                        mode,
                        params['orb_duration'],
                        params['use_volume_filter'],
                        params['trailing_stop'],
                        params['trailing_stop_activation'],
                        params['use_volatility_filter'],
                        params['scale_out_levels'],
                        params['profit_target_multiplier']
                    ))
        
        results = [f.result() for f in futures]
        all_results.extend(results)
    
    # Create consolidated report
    create_consolidated_report(all_results, report_dir)
    
    # Save summary to text file
    with open(os.path.join(report_dir, 'backtest_summary.txt'), 'w') as f:
        f.write("Backtest Summary:\n")
        f.write("=" * 50 + "\n")
        
        # Group results by parameter combination
        for params in param_combinations:
            f.write(f"\nParameter Combination:\n")
            f.write(f"ORB Duration: {params['orb_duration']} minutes\n")
            f.write(f"Volume Filter: {params['use_volume_filter']}\n")
            f.write(f"Trailing Stop: {params['trailing_stop']}\n")
            f.write(f"Trailing Stop Activation: {params['trailing_stop_activation']}\n")
            f.write(f"Volatility Filter: {params['use_volatility_filter']}\n")
            f.write(f"Scale Out Levels: {params['scale_out_levels']}\n")
            f.write(f"Profit Target Multiplier: {params['profit_target_multiplier']}\n")
            f.write("-" * 30 + "\n")
            
            # Filter results for this parameter combination
            param_results = [r for r in results if all(
                r.get(k) == v for k, v in params.items()
            )]
            
            for result in param_results:
                stats = result['stats']
                f.write(f"\n{result['ticker']} - {result['entry_mode']} Entry Mode:\n")
                f.write(f"Total Trades: {stats[0]}\n")
                f.write(f"Win Rate: {stats[3]:.1f}%\n")
                f.write(f"Total P&L: ${stats[4]:.2f}\n")
                f.write(f"Profit Factor: {stats[7]:.2f}\n")
                f.write(f"Average Win: ${stats[5]:.2f}\n")
                f.write(f"Average Loss: ${stats[6]:.2f}\n")
    
    # Print summary to console
    print(f"\nBacktest completed. Results saved in: {report_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()
