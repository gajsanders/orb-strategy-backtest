# Opening Range Breakout (ORB) Strategy

A sophisticated trading strategy implementing Opening Range Breakout with enhanced volume analysis and dynamic profit targets.

## Features

- **Opening Range Breakout Strategy**: Identifies and trades breakouts from the first 30 minutes of trading
- **Dynamic Risk Management**: 
  - Configurable ORB duration (15, 30, or 60 minutes)
  - Dynamic trailing stops based on volatility
  - Volume-based confirmation filters
- **Paper Trading Support**: Uses Alpaca's paper trading environment for safe testing
- **Backtesting Capabilities**: Includes comprehensive backtesting functionality
- **Performance Analytics**: Detailed trade analysis including:
  - Win rate and profit metrics
  - Trade duration analysis
  - Drawdown calculations
  - Streak analysis
  - Time-based trade distribution
  - Volatility-adjusted performance metrics

## Prerequisites

- Python 3.8+
- Alpaca Paper Trading Account
- Required Python packages (see Installation section)

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Alpaca API credentials to `.env`:
     ```
     ALPACA_PAPER_API_KEY=your_paper_api_key_here
     ALPACA_PAPER_API_SECRET=your_paper_api_secret_here
     ```

## Usage

### Running the Strategy

```python
from strategy_orb_mcp import ORBStrategy
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize strategy with optimized parameters
strategy = ORBStrategy(
    data,
    entry_mode='immediate',
    orb_duration=30,
    use_volume_filter=True,
    trailing_stop=True,
    trailing_stop_activation=0.4
)

# Run backtest
trades, equity, timestamps = strategy.backtest()

# Analyze performance
stats = strategy.analyze_performance(trades)
```

### Strategy Parameters

The strategy uses the following optimized parameters:
- Opening Range: First 30 minutes of trading
- Volume Filter: Requires 20% above average volume for confirmation
- Dynamic Trailing Stop: 
  - Base activation at 40% of profit target
  - Adjusts based on market volatility
  - More aggressive in high volatility (70% of base)
  - Less aggressive in low volatility (130% of base)
- Entry Modes:
  - Immediate: Enters on breakout confirmation
  - Breakout Hold: Requires 3-bar confirmation

## Project Structure

```
.
├── strategy_orb_mcp.py     # Main strategy implementation
├── download_historical_data.py  # Data download utility
├── requirements.txt        # Project dependencies
├── .env.example           # Example environment variables
├── backtest_reports/      # Directory for backtest results
└── README.md             # This file
```

## Performance Metrics

The strategy provides detailed performance analysis including:
- Total trades and win rate
- Average win/loss
- Profit factor
- Maximum drawdown
- Trade duration analysis
- Time-based trade distribution
- Streak analysis
- Volatility-adjusted metrics
- Dynamic trailing stop effectiveness

## Recent Updates

### Enhanced Volume Analysis
- Implemented multi-factor volume confirmation:
  - Volume trend analysis (20-period vs 40-period moving average)
  - Volume momentum measurement
  - Increasing volume pattern detection
  - Recent volume average comparison

### Dynamic Profit Targets
- Added volatility-based profit target adjustment:
  - Low volatility: 1.5x risk
  - Medium volatility: 2.0x risk
  - High volatility: 2.5x risk
- Market trend consideration:
  - Strong trend: 20% increase in target
  - Weak trend: 20% decrease in target

### Performance Metrics
- Profit Factor: ~1.8-2.2
- Average Win: $2,550-4,400 per trade
- Average Loss: $1,360-2,640 per trade
- Position Sizing: 1700-2200 shares (dynamic based on volatility)

## Strategy Components

### Entry Conditions
1. Opening Range Breakout
2. Volume Confirmation
   - Above average volume
   - Increasing volume trend
   - Strong volume momentum
3. Volatility Filter
   - Dynamic thresholds based on market conditions
   - Ticker-specific adjustments

### Risk Management
- Dynamic position sizing
- Trailing stops with volatility adjustment
- Scale-out levels for partial profit taking
- Maximum risk per trade: 2% of account

### Exit Conditions
1. Stop Loss
   - Initial: Based on ORB levels
   - Trailing: Activated at 40% of target profit
2. Profit Target
   - Dynamic based on volatility
   - Adjusted for market trend
3. Scale-out Levels
   - Partial exits at 50%, 75%, and 100% of target

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
ALPACA_PAPER_API_KEY=your_key
ALPACA_PAPER_API_SECRET=your_secret
```

3. Run backtest:
```bash
python strategy_orb_mcp.py
```

## Configuration

Key parameters can be adjusted in the `main()` function:
- `orb_duration`: Opening range period (default: 30 minutes)
- `risk_per_trade`: Maximum risk per trade (default: 2%)
- `profit_target_multiplier`: Base profit target multiplier (default: 1.5)
- `trailing_stop_activation`: When to activate trailing stop (default: 0.4)

## Performance Analysis

The strategy generates detailed performance reports including:
- Equity curve
- Trade markers on price chart
- Win rate and profit factor
- Average win/loss metrics
- Position sizing analysis

Reports are saved in the `backtest_reports` directory with timestamps.

## Future Improvements

1. Profit Factor Enhancement:
   - Tighter stop losses during high volatility
   - Partial profit taking at 1:1 risk-reward
   - Trend strength confirmation

2. Risk Management:
   - Maximum position size limits
   - Daily loss limits
   - Correlation-based position sizing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading strategy is for educational purposes only. Past performance is not indicative of future results. Always do your own research and risk management before trading with real money.

## Acknowledgments

- Alpaca Markets for providing the trading API
- The trading community for insights and feedback
- This strategy was inspired by the Opening Range Breakout strategy explained in [this YouTube video](https://youtu.be/GEnKQ834U1c?si=MkBGil_6TfEK71bF) 