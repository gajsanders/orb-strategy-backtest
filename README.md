# Opening Range Breakout (ORB) Trading Strategy

This project implements an Opening Range Breakout (ORB) trading strategy using the Alpaca trading API. The strategy is designed to identify and trade breakouts from the opening range of trading sessions.

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

### Version 2.0.0
- Implemented dynamic trailing stops based on market volatility
- Added volume confirmation filter
- Optimized ORB duration to 30 minutes
- Improved entry conditions with 3-bar confirmation
- Enhanced backtesting with parallel processing
- Added detailed performance reporting
- Focused on best-performing tickers (XLK, SMH)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading strategy is for educational purposes only. Past performance is not indicative of future results. Always do your own research and risk management before trading with real money.

## Acknowledgments

- Alpaca Markets for providing the trading API
- The trading community for insights and feedback
- This strategy was inspired by the Opening Range Breakout strategy explained in [this YouTube video](https://youtu.be/GEnKQ834U1c?si=MkBGil_6TfEK71bF) 