# Opening Range Breakout (ORB) Trading Strategy

This project implements an Opening Range Breakout (ORB) trading strategy using the Alpaca trading API. The strategy is designed to identify and trade breakouts from the opening range of trading sessions.

## Features

- **Opening Range Breakout Strategy**: Identifies and trades breakouts from the first 15 minutes of trading
- **Paper Trading Support**: Uses Alpaca's paper trading environment for safe testing
- **Backtesting Capabilities**: Includes comprehensive backtesting functionality
- **Performance Analytics**: Detailed trade analysis including:
  - Win rate and profit metrics
  - Trade duration analysis
  - Drawdown calculations
  - Streak analysis
  - Time-based trade distribution

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

# Initialize strategy
strategy = ORBStrategy(
    os.getenv("ALPACA_PAPER_API_KEY"),
    os.getenv("ALPACA_PAPER_API_SECRET")
)

# Run backtest
trades = strategy.backtest("SPY_20250501_20250531.csv")

# Analyze performance
strategy.analyze_performance(trades)
```

### Strategy Parameters

The strategy uses the following default parameters:
- Opening Range: First 15 minutes of trading
- Breakout Confirmation: Price must break above/below the range and retest
- Exit: End of day (4:00 PM ET)

## Project Structure

```
.
├── strategy_orb_mcp.py     # Main strategy implementation
├── download_historical_data.py  # Data download utility
├── requirements.txt        # Project dependencies
├── .env.example           # Example environment variables
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