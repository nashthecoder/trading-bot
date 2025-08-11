# 🤖 AI-Powered Cryptocurrency Trading Bot

An advanced, AI-driven cryptocurrency trading bot with a comprehensive web interface for automated trading on Coinbase. This bot supports multiple trading strategies, real-time market analysis, and sophisticated risk management features.

## 🚀 Features

### Core Trading Capabilities
- **50+ Cryptocurrency Pairs**: Support for major cryptocurrencies including BTC, ETH, ADA, SOL, and many more
- **AI-Assisted Trading**: Machine learning algorithms for market prediction and trend analysis
- **Multiple Trading Strategies**: 9 predefined trading series with different risk profiles
- **Real-time Market Data**: Live price feeds and market analysis
- **Automated Portfolio Management**: Dynamic position sizing and portfolio rebalancing

### Trading Strategies
1. **Serie 1 - Sécurisée**: Conservative strategy with low risk
2. **Serie 2 - IA dynamique**: AI-driven dynamic trading 
3. **Serie 3 - Scalping**: High-frequency scalping strategy
4. **Serie 4 - Tendance IA**: AI-powered trend following
5. **Serie 5 - Swing Trading**: Medium-term swing trading
6. **Serie 6 - Scalping Volatile**: Volatile market scalping
7. **Serie 7 - Stablecoin Hedging**: Hedging with stablecoins
8. **Serie 8 - Hold Moyen-Terme**: Medium-term holding strategy
9. **Serie 9 - Anti-Volatility**: Anti-volatility protection

### Risk Management
- **Stop Loss & Take Profit**: Configurable profit targets and loss limits
- **Trailing Stop Loss**: Dynamic stop-loss adjustment
- **Position Sizing**: Intelligent capital allocation
- **Drawdown Protection**: Maximum loss thresholds
- **Simulation Mode**: Paper trading for strategy testing

### Web Interface Features
- **Real-time Dashboard**: Live performance monitoring and charts
- **Strategy Configuration**: Easy setup of trading parameters
- **Portfolio Tracking**: Account balance and profit/loss tracking
- **Trade History**: Detailed order and execution logs
- **Performance Analytics**: Charts and KPI visualization

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Coinbase Advanced Trading API access
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/nashthecoder/trading-bot.git
cd trading-bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API credentials**
Create a `.env` file with your Coinbase API credentials:
```env
COINBASE_API_KEY=your_api_key
COINBASE_API_SECRET=your_api_secret
COINBASE_PASSPHRASE=your_passphrase
```

4. **Run the application**
```bash
python trading_bot.py
```

## 🔧 Code Structure

### Main Files
- `trading_bot.py` - Main application file (formerly LUTESSIA_FINAL_PRODUCTION_EXEC_REBUILT_WITH_IA_CONNECTED_FINAL_v2_20250805_164551.py)
- `config.py` - Configuration settings and constants
- `utils.py` - Common utility functions
- `index.html` - Web interface
- `requirements.txt` - Python dependencies

### Recent Improvements
- ✅ Removed duplicate function definitions
- ✅ Fixed AI scoring logic issues
- ✅ Improved error handling for volatility calculations
- ✅ Standardized logging functions
- ✅ Added proper .gitignore for version control
- ✅ Fixed deprecated sklearn dependency

## 📊 Usage

### Web Interface
Access the trading bot interface at `http://localhost:5000` after starting the application.

#### Main Dashboard Features:
- **Bot Control**: Start/stop trading operations
- **Strategy Selection**: Choose from 9 predefined strategies
- **Crypto Pair Selection**: Select which cryptocurrencies to trade
- **Risk Parameters**: Configure stop-loss, take-profit, and position sizing
- **Performance Monitoring**: Real-time charts and profit tracking

#### Configuration Options:
- **Buy Percentage**: Percentage of capital to use per trade
- **Profit Target**: Target profit percentage for trades
- **Stop Loss**: Maximum acceptable loss per trade
- **Trade Frequency**: How often to execute trades (in seconds)
- **Order Types**: Market, limit, or auto order execution
- **Simulation Mode**: Enable paper trading for testing

### Trading Parameters

#### Basic Settings
- `buy_percentage_of_capital`: Percentage of available capital to use per trade (default: 0.08)
- `sell_profit_target`: Target profit percentage (default: 0.016)
- `sell_stop_loss_target`: Stop loss percentage (default: 0.008)
- `trade_frequency`: Trading frequency in seconds (default: varies by strategy)

#### Advanced Settings
- `trailing_start_threshold`: When to start trailing stop-loss
- `trailing_sl_step`: Step size for trailing stop-loss
- `min_hold_duration`: Minimum time to hold positions
- `max_hold_duration`: Maximum time to hold positions
- `compound_enabled`: Enable profit compounding

## 🔧 Dependencies

### Core Libraries
- **Flask**: Web application framework
- **TensorFlow/Keras**: Machine learning and AI predictions
- **pandas/numpy**: Data analysis and manipulation
- **scikit-learn**: Additional ML algorithms
- **matplotlib/seaborn**: Data visualization

### Trading & Finance
- **coinbase**: Coinbase API integration
- **yfinance**: Yahoo Finance data
- **pandas-ta/ta**: Technical analysis indicators

### Web Interface
- **Flask-SocketIO**: Real-time web communication
- **Chart.js**: Interactive charts and graphs
- **Bootstrap**: UI framework

See `requirements.txt` for the complete list of dependencies.

## 📈 Supported Trading Pairs

The bot supports 50+ cryptocurrency trading pairs, including:

**Major Cryptocurrencies:**
- BTC-USDC, ETH-USDC, ADA-USDC, SOL-USDC
- AVAX-USDC, DOT-USDC, LINK-USDC, UNI-USDC

**DeFi Tokens:**
- AAVE-USDC, CRV-USDC, SUSHI-USDC, YFI-USDC

**Meme Tokens:**
- DOGE-USDC, PEPE-USDC, SHIB-USDC

**Layer 1/2 Tokens:**
- ARB-USDC, MATIC-USDC, FIL-USDC, NEAR-USDC

*Note: Some pairs may have limited AI trading support (marked in red in the interface)*

## ⚠️ Risk Warning

**IMPORTANT DISCLAIMER:**
- Cryptocurrency trading involves substantial risk of loss
- Past performance does not guarantee future results
- Only trade with funds you can afford to lose
- This bot is provided for educational purposes
- Always test strategies in simulation mode first
- Consider consulting with financial advisors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔍 Monitoring & Logs

The bot generates comprehensive logs including:
- Trade execution details
- Market analysis results
- Profit/loss tracking
- Error handling and debugging information

Logs are stored in `logs.txt` and displayed in real-time through the web interface.

## 🆘 Support

For support, issues, or feature requests:
1. Check existing [GitHub Issues](https://github.com/nashthecoder/trading-bot/issues)
2. Create a new issue with detailed information
3. Include logs and configuration details when reporting bugs

---

⚡ **Happy Trading!** Remember to always trade responsibly and never invest more than you can afford to lose.