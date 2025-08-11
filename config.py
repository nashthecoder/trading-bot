"""
Configuration module for the trading bot.
Centralizes important settings to avoid magic numbers in the code.
"""

# Trading Parameters
DEFAULT_BUY_PERCENTAGE = 0.08  # 8% of capital per trade
DEFAULT_PROFIT_TARGET = 0.016  # 1.6% profit target
DEFAULT_STOP_LOSS = 0.008     # 0.8% stop loss
DEFAULT_TRADE_FREQUENCY = 120  # 2 minutes between trades

# AI Scoring Thresholds
AI_SCORE_THRESHOLD = 0.6      # Minimum AI score to allow trades
AI_GAIN_THRESHOLD = 1.5       # Minimum expected gain percentage
LUTE_SCORE_MIN = 0.0         # Minimum LUTE score

# API Rate Limiting
BALANCE_CHECK_INTERVAL = 30   # Seconds between balance checks
API_RETRY_DELAY = 1.5        # Seconds to wait between retries

# Fee Configuration
FEE_RATE_TAKER = 0.006       # 0.6% taker fee
FEE_RATE_MAKER = 0.003       # 0.3% maker fee

# Supported Trading Pairs
SUPPORTED_PAIRS = [
    'BTC-USDC', 'ETH-USDC', 'ADA-USDC', 'SOL-USDC',
    'AVAX-USDC', 'DOT-USDC', 'LINK-USDC', 'UNI-USDC',
    'AAVE-USDC', 'CRV-USDC', 'SUSHI-USDC', 'YFI-USDC',
    'DOGE-USDC', 'PEPE-USDC', 'SHIB-USDC',
    'ARB-USDC', 'MATIC-USDC', 'FIL-USDC', 'NEAR-USDC',
    'ALGO-USDC', 'XRP-USDC', 'LTC-USDC', 'ETC-USDC',
    'FET-USDC', 'GRT-USDC', 'HBAR-USDC', 'ICP-USDC',
    'IDEX-USDC', 'SUI-USDC', 'SUPER-USDC', 'SWFTC-USDC',
    'USDT-USDC', 'VET-USDC', 'XLM-USDC'
]

# Logging Configuration
LOG_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = "logs.txt"