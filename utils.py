"""
Utility functions for the trading bot.
Consolidates common functionality to reduce code duplication.
"""

import logging
from datetime import datetime
from decimal import Decimal

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

def log_message(message):
    """
    Centralized logging function with timestamp.
    Prints to console and logs to file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamped_message = f"{timestamp} - {message}"
    print(timestamped_message)
    logging.info(timestamped_message)

def get_volatility_score(closes):
    """
    Calculate volatility score from price data.
    Handles various input formats and edge cases.
    """
    if not HAS_NUMPY:
        log_message("⚠️ NumPy not available, returning default volatility score")
        return 0.0
        
    try:
        if closes is None:
            return 0.0
        
        # Convert to numpy array and handle various input types
        if hasattr(closes, 'values'):  # pandas Series/DataFrame
            arr = np.asarray(closes.values).flatten()
        else:
            arr = np.asarray(closes).flatten()
        
        # Ensure we have enough data points
        if len(arr) < 2:
            log_message(f"⚠️ get_volatility_score: Not enough data points ({len(arr)})")
            return 0.0
        
        # Remove any NaN values
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2:
            log_message("⚠️ get_volatility_score: All values are NaN")
            return 0.0
        
        # Calculate returns (percentage change)
        returns = np.diff(arr) / arr[:-1]
        volatility = np.std(returns)
        
        return float(volatility)
        
    except Exception as e:
        log_message(f"❌ Erreur dans get_volatility_score: {e}")
        return 0.0

def should_buy_lutessia(score_final, lute_score, expected_gain, seuil_score=0.6, seuil_gain=1.5):
    """
    Centralized AI decision function for buy orders.
    """
    log_message(
        f"🧠 [CHECK IA] Score: {score_final:.4f}, LUTE: {lute_score:.4f}, Gain: {expected_gain:.2f}%"
    )
    if score_final < seuil_score:
        log_message(f"⛔ Refus IA : Score trop bas (< {seuil_score})")
        return False
    if lute_score <= 0:
        log_message("⛔ Refus IA : LUTE négatif ou nul (≤ 0)")
        return False
    if expected_gain < seuil_gain:
        log_message(f"⛔ Refus IA : Gain insuffisant (< {seuil_gain}%)")
        return False

    log_message("✅ Achat autorisé par IA (score, LUTE et gain OK)")
    return True

def validate_lutessia_before_buy(score_final, lute_score, expected_gain_pct, product_id="N/A"):
    """
    Validate trade using Lutessia AI before executing buy order.
    """
    if not should_buy_lutessia(score_final, lute_score, expected_gain_pct):
        log_message(
            f"⛔ Achat bloqué [Lutessia] ➜ {product_id} | score={score_final:.4f} | LUTE={lute_score} | gain={expected_gain_pct:.2f}%"
        )
        return False
    return True

def compare_first_real_and_last_pred(predictions_or_yesterday=None, today_pred=None):
    """
    Flexible function that handles both single array and two-parameter calls.
    Returns True/trend message if price will increase, False/trend message if decrease.
    """
    try:
        # Handle two-parameter call (yesterday_value, today_prediction)
        if today_pred is not None:
            if today_pred > predictions_or_yesterday:
                return f"📈 Le prix de la crypto va augmenter 🚀"
            else:
                return f"📉 Le prix de la crypto va baisser ⚠️"
        
        # Handle single array parameter (backward compatibility)
        if predictions_or_yesterday is not None and hasattr(predictions_or_yesterday, '__len__'):
            if len(predictions_or_yesterday) >= 2:
                return predictions_or_yesterday[-1] > predictions_or_yesterday[0]
        
        return False

    except Exception as e:
        log_message(f"❌ Erreur dans compare_first_real_and_last_pred: {e}")
        return False

def format_currency(amount, currency="USDC"):
    """Format currency amounts for display."""
    try:
        if isinstance(amount, (int, float, Decimal)):
            return f"{float(amount):.2f} {currency}"
        return f"0.00 {currency}"
    except:
        return f"0.00 {currency}"

def safe_float(value, default=0.0):
    """Safely convert value to float with fallback."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_decimal(value, default="0.0"):
    """Safely convert value to Decimal with fallback."""
    try:
        return Decimal(str(value))
    except:
        return Decimal(default)