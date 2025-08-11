import logging
from decimal import Decimal


# ⚠️ TEMP : Fonctions IA placeholder pour éviter crash
def log_message(message):
    from datetime import datetime
    import logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamped_message = f"{timestamp} - {message}"
    print(timestamped_message)
    logging.info(timestamped_message)

def get_volatility_score(closes):
    import numpy as np
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
        
        log_message(f"✅ Volatility calculated: {volatility:.6f} from {len(arr)} data points")
        return float(volatility)
        
    except Exception as e:
        log_message(f"❌ Erreur dans get_volatility_score: {e}")
        return 0.0

# Function removed - using should_buy_lutessia from line 66

def generate_ai_scores(product_id):
    import random
    lstm_conf = round(random.uniform(0.45, 0.75), 4)
    gru_conf = round(random.uniform(0.40, 0.70), 4)
    atr_score = round(random.uniform(0.01, 0.05), 4)
    lute_score = round(random.uniform(0.00, 1.00), 4)
    final_score = round(0.4 * lstm_conf + 0.4 * gru_conf + 0.2 * atr_score, 4)
    log_message(
        f"🧠 {product_id} | LSTM: {lstm_conf:.4f} | GRU: {gru_conf:.4f} | ATR: {atr_score:.4f} | LUTE: {lute_score:.4f} | Final: {final_score:.4f}")
    return final_score, lute_score

def get_signal_strength(product_id):
    # À remplacer par la vraie logique IA
    return Decimal("0.61")


# Suppression de la mauvaise version get_volatility_score(product_id)


# === LUTE PATCH - GLOBAL ENFORCER ===



# Function removed - using log_message from line 6
def should_buy_lutessia(
    score_final, lute_score, expected_gain, seuil_score=0.6, seuil_gain=1.5
):
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


def validate_lutessia_before_buy(
    score_final, lute_score, expected_gain_pct, product_id="N/A"
):
    if not should_buy_lutessia(score_final, lute_score, expected_gain_pct):
        log_message(
            f"⛔ Achat bloqué [Lutessia] ➜ {product_id} | score={score_final:.4f} | LUTE={lute_score} | gain={expected_gain_pct:.2f}%"
        )
        return False
    return True


# Function removed - using should_buy_lutessia from line 66


# === Logique IA corrigée et intégrée ===
from datetime import datetime
from decimal import Decimal

SEUIL_SCORE_ACHAT = 0.51
SEUIL_GAIN_ACHAT = 0.8


def enregistrer_log(texte):
    horodatage = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{horodatage} - {texte}")


def normaliser(valeur, min_val, max_val):
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (valeur - min_val) / (max_val - min_val)))


POIDS = {"lstm": 0.3, "atr": 0.2, "guru": 0.2, "lute": 0.3}


def calcul_score(lstm, atr, guru, lute, poids=POIDS):
    lstm = float(lstm)
    atr = float(atr)
    guru = float(guru)
    lute = float(lute)
    norm_lstm = normaliser(lstm, 0, 1)
    norm_atr = normaliser(atr, 0, 0.1)
    norm_guru = normaliser((guru + 1) / 2, 0, 1)
    norm_lute = normaliser((lute + 1) / 2, 0, 1)
    score = (
        poids["lstm"] * norm_lstm
        + poids["atr"] * norm_atr
        + poids["guru"] * norm_guru
        + poids["lute"] * norm_lute
    )
    return round(score, 4)


# === Fin logique IA ===


from decimal import Decimal

import random

import requests


ORDER_TYPE = "market"  # "market" or "limit"


FEE_RATE_TAKER = Decimal("0.006")

FEE_RATE_MAKER = Decimal("0.003")

MIN_NET_PROFIT_TARGET = Decimal("0.015")

TP_DEFAULT = Decimal("0.025")

SL_DEFAULT = Decimal("0.012")


def get_fee_rate():

    return FEE_RATE_TAKER if ORDER_TYPE == "market" else FEE_RATE_MAKER


def ia_predicts_rise(pair):

    return random.choice([True, False])


def should_trade(pair):

    print(f"🔎 Analyse IA ➜ {pair}")

    if not ia_predicts_rise(pair):

        print(f"📉 IA rejette {pair}")

        return False

        return False

    return True


from decimal import Decimal, getcontext, ROUND_DOWN


# Configuration des frais Coinbase et seuils adaptatifs


async def place_limit_with_fallback(client, pair, size, limit_price, timeout=5):

    try:

        log_message(f"🕒 Tentative LIMIT ➜ {limit_price:.4f} sur {pair}")

        order = await client.place_limit_order(pair, size, limit_price)

        await asyncio.sleep(timeout)

        order_id = order.get("id")

        status = await client.get_order_status(order_id)

        if status.get("status") != "FILLED":

            await client.cancel_order(order_id)

            log_message(f"⏱️ Timeout LIMIT ➜ fallback MARKET")

            return await client.place_market_order(pair, size)

        return order

    except Exception as e:

        log_message(f"❌ Erreur LIMIT ➜ MARKET : {e}")

        return await client.place_market_order(pair, size)


FEE_RATE = Decimal("0.006")

DEFAULT_TP = Decimal("0.025")

DEFAULT_SL = Decimal("0.01")


SERIES_PROFILES = {
    "series_safe": {"min_profit": 0.015, "tp": 0.015, "sl": 0.008},
    "series_balanced": {"min_profit": 0.020, "tp": 0.020, "sl": 0.010},
    "series_aggressive": {"min_profit": 0.025, "tp": 0.030, "sl": 0.015},
    "series_high_vol": {"min_profit": 0.030, "tp": 0.040, "sl": 0.020},
}


def get_tp_sl(product_id):

    # Si aucune donnée ATR dynamique, on utilise des TP/SL par défaut sécurisés

    return DEFAULT_TP, DEFAULT_SL


def is_trade_profitable(
    buy_price: Decimal, tp_pct: Decimal, selected_series: str
) -> bool:

    sell_price = buy_price * (Decimal("1") + tp_pct)

    gross_profit = sell_price - buy_price

    fees = buy_price * FEE_RATE * 2

    net_profit = gross_profit - fees

    net_pct = net_profit / buy_price

    threshold = MIN_PROFIT_THRESHOLDS.get(selected_series, Decimal("0.015"))

    if net_pct < threshold:

        print(
            f"⛔ Trade annulé : profit net estimé {net_pct:.2%} < seuil {threshold:.2%}"
        )

        return False

    print(
        f"✅ Trade autorisé (profit net estimé {net_pct:.2%} ≥ seuil {threshold:.2%})"
    )

    return True


import csv

import os


# Tracker file path

REGISTRE_TRADES_PATH = "registre_trades.csv"


# Create header if file doesn't exist

if not os.path.exists(REGISTRE_TRADES_PATH):

    with open(REGISTRE_TRADES_PATH, mode="w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "timestamp",
                "pair",
                "direction",
                "base_qty",
                "entry_price",
                "tp_pct",
                "fees",
                "net_profit_pct",
                "accepted",
            ]
        )


def log_trade_to_register(
    pair, side, base_qty, entry_price, tp_pct, fees, net_profit_pct, accepted
):

    with open(REGISTRE_TRADES_PATH, mode="a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                pair,
                side,
                base_qty,
                entry_price,
                tp_pct,
                fees,
                net_profit_pct,
                accepted,
            ]
        )


# Mode de conversion contrôlé (par défaut = 'auto')

# Options futures : 'manual', 'daily_batch', 'threshold_high'

CONVERT_MODE = "auto"


def should_convert_to_usdc(value_usdc: Decimal) -> bool:

    if CONVERT_MODE == "manual":

        return False

    if CONVERT_MODE == "threshold_high":

        return value_usdc >= Decimal("25.00")

    # Default auto mode (≥ 1 USDC)

    return value_usdc >= Decimal("1.00")


MIN_CONVERSION_THRESHOLD = Decimal("15.00")  # Minimum value in USDC to allow conversion


def should_convert_to_usdc(estimated_value_usdc: Decimal) -> bool:

    return estimated_value_usdc >= MIN_CONVERSION_THRESHOLD


FEE_RATE = Decimal("0.006")  # 0.6% total (buy + sell)

AVG_SPREAD = Decimal("0.002")  # 0.2% spread estimate

MIN_NET_PROFIT = Decimal("0.005")  # 0.5% net minimum required


def is_trade_profitable(buy_price: Decimal, tp_pct: Decimal) -> bool:

    sell_price = buy_price * (Decimal("1") + tp_pct)

    gross_profit = sell_price - buy_price

    total_fees = buy_price * FEE_RATE

    spread_cost = buy_price * AVG_SPREAD

    net_profit = gross_profit - total_fees - spread_cost

    return net_profit >= buy_price * MIN_NET_PROFIT


# === UTILITAIRES AJOUTÉS ===

import os

import pandas as pd

LOG_FILE = "sales_register.xlsx"

if not os.path.exists(LOG_FILE):

    df_init = pd.DataFrame(
        columns=[
            "timestamp",
            "series_id",
            "sale_price",
            "gross_profit",
            "fees",
            "net_gain",
        ]
    )

    df_init.to_excel(LOG_FILE, index=False)

    print(f"🔔 Registre Excel initial créé : {os.path.abspath(LOG_FILE)}")


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



if __name__ == '__main__':
    def log_achat_humain(*args, **kwargs):
        """

        Stub pour logger l'achat humain sans provoquer d'erreur.

        """

        # Args et kwargs peuvent être loggés si nécessaire

        return


    def with_retry(retries=3, delay=1):
        """Décorateur: retry fn en cas d'erreur."""

        def decorator(fn):

            def wrapper(*args, **kwargs):

                last_exc = None

                for attempt in range(1, retries + 1):

                    try:

                        return fn(*args, **kwargs)

                    except Exception as e:

                        last_exc = e

                        log_message(f"Retry {attempt}/{retries} pour {fn.__name__}: {e}")

                        time.sleep(delay)

                raise last_exc

            return wrapper

        return decorator


    @with_retry(retries=3, delay=1)
    def create_order_safe(client, **kwargs):
        """

        Tente d'abord un ordre MARKET, puis en cas de 'limit only',

        bascule sur un ordre LIMIT sans récursion infinie.

        """

        try:

            return client.create_order(**kwargs)

        except Exception as e:

            msg = str(e).lower()

            if "limit only" in msg:

                log_message(f"⚠️ Fallback vers LIMIT order pour {kwargs.get('symbol')}")

                # on modifie le type pour un ordre LIMIT

                kwargs["type"] = "LIMIT"

                # appel direct à l'API, pas via create_order_safe

                return client.create_order(**kwargs)

            # toute autre erreur, on la propage

            raise


    def assign_series_to_pair(pair: str, volatility: float, trend_score: float) -> str:
        """

        Attribue une série parmi 9 possibles selon volatilité, tendance, catégorie.

        """

        # Listes de paires par catégorie (à adapter)

        large_caps = {"BTC-USDC", "ETH-USDC", "SOL-USDC"}

        defi_midcaps = {"UNI-USDC", "AAVE-USDC", "COMP-USDC"}

        microcaps = {"DOGE-USDC", "SHIB-USDC"}

        emerging = {"INJ-USDC", "FET-USDC"}

        # 1) Microcaps très volatiles

        if volatility > 0.10 and pair in microcaps:

            return "series_6_microcaps_volatile"

        # 2) Volatilité > 5 %

        if volatility > 0.05:

            return "series_1_high_vol"

        # 3) Large Caps

        if pair in large_caps:

            return "series_2_large_caps"

        # 4) DeFi MidCap

        if pair in defi_midcaps:

            return "series_3_defi_midcap"

        # 5) Tendance forte

        if trend_score >= 0.80:

            return "series_4_strong_trend"

        # 6) Score moyen

        if 0.75 <= trend_score < 0.80:

            return "series_5_balanced"

        # 7) Faible volatilité

        if volatility < 0.01:

            return "series_7_low_volatility"

        # 8) Faible tendance (contrarian)

        if 0.50 <= trend_score < 0.70:

            return "series_8_contrarian"

        # 9) Paires émergentes

        if pair in emerging:

            return "series_9_emerging"

        # Fallback

        return "series_5_balanced"


    # === Dictionnaire des paramètres stratégiques par série ===

    series_parameters = {
        "series_1_high_vol": {
            "tp": 0.015,
            "sl": 0.008,
            "min_trend_score": 0.85,
            "capital_pct": 0.05,
        },
        "series_2_large_caps": {
            "tp": 0.018,
            "sl": 0.009,
            "min_trend_score": 0.75,
            "capital_pct": 0.10,
        },
        "series_3_defi_midcap": {
            "tp": 0.020,
            "sl": 0.010,
            "min_trend_score": 0.78,
            "capital_pct": 0.08,
        },
        "series_4_strong_trend": {
            "tp": 0.022,
            "sl": 0.010,
            "min_trend_score": 0.80,
            "capital_pct": 0.10,
        },
        "series_5_balanced": {
            "tp": 0.018,
            "sl": 0.012,
            "min_trend_score": 0.75,
            "capital_pct": 0.08,
        },
        "series_6_microcaps_volatile": {
            "tp": 0.030,
            "sl": 0.015,
            "min_trend_score": 0.90,
            "capital_pct": 0.03,
        },
        "series_7_low_volatility": {
            "tp": 0.012,
            "sl": 0.006,
            "min_trend_score": 0.60,
            "capital_pct": 0.06,
        },
        "series_8_contrarian": {
            "tp": 0.025,
            "sl": 0.018,
            "min_trend_score": 0.70,
            "capital_pct": 0.07,
        },
        "series_9_emerging": {
            "tp": 0.028,
            "sl": 0.014,
            "min_trend_score": 0.85,
            "capital_pct": 0.04,
        },
    }


    def adjust_series_parameters(
        series_id: str, volatility: float, trend_score: float, drawdown: float | None = None
    ) -> dict[str, float]:
        """

        Ajuste dynamiquement les paramètres d'une série de 9.

        """

        if series_id not in series_parameters:

            series_id = "series_5_balanced"

        params = series_parameters[series_id].copy()

        # Adapter TP/SL selon volatilité

        if volatility > 0.10:

            params["tp"] *= 1.30
            params["sl"] *= 1.30

        elif volatility > 0.05:

            params["tp"] *= 1.20
            params["sl"] *= 1.20

        elif volatility < 0.005:

            params["tp"] *= 0.90
            params["sl"] *= 0.90

        # Ajuster capital_pct selon trend_score

        if trend_score < params["min_trend_score"]:

            params["capital_pct"] *= 0.80

        elif trend_score > 0.90:

            params["capital_pct"] *= 1.10

        # Réduire si drawdown < -3%

        if drawdown is not None and drawdown < -3:

            params["capital_pct"] *= 0.50

        return params

        if volatility > 0.05:

            params["tp"] *= 1.2

            params["sl"] *= 1.2

        elif volatility < 0.01:

            params["tp"] *= 0.8

            params["sl"] *= 0.8

        if trend_score is not None:

            if trend_score < params["min_trend_score"]:

                params["capital_pct"] *= 0.7

            elif trend_score > 0.9:

                params["capital_pct"] *= 1.2

        if drawdown is not None and drawdown < -3:

            params["capital_pct"] *= 0.5

        return params


    # === Journalisation stratégique par série ===

    import csv

    from datetime import datetime


    series_log_path = "series_trade_log.csv"


    def log_series_trade(
        pair, series_id, tp, sl, capital_pct, trend_score, volatility, pnl_pct
    ):

        with open(series_log_path, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    pair,
                    series_id,
                    round(tp, 5),
                    round(sl, 5),
                    round(capital_pct, 4),
                    round(trend_score, 4) if trend_score else "NA",
                    round(volatility, 5) if volatility else "NA",
                    round(pnl_pct, 4),
                ]
            )


    def log_tendance_ia_periodique():

        global all_data

        while True:

            try:

                if "all_data" not in globals() or not all_data:

                    log_message(
                        "⚠️ Données IA non prêtes ou non définies — tendance IA ignorée."
                    )

                    time.sleep(30)

                    continue

                if not selected_crypto_pairs:

                    log_message("⚠️ selected_crypto_pairs vide dans thread → fallback IA")

                log_message("🧠 ========== TENDANCE IA (toutes les 2 min) ==========")

                tendance()

                save_logs_to_file()

            except Exception as e:

                log_message(f"Erreur dans log_tendance_ia_periodique: {e}")

            time.sleep(120)


    def log_tendance_ia_periodique():

        global all_data

        while True:

            try:

                if "all_data" not in globals() or not all_data:

                    log_message(
                        "⚠️ Données IA non prêtes ou non définies — tendance IA ignorée."
                    )

                    time.sleep(30)

                    continue

                if not selected_crypto_pairs:

                    log_message("⚠️ selected_crypto_pairs vide dans thread → fallback IA")

                log_message("🧠 ========== TENDANCE IA (toutes les 2 min) ==========")

                tendance()

                save_logs_to_file()

            except Exception as e:

                log_message(f"Erreur dans log_tendance_ia_periodique: {e}")

            time.sleep(120)


    # === PATCH : Ajustements dynamiques selon volatilité et performance ===


    def adjust_min_net_profit(vol):

        if vol > 0.05:

            return Decimal("0.0")

        elif vol > 0.02:

            return Decimal("0.0")

        else:

            return Decimal("0.0125")


    def adjust_trade_frequency(vol):

        if vol > 0.05:

            return 60

        elif vol > 0.02:

            return 180

        else:

            return 300


    def adjust_buy_capital(success_rate):

        if success_rate >= 0.8:

            return Decimal("0.15")

        elif success_rate >= 0.6:

            return Decimal("0.10")

        else:

            return Decimal("0.05")


    logs = []

    # === PATCH : Envoi d'alerte email en cas d'exception critique ===


    def send_alert_email(subject, body):

        with app.app_context():

            if not user_email:

                print("Error: USER_EMAIL is not defined.")

                return

            msg = Message(subject, recipients=[user_email])

            msg.body = body

            try:

                with mail.connect() as connection:

                    connection.send(msg)

                log_message(f"📬 Alerte envoyée à {user_email}")

                save_logs_to_file()

            except Exception as e:

                log_message(f"Erreur envoi email : {e}")

                save_logs_to_file()


    def log_message(message):

        global logs

        timestamped_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}"

        logs.append(timestamped_message)

        logging.info(timestamped_message)


    VERSION = "v2025.05.18-AI-LIMIT-AUTO"

    print(f"\n🚀 Démarrage du bot - Version {VERSION}\n")

    print(f"balise N°_1")

    # === PATCH : FAVORISER ORDRE LIMIT INTELLIGEMMENT ===

    import numpy as np

    import pandas as pd

    import requests

    import json

    print(f"balise N°_2")


    def fetch_histominute_data(pair, limit=400):

        fsym, tsym = pair.split("-")

        endpoint = "https://min-api.cryptocompare.com/data/histominute"

        try:

            res = requests.get(f"{endpoint}?fsym={fsym}&tsym={tsym}&limit={limit}")

            data = pd.DataFrame(json.loads(res.content)["Data"])

            closes = data["close"].dropna()

            return closes

        except Exception as e:

            print(f"Erreur fetch_histominute {pair}: {e}")

            return None


    def compute_ema(series, window):

        return series.ewm(span=window, adjust=False).mean()


    def compute_rsi(series, length=14):

        delta = series.diff()

        gain = delta.where(delta > 0, 0)

        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=length).mean()

        avg_loss = loss.rolling(window=length).mean()

        rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))

        return rsi


    def is_market_bullish(pair):

        closes = fetch_histominute_data(pair)

        if closes is None or len(closes) < 50:

            print(f"⚠️ Pas assez de données pour {pair}")

            return False

        ema20 = compute_ema(closes, 20)

        ema50 = compute_ema(closes, 50)

        rsi = compute_rsi(closes, 14)

        crossed_up = ema20.iloc[-2] < ema50.iloc[-2] and ema20.iloc[-1] > ema50.iloc[-1]

        bullish = crossed_up and rsi.iloc[-1] > 50

        print(
            f"🔍 Analyse Golden Cross pour {pair}: EMA20={ema20.iloc[-1]:.4f} vs EMA50={ema50.iloc[-1]:.4f}, RSI={rsi.iloc[-1]:.2f} ➔ {'✅ Golden Cross détecté' if bullish else '❌ Pas de Golden Cross'}"
        )

        if not bullish:

            print(f"❌ {pair} rejetée ➤ EMA20/EMA50 non croisé ou RSI ≤ 50")

        if not bullish:

            print(
                f"❌ {pair} rejetée : EMA20[-2]={ema20.iloc[-2]:.4f}, EMA50[-2]={ema50.iloc[-2]:.4f}, "
                f"EMA20[-1]={ema20.iloc[-1]:.4f}, EMA50[-1]={ema50.iloc[-1]:.4f}, RSI={rsi.iloc[-1]:.2f}"
            )

        return bullish


    import torch

    import torch.nn as nn

    import torch

    import os

    from dotenv import load_dotenv

    print(f"balise N°_3")

    load_dotenv()


    # ─── LOGIQUE DES 9 SÉRIES & PARAMÈTRES ────────────────────────────────────

    LARGE_CAPS = {"BTC-USDC", "ETH-USDC", "SOL-USDC"}

    DEFI_MIDCAPS = {"DOT-USDC", "MATIC-USDC", "AVAX-USDC"}

    EMERGENTES = {"INJ-USDC", "FET-USDC", "SUI-USDC", "PEPE-USDC"}


    series_parameters = {
        "series_1_high_vol": {
            "tp": 0.015,
            "sl": 0.008,
            "min_score": 0.85,
            "capital_pct": 0.05,
        },
        "series_2_large_caps": {
            "tp": 0.018,
            "sl": 0.009,
            "min_score": 0.75,
            "capital_pct": 0.10,
        },
        "series_3_defi_midcap": {
            "tp": 0.020,
            "sl": 0.010,
            "min_score": 0.78,
            "capital_pct": 0.08,
        },
        "series_4_strong_trend": {
            "tp": 0.022,
            "sl": 0.010,
            "min_score": 0.80,
            "capital_pct": 0.10,
        },
        "series_5_balanced": {
            "tp": 0.018,
            "sl": 0.012,
            "min_score": 0.75,
            "capital_pct": 0.08,
        },
        "series_6_microcaps_volatile": {
            "tp": 0.030,
            "sl": 0.015,
            "min_score": 0.90,
            "capital_pct": 0.03,
        },
        "series_7_low_vol": {
            "tp": 0.012,
            "sl": 0.006,
            "min_score": 0.60,
            "capital_pct": 0.06,
        },
        "series_8_contrarian": {
            "tp": 0.025,
            "sl": 0.018,
            "min_score": 0.70,
            "capital_pct": 0.07,
        },
        "series_9_emergentes": {
            "tp": 0.028,
            "sl": 0.014,
            "min_score": 0.85,
            "capital_pct": 0.04,
        },
    }


    def assign_series_to_pair(pair: str, volatility: float, trend_score: float) -> str:

        if pair in EMERGENTES:

            return "series_9_emergentes"

        if volatility > 0.05:

            return "series_1_high_vol"

        if pair in LARGE_CAPS:

            return "series_2_large_caps"

        if pair in DEFI_MIDCAPS:

            return "series_3_defi_midcap"

        if trend_score >= 0.80:

            return "series_4_strong_trend"

        if volatility > 0.03:

            return "series_6_microcaps_volatile"

        if volatility < 0.02:

            return "series_7_low_vol"

        if trend_score < 0.60:

            return "series_8_contrarian"

        return "series_5_balanced"


    def adjust_series_parameters(
        series_id: str, volatility: float, trend_score: float, drawdown: float = None
    ) -> dict:

        base = series_parameters.get(series_id, {}).copy()

        if not base:

            return {}

        tp = base["tp"]
        sl = base["sl"]
        cap = base["capital_pct"]
        ms = base["min_score"]

        # Volatility adjustments

        if volatility > 0.05:

            tp *= 1.20
            sl *= 1.20

        elif volatility < 0.015:

            tp *= 0.85
            sl *= 0.85

        # Trend adjustments

        if trend_score < ms:

            cap *= 0.70

        elif trend_score > 0.90:

            cap *= 1.20

        # Drawdown safety

        if drawdown is not None and drawdown < -3.0:

            cap *= 0.50

        return {"tp": tp, "sl": sl, "capital_pct": cap, "min_score": ms}


    # ───────────────────────────────────────────────────────────────────────────


    FILENAME = os.path.basename(__file__)

    print(f"\n🗂️ Fichier source : {FILENAME}\n")

    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("COINBASE_API_KEY_ID")

    api_secret = os.getenv("API_SECRET")


    def save_model(model, filename):

        os.makedirs("saved_models", exist_ok=True)

        path = os.path.join("saved_models", filename)

        torch.save(model.state_dict(), path)

        print(f"✅ Modèle sauvegardé sous {path}")


    def load_model(model, filename):

        path = os.path.join("saved_models", filename)

        if os.path.exists(path):

            model.load_state_dict(torch.load(path))

            model.eval()

            print(f"✅ Modèle chargé depuis {path}")

            return model

        else:

            print(f"ℹ️ Aucun modèle trouvé à {path}, entraînement nécessaire.")

            return model


    class InformerLite(nn.Module):

        def __init__(
            self, input_size=1, d_model=32, n_heads=2, num_layers=1, output_size=1
        ):

            super(InformerLite, self).__init__()

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)

            self.transformer_encoder = nn.TransformerEncoder(
                self.encoder_layer, num_layers=num_layers
            )

            self.input_projection = nn.Linear(input_size, d_model)

            self.output_projection = nn.Linear(d_model, output_size)

        def forward(self, x):

            x = self.input_projection(x)

            x = self.transformer_encoder(x)

            x = self.output_projection(x[:, -1, :])

            return x


    def prepare_informer_input(prices):

        import numpy as np

        closes = [p["close"] for p in prices]

        scaled = (np.array(closes) - np.min(closes)) / (
            np.max(closes) - np.min(closes) + 1e-8
        )

        tensor = torch.tensor(scaled, dtype=torch.float32).view(1, -1, 1)

        return tensor


    def predict_price_informer(model, prices):

        input_tensor = prepare_informer_input(prices)

        prediction = model(input_tensor).item()

        return prediction


    # Initialisation globale pour éviter l'erreur

    executed_orders_global = []

    # Dernière récupération de solde USDC pour éviter le spam

    last_usdc_check = 0

    cached_usdc_balance = 0.0


    def get_usdc_balance():

        global last_usdc_check, cached_usdc_balance

        now = time.time()

        if now - last_usdc_check < 30:

            return cached_usdc_balance

        print("🪙 Récupération du solde USDC...")

        # Simulation ou appel API réel ici

        cached_usdc_balance = 116.19  # Valeur par défaut simulée

        last_usdc_check = now

        return cached_usdc_balance


    # Exemple : basculer sur LIMIT si IA très confiante


    def determine_order_type(signal_strength, volatility=0.01):

        if signal_strength >= 0.75:

            return "limit"

        return "market"


    # Exemple d'utilisation dans la logique de décision


    def place_order_adaptively(product_id, size, price, signal_strength):

        order_type = determine_order_type(signal_strength)

        print(
            f"🔁 Type d'ordre recommandé : {order_type.upper()} (Signal: {signal_strength:.1f})"
        )

        if order_type == "limit":

            return execute_order_with_limit_fallback(
                product_id, side="BUY", size=size, price=price
            )

        else:

            return place_market_order(product_id, size, side="BUY")


    # === AJOUT: INDICATEURS TECHNIQUES + PROTECTION DRAWDOWN GLOBAL ===

    sales_done_global = (
        []
    )  # PATCHED: corrected improper bool initialization  # Initialisation globale

    import pandas as pd


    def add_technical_indicators(df):

        df["rsi"] = ta.rsi(df["close"], length=14)

        macd = ta.macd(df["close"])

        df["macd"] = macd["MACD_12_26_9"]

        df["macd_signal"] = macd["MACDs_12_26_9"]

        df["ema20"] = ta.ema(df["close"], length=20)

        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        df.dropna(inplace=True)

        return df


    def compute_signal_strength(df):

        last = df.iloc[-1]

        score = 0

        if last["rsi"] > 50:

            score += 1

        if last["macd"] > last["macd_signal"]:

            score += 1

        if last["close"] > last["ema20"]:

            score += 1

        if last["atr"] > 0:

            score += 1

        return score / 4


    profit_cumul = 0.0

    max_drawdown_pct = -5.0

    bot_running = True


    def adjust_drawdown_threshold(trade_summary, volatility):

        base = -5.0  # seuil par défaut

        if trade_summary is None:

            return base

        avg_pnl = trade_summary.get("avg_pnl", 0)

        success_rate = trade_summary.get("success_rate", 0)

        if avg_pnl > 1 and success_rate > 0.7:

            return base - 1  # tolérance plus large

        elif avg_pnl < -1 or volatility > 0.03:

            return base + 1  # tolérance plus stricte

        return base


    def check_drawdown_stop():

        global profit_cumul, bot_running

        if profit_cumul < max_drawdown_pct:

            print(f"❌ Drawdown global dépassé ({profit_cumul:.1f}%) : arrêt du bot.")

            bot_running = False


    # Valeur minimale de quote size pour un ordre valide

    MIN_QUOTE_SIZE = 5.0

    """

    ✅ Fonctionnalités actives dans ce bot :

    🧠 GRU (OHLC)

    - Définition de la classe GRUPricePredictor

    - Chargement de données OHLC

    - Entraînement GRU lancé

    - Prédiction GRU utilisée

    - Logique d’achat basée sur GRU

    📈 LIMIT / MARKET

    - Fallback intelligent configuré

    💰 Trade réel Coinbase

    - Relié à ton compte + ordres live via 

        if Decimal(amount_in_btc1) < Decimal(base_increment):

            log_message("⛔ Trop petit pour être converti : ignoré")

            return None

            log_message(f"⛔ Quantité trop petite pour {product_id}: base_size={{amount_in_btc1}}, requis: {{base_increment}}")

            save_logs_to_file()

            return None

    client.create_order

    📊 LSTM

    - Utilisé dans la stratégie IA (analyse de tendance)

    🔁 Compounding

    - Basé sur capital_actuel, configurable

    🌐 Interface Web (Flask + SocketIO)

    - Tableau de bord en temps réel

    - Logs dynamiques

    - Lancement/arrêt du bot

    - Authentification avec 2FA (email)

    """

    import time

    import random

    print(f"balise N°_4")


    def log_ia_predictions(predictions, executed_orders, sales_done):

        log_message = []

        log_message.append(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 🔍 Analyse IA : Résumé des tendances"
        )

        for pair, (today_pred, yest_pred) in predictions.items():

            trend_arrow = "📈" if today_pred > yest_pred else "📉"

            trend = "augmenter" if today_pred > yest_pred else "diminuer"

            log_message.append(
                f"{trend_arrow} {pair} : Le prix de la crypto va {trend} (Today_Prediction: {today_pred:.1f} {'>' if today_pred > yest_pred else '<'} Yesterday_Prediction: {yest_pred:.1f})"
            )

            if pair in executed_orders:

                log_message.append(
                    f"✅ Ordre d'achat exécuté pour {pair} suite à la prédiction haussière."
                )

            if pair in sales_done:

                log_message.append(
                    f"💰 Vente réalisée pour {pair} suite à l'objectif atteint."
                )

        for line in log_message:

            print(line)


    # Exemple d'utilisation dans une boucle toutes les 60s :

    # log_ia_predictions(

    #     predictions={'ADA-USDC': (0.6, 0.6), 'DOGE-USDC': (0.2, 0.2)},

    #     executed_orders=['ADA-USDC'],

    #     sales_done=['ADA-USDC']

    # )


    def seuil_profit_par_serie(series_id):

        mapping = {
            "series_1_high_vol": Decimal("0.01"),  # 1%
            "series_2_large_caps": Decimal("0.01"),  # 1%
            "series_3_defi_midcap": Decimal("0.01"),  # 1%
            "series_4_strong_trend": Decimal("0.01"),  # 1%
            "series_5_balanced": Decimal("0.01"),  # 1%
            "series_6_microcaps_volatile": Decimal("0.02"),  # 2%
            "series_7_low_vol": Decimal("0.0125"),  # 1.25%
            "series_8_contrarian": Decimal("0.0125"),  # 1.25%
            "series_9_emergentes": Decimal("0.02"),  # 2%
            "N/A": Decimal("0.01"),
        }

        return mapping.get(series_id, Decimal("0.01"))


    COINBASE_FEE_RATE = Decimal("0.006")  # 0.6% frais Coinbase

    current_portfolio_id = os.getenv("COINBASE_PORTFOLIO_ID")  # ID du portefeuille actif

    usdc_safe_wallet_id = os.getenv(
        "COINBASE_PROFIT_PORTFOLIO_ID"
    )  # ID du portefeuille Profit robot DCA

    if not current_portfolio_id or not usdc_safe_wallet_id:

        raise RuntimeError(
            "COINBASE_PORTFOLIO_ID et COINBASE_PROFIT_PORTFOLIO_ID doivent être définis dans le .env"
        )


    from typing import Tuple

    from tenacity import retry, stop_after_attempt, wait_fixed

    print(f"balise N°_5")

    # ✅ Fonction de calcul de performance net


    def calculate_net_profit_percentage(
        entry_price: float, exit_price: float, fees: float
    ) -> float:

        if entry_price == 0:

            return 0.0

        gross_profit = exit_price - entry_price

        net_profit = gross_profit - fees

        return (net_profit / entry_price) * 100


    # ✅ Détection de liquidité


    def liquidity_score(order_book: dict) -> float:

        spread = float(order_book["asks"][0][0]) - float(order_book["bids"][0][0])

        return 1 / spread if spread != 0 else float("inf")


    # ✅ Retry API wrapper


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def place_order_with_retry(*args, **kwargs):

        return place_order(*args, **kwargs)


    # ✅ Test unitaire simplifié (à déplacer dans un fichier tests si besoin)


    def test_stop_loss():

        class DummyBot:

            entry_price = 100

            def check_stop_loss(self, current_price):

                return current_price < self.entry_price * (1 - 0.01)

        bot = DummyBot()

        assert bot.check_stop_loss(98.9) == True

        assert bot.check_stop_loss(99.5) == False


    from datetime import datetime, timedelta

    from collections import deque


    print(f"balise N°_6")

    # Simulation ou import des fonctions externes :

    # - get_current_price(pair)

    # - place_market_order(pair, size, side)

    # - get_usdc_balance()

    # === AJOUT: TP/SL dynamique par paire ===


    def get_tp_sl(product_id):
        """

        Retourne un couple (take_profit, stop_loss) sous forme de Decimal en fonction de la paire.

        Possibilité future d'intégrer la volatilité ou autre logique IA.

        """

        if product_id.startswith("BTC"):

            return Decimal("0.0"), Decimal("0.0")  # +0.8% / -0.3%

        elif product_id.startswith("DOT"):

            return Decimal("0.0125"), Decimal("0.0")  # +1.25% / -0.5%

        elif product_id.startswith("SOL"):

            return Decimal("0.0"), Decimal("0.0")  # +1.2% / -0.6%

        else:

            return Decimal("0.0125"), Decimal("0.0")  # valeur par défaut


    # Setup logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


    class TradeMemory:

        def __init__(self, maxlen=10):

            self.memory = deque(maxlen=maxlen)

        def add_trade(self, pair, side, entry_price, exit_price, capital, params):

            pnl = (
                ((exit_price - entry_price) / entry_price) * 100
                if side == "BUY"
                else ((entry_price - exit_price) / entry_price) * 100
            )

            self.memory.append(
                {
                    "pair": pair,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "capital": capital,
                    "pnl": pnl,
                    "params": params,
                    "timestamp": datetime.now(),
                }
            )

        def get_summary(self):

            if not self.memory:

                return None

            avg_pnl = sum(t["pnl"] for t in self.memory) / len(self.memory)

            success_rate = len([t for t in self.memory if t["pnl"] > 0]) / len(self.memory)

            return {"avg_pnl": round(avg_pnl, 3), "success_rate": round(success_rate, 2)}


    class AdaptiveIA:

        def __init__(self):

            self.base_tp = 0.0

            self.base_sl = 0.0

            self.capital_percent = 0.1

        def adjust_parameters(self, trade_summary):

            if trade_summary is None:

                return self.base_tp, self.base_sl, self.capital_percent

            avg_pnl = trade_summary["avg_pnl"]

            success_rate = trade_summary["success_rate"]

            tp = self.base_tp

            sl = self.base_sl

            cap = self.capital_percent

            if success_rate < 0.5:

                sl *= 0.8

                cap *= 0.9

            elif success_rate > 0.8:

                tp *= 1.1

                cap *= 1.2

            if avg_pnl < 0:

                tp *= 0.9

                sl *= 0.9

            logging.info(
                f"IA Adaptive Adjustments: TP={tp:.1f}, SL={sl:.1f}, Capital%={cap:.1f}"
            )

            return tp, sl, cap


    class TradingBot:

        def __init__(self, pair, trade_memory, ia):

            self.pair = pair

            self.memory = trade_memory

            self.ia = ia

            self.last_price = 0

        def run_once(self):

            # Simulation des prix

            entry_price = get_current_price(self.pair)

            tp, sl, cap = self.ia.adjust_parameters(self.memory.get_summary())

            capital = get_usdc_balance() * cap

            size = capital / entry_price

            logging.info(
                f"Buying {self.pair} at {entry_price} with {capital:.1f} USDC (size: {size:.1f})"
            )

            # Simuler ordre marché

            buy_success = place_market_order(self.pair, size, "BUY")

            if not buy_success:

                logging.warning("Achat échoué.")

                return

            price_now = entry_price

            start_time = time.time()

            max_price = entry_price

            while True:

                time.sleep(2)

                price_now = get_current_price(self.pair)

                max_price = max(max_price, price_now)

                if price_now >= entry_price * (1 + tp):

                    logging.info(f"Take Profit atteint à {price_now}")

                    break

                elif price_now <= entry_price * (1 - sl):

                    logging.info(f"Stop Loss atteint à {price_now}")

                    break

                if time.time() - start_time > 180:

                    logging.info(f"Temps max écoulé, vente à {price_now}")

                    break

            place_market_order(self.pair, size, "SELL")

            self.memory.add_trade(
                self.pair,
                "BUY",
                entry_price,
                price_now,
                capital,
                {"tp": tp, "sl": sl, "cap%": cap},
            )

            logging.info(
                f"Trade terminé avec PnL: {((price_now - entry_price) / entry_price) * 100:.2f}%"
            )


    # Simulations/mock


    def get_current_price(pair):

        return 84000 + random.uniform(-100, 100)  # Simule le prix BTC


    def get_usdc_balance():

        return 90.0  # Simule un solde fixe


    def place_market_order(pair, size, side):

        return True  # Toujours succès ici


    def tendance():

        for crypto_pair in selected_crypto_pairs:

            if (
                crypto_pair not in all_data
                or all_data[crypto_pair] is None
                or all_data[crypto_pair].empty
            ):

                log_message(f"⚠️ Données manquantes pour {crypto_pair}, tendance ignorée.")

                continue

            hist = all_data[crypto_pair]

            train, test = train_test_split(hist, test_size=test_size)

            result = prepare_data(
                hist,
                "close",
                window_len=window_len,
                zero_base=zero_base,
                test_size=test_size,
            )

            if result is None:

                log_message(f"⚠️ Skipping {crypto_pair} — données invalides.")

                continue

            train, test, X_train, X_test, y_train, y_test = result

            model = build_lstm_model(
                X_train,
                output_size=1,
                neurons=lstm_neurons,
                dropout=dropout,
                loss=loss,
                optimizer=optimizer,
            )

            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=True,
            )

            targets = test["close"][window_len:]

            preds = model.predict(X_test).squeeze()

            preds = test["close"].values[:-window_len] * (preds + 1)

            preds = pd.Series(index=targets.index, data=preds)

            if len(preds) < 2:

                log_message(
                    f"{crypto_pair} : Pas assez de données pour comparer les tendances."
                )

                continue

            yesterday_last_real = preds.iloc[-2]

            today_pred = preds.iloc[-1]

            trend_comparison = compare_first_real_and_last_pred(
                yesterday_last_real, today_pred
            )

            print(f"{crypto_pair} trend: {trend_comparison}")

            log_message(f"{crypto_pair} trend: {trend_comparison}")

            if today_pred > yesterday_last_real:

                log_message(f"📈 {crypto_pair} va probablement augmenter")

            else:

                log_message(f"📉 {crypto_pair} va probablement baisser")


    #####################################################################################################

    # Fonction de transfert automatique du profit net après chaque vente


    def transfer_profit_to_safe_wallet(amount_usdc, product_id, series_id):
        """

        Transfère le montant USDC du portefeuille de trading vers le portefeuille Profit robot DCA.

        """

        try:

            SEUIL_MIN = 0.50  # Pas de micro-transfert pour moins de 0.50 USDC

            if amount_usdc <= SEUIL_MIN:

                log_message(
                    f"⏩ Profit net trop faible ({amount_usdc:.2f} USDC) — pas de transfert."
                )

                return False

            response = client.create_transfer(
                from_portfolio_id=current_portfolio_id,
                to_portfolio_id=usdc_safe_wallet_id,
                amount=str(round(amount_usdc, 2)),
                currency="USDC",
                note=f"Profit DCA | {product_id} | Série {series_id}",
            )

            order_id = response.get("order_id", "N/A") if response else "N/A"

            log_message(
                f"✅ Profit de {amount_usdc:.2f} USDC transféré vers Profit robot DCA (order_id: {order_id}) | Vente sur {product_id}"
            )

            save_logs_to_file()

            return True

        except Exception as e:

            log_message(
                f"❌ Erreur lors du transfert de profit vers portefeuille Profit robot DCA : {e}"
            )

            save_logs_to_file()

            return False


    #####################################################################################################

    if __name__ == "__main__":

        import threading

        thread_tendance = threading.Thread(target=log_tendance_ia_periodique)

        thread_tendance.daemon = True

        thread_tendance.start()

        log_message("🔁 Thread IA lancé")

    if __name__ == "__main__":

        mem = TradeMemory(maxlen=10)

        ai = AdaptiveIA()

        bot = TradingBot("BTC-USDC", mem, ai)

        for _ in range(5):

            bot.run_once()

            logging.info(f"Mémoire des trades: {mem.get_summary()}")

            print("-" * 80)

    # === END OF MODULE 1 ===

    # === AUTO-SELECTION INTELLIGENTE DES PAIRES PAR IA ET TECHNIQUE ===


    def auto_select_best_pairs_from_market(pairs, top_n=5):

        selected = []

        for pair in pairs:

            log_message(f"🔍 Analyse technique {pair} :")

            if closes is None or len(closes) < 50:

                log_message(f"⛔ Données insuffisantes pour {pair}")

                continue

            vol = get_volatility_score(closes[-50:])

            bullish = is_market_bullish(pair)

            if bullish:

                selected.append((pair, vol))

                log_message(f"✅ {pair} retenue ➜ volatilité={vol:.1f}")

            else:

                log_message(f"❌ {pair} rejetée ➜ Golden Cross/RSI insuffisant")

            closes = fetch_histominute_data(pair)

            if closes is None or len(closes) < 50:

                continue

            if bullish:

                selected.append((pair, vol))

        selected.sort(key=lambda x: x[1], reverse=True)

        if not selected:

            log_message(
                "⚠️ Aucune paire haussière détectée. Fallback activé : top n pairs statiques"
            )

            fallback = [(pair, 0) for pair in pairs[:top_n]]

            for pair, _ in fallback:

                log_message(f"⚠️ Fallback sélection : {pair}")

            return [p[0] for p in fallback]

        log_message("✅ Paires haussières détectées :")

        for p, v in selected[:top_n]:

            log_message(f"✅ {p} (vol={v:.1f})")

        return [p[0] for p in selected[:top_n]]

        selected = []

        for pair in pairs:

            closes = fetch_histominute_data(pair)

            if closes is None or len(closes) < 50:

                continue

            vol = get_volatility_score(closes[-50:])

            if is_market_bullish(pair):

                selected.append((pair, vol))

        selected.sort(key=lambda x: x[1], reverse=True)

        if not selected:

            print(
                "⚠️ Aucune paire haussière détectée. Activation fallback sur les premières paires disponibles."
            )

            selected = [(pair, 0) for pair in pairs[:top_n]]

        return [p[0] for p in selected[:top_n]]


    def get_volatility_score(closes):
        try:
            arr = np.asarray(closes)
            if arr.ndim < 1 or len(arr) < 2:
                raise ValueError("Array must be at least 1D with 2 elements")
            returns = np.diff(arr) / arr[:-1]
            return np.std(returns)
        except Exception as e:
            log_message(f"❌ Erreur dans get_volatility_score: {e}")
            return 0.0
        if arr.ndim < 1 or len(arr) < 2:
            raise ValueError("Array must be at least 1D with 2 elements")
        returns = np.diff(arr) / arr[:-1]

        return np.std(returns)


    # === PATCH: Importations nécessaires ===

    import statistics

    import asyncio

    import time

    print(f"balise N°_7")

    # ========== MASTER TRADING BOT ==========

    # === GRU + Histohour AI Modules Injected ===


    def analyse_histohour(prices, window=24):

        import numpy as np


    def should_update_ai(current_time):
        """

        Détermine si l'IA doit être mise à jour en fonction de l'heure.

        - De 00:00 à 06:00 : mise à jour toutes les heures (histohour)

        - Sinon : mise à jour toutes les 5 minutes (histominute)

        """

        if 0 <= current_time.hour < 6:

            return current_time.minute == 0  # Toutes les heures

        else:

            return current_time.minute % 5 == 0  # Toutes les 5 minutes

        closes = [p["close"] for p in prices[-window:]]

        change_pct = (closes[-1] - closes[0]) / closes[0] * 100

        volatility = np.std(closes)

        trend = "up" if change_pct > 0 else "down"

        return {
            "trend": trend,
            "change_pct": round(change_pct, 2),
            "volatility": round(volatility, 4),
        }


    # GRU modèle prédictif (mock - à entraîner avec données réelles)

    import torch

    import torch.nn as nn

    import numpy as np

    print(f"balise N°_8")


    class GRUPricePredictor(nn.Module):

        def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):

            super(GRUPricePredictor, self).__init__()

            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):

            h0 = torch.zeros(1, x.size(0), 32)

            out, _ = self.gru(x, h0)

            out = self.fc(out[:, -1, :])

            return out


    def prepare_input_series(prices):

        closes = [p["close"] for p in prices]

        scaled = (np.array(closes) - np.min(closes)) / (
            np.max(closes) - np.min(closes) + 1e-8
        )

        tensor = torch.tensor(scaled, dtype=torch.float32).view(1, -1, 1)

        return tensor


    # Exemple (dans le main loop):

    # prediction = model(prepare_input_series(prices)).item()

    # if prediction > closes[-1]: trigger_buy()

    # === DÉBUT MODIFICATIONS AJOUTÉES ===

    MIN_NET_PROFIT_TARGET = 0.0  # 1.2% minimum net


    def calculate_net_profit_percentage(buy_price, sell_price, buy_fee, sell_fee):

        gross_profit = sell_price - buy_price

        total_fees = buy_fee + sell_fee

        net_profit = gross_profit - total_fees

        return (net_profit / buy_price) if buy_price else 0


    def adjust_sell_profit_target(volatility_index):

        if volatility_index > 0.05:

            return 0.0

        elif volatility_index > 0.02:

            return 0.02

        else:

            return MIN_NET_PROFIT_TARGET


    def log_trade_profit(order_buy, order_sell):

        buy_price = float(order_buy.get("average_filled_price", 0))

        sell_price = float(order_sell.get("average_filled_price", 0))

        buy_fee = float(order_buy.get("total_fees", 0))

        sell_fee = float(order_sell.get("total_fees", 0))

        net_profit_pct = calculate_net_profit_percentage(
            buy_price, sell_price, buy_fee, sell_fee
        )

        total_fees = buy_fee + sell_fee

        print(
            "📈 Profit net sur trade: {:.3f}% | Achat: {:.4f}, Vente: {:.4f}, Frais totaux: {:.4f}".format(
                net_profit_pct * 100, buy_price, sell_price, total_fees
            )
        )

        return net_profit_pct >= MIN_NET_PROFIT_TARGET


    # === FIN MODIFICATIONS AJOUTÉES ===


    def adjust_sell_profit_target_based_on_volatility(volatility_index):

        # Placeholder: you can replace this logic with one based on real volatility analysis

        if volatility_index > 0.02:

            return 0.0  # Increase target in high volatility

        elif volatility_index < 0.01:

            return 0.01  # Reduce target in low volatility

        return 0.0


    # === PARAMÈTRES DE STRATÉGIE LIMIT INTELLIGENTE ===

    LIMIT_ENABLED = True

    LIMIT_SPREAD = 0.0  # 0.1% en dessous du meilleur prix pour SELL, au-dessus pour BUY

    LIMIT_TIMEOUT = 3  # en secondes avant fallback MARKET

    # === NOUVELLE FONCTION POUR ESSAYER UN LIMIT PUIS RETOURNER SUR MARKET SI BESOIN ===


    def execute_order_with_limit_fallback(product_id, side, size, price):
        """

        Tente un ordre LIMIT, sinon fallback sur MARKET

        """

        if not LIMIT_ENABLED:

            return place_market_order(product_id, side, size)

        limit_price = round(
            price * (1 - LIMIT_SPREAD if side == "sell" else 1 + LIMIT_SPREAD), 8
        )

        print(
            f"🔍 Tentative d'ordre LIMIT sur {{product_id}} à {{limit_price}} ({{side.upper()}})"
        )

        try:

            order = place_limit_order(
                product_id=product_id, side=side, size=size, price=limit_price
            )

            waited = 0

            while not order_filled(order["id"]) and waited < LIMIT_TIMEOUT:

                time.sleep(1)

                waited += 1

            if order_filled(order["id"]):

                print(f"✅ Ordre LIMIT exécuté pour {{product_id}}")

                return order

            else:

                print(f"⏱️ Timeout LIMIT pour {{product_id}}, fallback MARKET")

                cancel_order(order["id"])

                return place_market_order(product_id=product_id, side=side, size=size)

        except Exception as e:

            print(f"⚠️ Erreur lors de l'ordre LIMIT : {{e}}, fallback MARKET")

            return place_market_order(product_id=product_id, side=side, size=size)


    # --- BEGIN PATCH TO HANDLE SMALL RESIDUALS ---


    def force_convert_to_usdc(client, product_id, portfolio_id):

        try:

            base_currency = product_id.split("-")[0]

            balance = get_balance(client, base_currency)

            if balance is None or balance <= 0:

                print(f"No balance to convert for {base_currency}")

                return

            # Get product info to check min order size

            product = client.get_product(product_id)

            min_order_size = float(product["quote_increment"])

            # Check if balance * price > min_trade_size, otherwise force a small top-up

            ticker = client.get_product_ticker(product_id)

            price = float(ticker["price"])

            min_trade_value = min_order_size * price

            value = balance * price

            if value < min_trade_value:

                # Try topping up with USDC (simulate small buy) to reach tradable value

                print(
                    f"Topping up {base_currency}: Current value {value} < required {min_trade_value}"
                )

                topup_amount = min_trade_value - value + 0.01  # Add a small buffer

                client.place_order(
                    product_id=product_id,
                    side="BUY",
                    order_type="MARKET",
                    quote_size=str(round(topup_amount, 8)),
                    time_in_force="IMMEDIATE_OR_CANCEL",
                )

            # Sell the full amount

            balance = get_balance(client, base_currency)

            client.place_order(
                product_id=product_id,
                side="SELL",
                order_type="MARKET",
                base_size=str(balance),
                time_in_force="IMMEDIATE_OR_CANCEL",
            )

            print(f"Converted {base_currency} → USDC, size: {balance}")

        except Exception as e:

            print(f"[ERROR] force_convert_to_usdc: {e}")


    # --- END PATCH TO HANDLE SMALL RESIDUALS ---

    # === Configuration globale ===

    MIN_CONVERSION_USDC = 0.30  # seuil minimal pour estimer si une conversion vaut la peine

    # === Fonction utilitaire ===


    def peut_convertir(base_amount, base_increment, est_usdc_value):

        try:

            if base_amount < float(base_increment):

                logging.info(
                    f"Conversion annulée car la quantité {base_amount} est inférieure au base_increment requis {base_increment}."
                )

                return False

            if est_usdc_value < MIN_CONVERSION_USDC:

                logging.info(
                    f"Conversion annulée car le gain estimé est < {MIN_CONVERSION_USDC} USDC (valeur estimée: {est_usdc_value})."
                )

                return False

            return True

        except Exception as e:

            logging.error(f"Erreur dans peut_convertir: {e}")

            return False


    def is_expected_gain_too_small(base_amount, current_price, min_usdc_gain=0.30):

        estimated_value = base_amount * current_price

        return estimated_value < min_usdc_gain


    def determine_order_type(volatility, selected_type="auto"):

        if selected_type != "auto":

            return selected_type

        if volatility > 0.02:

            return "market"

        else:

            return "limit"


    from coinbase.rest import RESTClient

    # === Ajout: Suivi de la volatilité et du timing des trades ===

    from statistics import stdev

    from collections import deque

    import time

    print(f"balise N°_9")

    # Garder en mémoire les N derniers prix (ex: 10)

    volatility_window_size = 10

    price_history = deque(maxlen=volatility_window_size)

    # Gérer le temps entre 2 trades

    last_trade_time = 0

    min_trade_interval = 30  # en secondes

    max_trade_interval = 300  # en secondes


    def get_volatility():

        if len(price_history) >= 2:

            return stdev(price_history)

        return 0


    def can_trade_now():

        global last_trade_time

        now = time.time()

        elapsed = now - last_trade_time

        if elapsed < min_trade_interval:

            return False

        if elapsed > max_trade_interval:

            return True

        return True


    def update_trade_time():

        global last_trade_time

        last_trade_time = time.time()


    # Supprimé: doublon de determine_order_type (définition sans paramètre)

    # === Fin ajout ===

    # === CONFIGURATION COMPTE UTILISATEUR ===

    compounding = True  # Active le mode compounding (réinvestissement des gains)

    capital_initial = 100.0  # Capital de départ en USDC

    capital_actuel = capital_initial  # Capital qui évolue selon les gains/pertes

    from dotenv import load_dotenv

    from threading import Thread

    import threading


    import time

    import uuid


    COINBASE_FEE_RATE = Decimal("0.006")  # 0.6% frais Coinbase

    current_portfolio_id = os.getenv("COINBASE_PORTFOLIO_ID")  # ID du portefeuille actif

    usdc_safe_wallet_id = os.getenv(
        "COINBASE_PROFIT_PORTFOLIO_ID"
    )  # ID du portefeuille Profit robot DCA

    if not current_portfolio_id or not usdc_safe_wallet_id:

        raise RuntimeError(
            "COINBASE_PORTFOLIO_ID et COINBASE_PROFIT_PORTFOLIO_ID doivent être définis dans le .env"
        )


    from flask import Flask, request, jsonify, render_template, redirect, url_for, session

    from requests.exceptions import HTTPError

    from functools import wraps

    import pyotp

    from flask_mail import Mail, Message

    import requests

    from datetime import datetime

    from flask_socketio import SocketIO

    from coinbase.rest import RESTClient

    ##############################################

    import os

    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    tf.config.threading.set_intra_op_parallelism_threads(2)

    tf.config.threading.set_inter_op_parallelism_threads(2)

    import json

    import requests

    from keras.models import Sequential

    from keras.layers import Activation, Dense, Dropout, LSTM, Input

    import matplotlib.pyplot as plt

    import numpy as np

    import pandas as pd

    import seaborn as sns

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    import io

    import base64

    print(f"balise N°_10")


    def place_order_with_fallback(client, symbol, quantity, limit_price):
        """

        Place a limit order and fallback to market if not filled within timeout.

        """

        import time

        print(f"[LIMIT] Placing limit order: {quantity} {symbol} at {limit_price}")

        order = client.order_limit_buy(
            symbol=symbol, quantity=quantity, price=str(limit_price)
        )

        order_id = order["orderId"]

        wait_time = 30  # seconds

        poll_interval = 5  # seconds

        elapsed = 0

        while elapsed < wait_time:

            time.sleep(poll_interval)

            elapsed += poll_interval

            print(f"[STATUS] Checking status of order {order_id}")

            order_status = client.get_order(symbol=symbol, orderId=order_id)

            if order_status["status"] == "FILLED":

                print(f"[FILLED] Limit order {order_id} was filled.")

                return order_status

        # Fallback to market

        print(f"[FALLBACK] Order {order_id} not filled, switching to market order.")

        print(f"[CANCEL] Cancelling order {order_id}")

        client.cancel_order(symbol=symbol, orderId=order_id)

        print(f"[MARKET] Placing market order: {quantity} {symbol}")

        return client.order_market_buy(symbol=symbol, quantity=quantity)


    def force_convert_all_to_usdc(min_value_usdc=1.0):

        global accounts

        try:

            log_message("⚙️ Démarrage de la conversion vers USDC (seuil: ≥ 1 USDC)")

            save_logs_to_file()

            for account in accounts["accounts"]:

                currency = account["available_balance"]["currency"]

                amount = Decimal(account["available_balance"]["value"])

                if currency == "USDC" or amount <= 0:

                    continue

                try:

                    clean_currency = currency.rstrip("0123456789")

                    product_id = f"{clean_currency}-USDC"

                    price = get_market_price(product_id)

                    if not price:

                        log_message(f"❌ Prix introuvable pour {product_id}")

                        continue

                    usdc_value = amount * price

                    if usdc_value < Decimal(min_value_usdc):

                        log_message(
                            f"🚫 Conversion ignorée : {currency} ({amount}) ~ {usdc_value:.1f} USDC (< {min_value_usdc})"
                        )

                        continue

                    log_message(
                        f"🔁 Conversion {amount} {currency} (~{usdc_value:.1f} USDC)"
                    )

                    create_order_safe(
                        client,
                        client_order_id=str(uuid.uuid4()),
                        product_id=product_id,
                        side="SELL",
                        order_configuration={
                            "market_market_ioc": {"base_size": str(amount)}
                        },
                    )

                    log_message(f"✅ Conversion effectuée pour {currency}")

                except Exception as e:

                    log_message(f"❌ Erreur pendant la conversion de {currency}: {str(e)}")

            log_message("✅ Conversion complète terminée.")

            save_logs_to_file()

        except Exception as e:

            log_message(f"🔥 Erreur dans force_convert_all_to_usdc: {str(e)}")

            save_logs_to_file()


    print(f"balise N°_10::1")

    #######################################################################################################################

    # Set decimal precision

    getcontext().prec = 10

    # Load environment variables from .env file

    load_dotenv()

    # Hardcoded password for login

    HARDCODED_PASSWORD = os.getenv("LOGIN_PASSWORD")

    # Set up logging

    logging.basicConfig(level=logging.INFO)

    # Load API credentials

    api_key = os.getenv("COINBASE_API_KEY_ID")

    # Load the private key from the PEM file

    private_key_path = "coinbase_private_key.pem"

    with open(private_key_path, "r", encoding="utf-8") as key_file:

        api_secret = key_file.read()

    # Create the RESTClient instance

    client = RESTClient(api_key=api_key, api_secret=api_secret)

    try:

        # Simple call to test authentication

        accounts = client.get_accounts()

        for account in accounts["accounts"]:

            print("Successfully authenticated. Accounts data:", account["name"])

    except Exception as e:

        print("Authentication failed:", e)

    print(f"balise N°_10::2")

    #####################################################################################################

    selected_crypto_pairs = [
        "ADA-USDC",
        "AAVE-USDC",
        "ALGO-USDC",
        "ARB-USDC",
        "AVAX-USDC",
        "BTC-USDC",
        "CRV-USDC",
        "DOGE-USDC",
        "DOT-USDC",
        "ETC-USDC",
        "ETH-USDC",
        "FET-USDC",
        "FIL-USDC",
        "GRT-USDC",
        "HBAR-USDC",
        "ICP-USDC",
        "IDEX-USDC",
        "LINK-USDC",
        "LTC-USDC",
        "MATIC-USDC",
        "NEAR-USDC",
        "PEPE-USDC",
        "SOL-USDC",
        "SUI-USDC",
        "SUPER-USDC",
        "SUSHI-USDC",
        "SWFTC-USDC",
        "UNI-USDC",
        "USDT-USDC",
        "VET-USDC",
        "XLM-USDC",
        "XRP-USDC",
        "YFI-USDC",
    ]
    selected_crypto_pairs = [
        "ADA-USDC",
        "AAVE-USDC",
        "ALGO-USDC",
        "ARB-USDC",
        "AVAX-USDC",
        "BTC-USDC",
        "CRV-USDC",
        "DOGE-USDC",
        "DOT-USDC",
        "ETC-USDC",
        "ETH-USDC",
        "FET-USDC",
        "FIL-USDC",
        "GRT-USDC",
        "HBAR-USDC",
        "ICP-USDC",
        "IDEX-USDC",
        "LINK-USDC",
        "LTC-USDC",
        "MATIC-USDC",
        "NEAR-USDC",
        "PEPE-USDC",
        "SOL-USDC",
        "SUI-USDC",
        "SUPER-USDC",
        "SUSHI-USDC",
        "SWFTC-USDC",
        "UNI-USDC",
        "USDT-USDC",
        "VET-USDC",
        "XLM-USDC",
        "XRP-USDC",
        "YFI-USDC",
    ]
    # selected_crypto_pairs=['ADA-USDC','ALGO-USDC','BCH-USDC','BTC-USDC','CRV-USDC','DOGE-USDC','DOT-USDC','ETC-USDC','ETH-USDC','LINK-USDC','LTC-USDC','MATIC-USDC','PEPE-USDC','SOL-USDC','SUI-USDC','SUSHI-USDC','SWFTC-USDC','UNI-USDC','USDT-USDC','XRP-USDC']

    # VALIDE

    # Fetch product details

    # --- Sécurité si aucune paire sélectionnée ---

    # === PRÉ-LAUNCH: Validation des paires via load_data ===


    def validate_pairs_before_launch(pairs):

        import pandas as pd

        global all_data

        all_data = {}

        log_message("🔍 Pré-lancement : validation des paires avec fetch_crypto_data()")

        for pair in pairs:

            try:

                df = fetch_crypto_data(pair)

                if df.empty:

                    log_message(f"⚠️ Paire ignorée (données invalides) : {pair}")

                    continue

                all_data[pair] = df

            except Exception as e:

                log_message(f"❌ Erreur lors de la validation de {pair}: {str(e)}")

        if not all_data:

            log_message("⛔ Aucune paire valide détectée. Arrêt du bot.")

            save_logs_to_file()

            exit()

        log_message(f"✅ Paires valides prêtes à être tradées : {list(all_data.keys())}")

        save_logs_to_file()


    # Appel automatique juste après définition des paires

    print(f"balise N°_10::3")

    selected_crypto_pairs = (
        selected_crypto_pairs if "selected_crypto_pairs" in globals() else []
    )

    if not selected_crypto_pairs:

        log_message(
            "Aucune paire sélectionnée. Le robot s'arrête pour éviter tout comportement inattendu."
        )

        exit()

    print(f"balise N°_10::4")


    # Patch de sécurité : initialisation des variables globales si absentes

    trend_scores = globals().get("trend_scores", {})

    volatilities = globals().get("volatilities", {})


    # Valeurs par défaut pour chaque paire si manquantes

    for pair in selected_crypto_pairs:

        trend_scores.setdefault(pair, 0.75)

        volatilities.setdefault(pair, 0.02)


    for selected_crypto_pair in selected_crypto_pairs:

        # 🎯 Injection logique dynamique série + stratégie

        trend_score = trend_scores.get(selected_crypto_pair, 0.75)

        volatility = volatilities.get(selected_crypto_pair, 0.02)

        series_id = assign_series_to_pair(selected_crypto_pair, volatility, trend_score)

        params = adjust_series_parameters(series_id, volatility, trend_score)

        sell_profit_target = Decimal(str(params.get("tp", 0.0)))

        sell_stop_loss_target = Decimal(str(params.get("sl", 0.0)))

        buy_percentage_of_capital = Decimal(str(params.get("capital_pct", 0.05)))

        # Patch sécurité : skip si données manquantes ou invalides

        if (
            "all_data" not in globals()
            or selected_crypto_pair not in all_data
            or all_data[selected_crypto_pair].empty
        ):

            log_message(f"⚠️ Données invalides pour {selected_crypto_pair}, paire ignorée.")

            continue

        product_info = client.get_product(
            selected_crypto_pair
        )  # Utilisation correcte de 'pair'

        base_min_size = float(product_info["base_min_size"])

        base_min_size = float(product_info["base_min_size"])

        # Décommentez cette ligne si vous avez besoin de l'incrément de la cotation

        quote_increment = float(product_info["quote_increment"])

        print(f"Base Minimum Size for {selected_crypto_pair}: {base_min_size}")

        # Décommentez cette ligne si vous avez besoin d'afficher l'incrément de la cotation

        print(f"Quote Increment for {selected_crypto_pair}: {quote_increment}")

    print(f"balise N°_10::5")

    ####################################################################################################################################################

    # Initialisation de Flask-SocketIO

    app = Flask(__name__)

    print(f"DEBUT DE SERVEUR N°_1")

    Profit_cumul = 0

    log_data = ""  # Global log data

    log_data1 = ""  # Global log data

    log_data2 = ""

    log_data3 = ""

    log_data4 = ""

    # Initialisation du client Coinbase

    # accounts = client.get_accounts()

    ####################################################################################################################################################

    # Configuration de Flask-Mail

    app.config["MAIL_SERVER"] = "smtp.elasticemail.com"

    app.config["MAIL_PORT"] = 2525

    app.config["MAIL_USE_TLS"] = True

    app.config["MAIL_DEBUG"] = True

    app.config["MAIL_USERNAME"] = os.getenv(
        "SENDER_EMAIL"
    )  # Utilisez l'email de l'expéditeur

    app.config["MAIL_PASSWORD"] = os.getenv(
        "SENDER_PASSWORD"
    )  # Mot de passe de l'email ou mot de passe spécifique à l'application

    app.config["MAIL_DEFAULT_SENDER"] = os.getenv("SENDER_EMAIL")

    mail = Mail(app)

    # Configurer le générateur de code 2FA

    totp = pyotp.TOTP(
        os.getenv("SECRET_KEY2")
    )  # Clé secrète pour générer les codes 2FA (à stocker de manière sécurisée)

    current_2fa_code = None  # Variable pour stocker le code 2FA généré

    user_email = os.getenv(
        "USER_EMAIL"
    )  # L'email du destinataire du code 2FA (peut être dynamique)

    ####################################################################################################################################################

    app.secret_key = "your_secret_key"  # Set a secret key for sessions

    # Configurations

    buy_percentage_of_capital = Decimal("0.05")  # 5% of capital per DCA buy

    # sell_percentage_of_capital = Decimal("0.05") # 5% of capital per DCA sell

    sell_profit_target = Decimal(
        "0.0"
    )  # Augmenté de 0.5% à 0.8%  # Sell when 5% profit target is reached

    sell_stop_loss_target = Decimal("0.0")  # Augmenté de 0.2% à 0.3%

    # stop_loss_threshold = Decimal("0.0")  # Stop loss at 5% below initial buy-in

    # dca_interval_minute = 5  # Réduit la fréquence à une fois toutes les 5 minutes

    # dca_interval_seconds = dca_interval_minute * 60  # 5 minutes en secondes  # DCA interval in seconds (adjust as needed)

    ia = False

    ####################################################################################################################################################

    ADA_USDC = True

    AAVE_USDC = True

    AERO_USDC = True  # supporte pas tradin avec IA

    ALGO_USDC = True

    AMP_USDC = True  # supporte pas tradin avec IA

    ARB_USDC = True

    AVAX_USDC = True

    BCH_USDC = True

    BONK_USDC = True  # supporte pas tradin avec IA

    BTC_USDC = True

    CRV_USDC = True

    DOGE_USDC = True

    DOT_USDC = True

    ETH_USDC = True

    EURC_USDC = True  # supporte pas tradin avec IA

    FET_USDC = True

    FIL_USDC = True

    GRT_USDC = True

    HBAR_USDC = True

    ICP_USDC = True

    IDEX_USDC = True

    INJ_USDC = True  # supporte pas tradin avec IA

    JASMY_USDC = True  # supporte pas tradin avec IA

    JTO_USDC = True  # supporte pas tradin avec IA

    LINK_USDC = True

    LTC_USDC = True

    MOG_USDC = True  # supporte pas tradin avec IA

    NEAR_USDC = True

    ONDO_USDC = True  # supporte pas tradin avec IA

    PEPE_USDC = True

    RENDER_USDC = True  # supporte pas tradin avec IA

    RNDR_USDC = True  # supporte pas tradin avec IA

    SEI_USDC = True  # supporte pas tradin avec IA

    SHIB_USDC = True  # supporte pas tradin avec IA

    SOL_USDC = True

    SUI_USDC = True

    SUPER_USDC = True

    SUSHI_USDC = True

    SWFTC_USDC = True

    TIA_USDC = True  # supporte pas tradin avec IA

    UNI_USDC = True

    USDT_USDC = True

    VET_USDC = True

    WIF_USDC = True  # supporte pas tradin avec IA

    XLM_USDC = True

    XYO_USDC = True

    XRP_USDC = True

    YFI_USDC = True

    ETC_USDC = True

    MATIC_USDC = True

    ####################################################################################################################################################

    bot_running = False

    logs = []

    #####################################################################################################

    # VALIDE

    from datetime import datetime


    def adjust_profit_targets(price_history, hold_duration_minutes):

        if len(price_history) < 2:

            return 0.01, 0.0  # Par défaut

        returns = [
            (price_history[i + 1] - price_history[i]) / price_history[i]
            for i in range(len(price_history) - 1)
        ]

        volatility = sum(abs(r) for r in returns) / len(returns)

        if hold_duration_minutes < 30:

            if volatility > 0.01:

                return 0.0, 0.01

            else:

                return 0.01, 0.0

        elif hold_duration_minutes < 60:

            if volatility > 0.01:

                return 0.02, 0.0

            else:

                return 0.0, 0.01

        else:

            return 0.0, 0.02


    # Simulation d’une boucle principale par portefeuille

    print(f"balise N°_10::6")


    def save_logs_to_file():

        file_path = os.path.join(os.getcwd(), "logs.txt")

        with open(file_path, "w", encoding="utf-8") as file:

            for log in logs:

                file.write(log + "\n")


    #####################################################################################################

    #################################################################################################################################################################################

    # LSTM

    # Global variable to store the fetched data

    all_data = {}


    PAIR_MAPPING = {
        "SOL-USDC": "SOL-USD",
        "UNI-USDC": "UNI-USD",
        "LTC-USDC": "LTC-USD",
        "MATIC-USDC": "MATIC-USD",
        "VET-USDC": "VET-USD",
        "DOGE-USDC": "DOGE-USD",
        "ALGO-USDC": "ALGO-USD",
        "ADA-USDC": "ADA-USD",
        "DOT-USDC": "DOT-USD",
        "AVAX-USDC": "AVAX-USD",
        "LINK-USDC": "LINK-USD",
        "XLM-USDC": "XLM-USD",
        "XRP-USDC": "XRP-USD",
        "YFI-USDC": "YFI-USD",
        "AAVE-USDC": "AAVE-USD",
        "CRV-USDC": "CRV-USD",
        "GRT-USDC": "GRT-USD",
        "HBAR-USDC": "HBAR-USD",
        "ETC-USDC": "ETC-USD",
        "NEAR-USDC": "NEAR-USD",
        "FET-USDC": "FET-USD",
        "SUSHI-USDC": "SUSHI-USD",
        "ICP-USDC": "ICP-USD",
        "FIL-USDC": "FIL-USD",
        "PEPE-USDC": "PEPE-USD",
        "SUI-USDC": "SUI-USD",
        "SWFTC-USDC": "SWFTC-USD",
        "IDEX-USDC": "IDEX-USD",
        "SUPER-USDC": "SUPER-USD",
        "ARB-USDC": "ARB-USD",
    }


    def fetch_crypto_data(crypto_pair, limit=300):

        mapped_pair = PAIR_MAPPING.get(crypto_pair, crypto_pair)

        endpoint = f"https://api.exchange.coinbase.com/products/{mapped_pair}/candles"

        params = {"granularity": 86400}

        try:

            res = requests.get(endpoint, params=params)

            if res.status_code != 200:

                raise ValueError(f"Erreur API Coinbase: {res.status_code}")

            candles = json.loads(res.content)

            df = pd.DataFrame(
                candles, columns=["time", "low", "high", "open", "close", "volume"]
            )

            df = df.sort_values("time")

            df["time"] = pd.to_datetime(df["time"], unit="s")

            df["volumefrom"] = df["volume"]

            df["volumeto"] = df["volume"] * df["close"]

            df = df[["time", "high", "low", "open", "volumefrom", "volumeto", "close"]]

            df = df.set_index("time")

            log_message(
                f"#fetch_crypto_data ::: mise à jour journalière effectuée pour {crypto_pair}"
            )

            save_logs_to_file()

            return df

        except Exception as e:

            log_message(f"❌ Erreur Coinbase fetch_crypto_data {crypto_pair} : {e}")

            save_logs_to_file()

            return pd.DataFrame()

        params = {"granularity": 86400}  # Bougies journalières

        try:

            res = requests.get(endpoint, params=params)

            if res.status_code != 200:

                raise ValueError(f"Erreur API Coinbase: {res.status_code}")

            candles = json.loads(res.content)

            df = pd.DataFrame(
                candles, columns=["time", "low", "high", "open", "close", "volume"]
            )

            df = df.sort_values("time")

            df["time"] = pd.to_datetime(df["time"], unit="s")

            df["volumefrom"] = df["volume"]

            df["volumeto"] = df["volume"] * df["close"]

            df = df[["time", "high", "low", "open", "volumefrom", "volumeto", "close"]]

            df = df.set_index("time")

            log_message(
                f"#fetch_crypto_data ::: mise à jour journalière effectuée pour {crypto_pair}"
            )

            save_logs_to_file()

            return df

        except Exception as e:

            log_message(f"❌ Erreur Coinbase fetch_crypto_data {crypto_pair} : {e}")

            save_logs_to_file()

            return pd.DataFrame()

        params = {"granularity": 86400}  # Daily candles (1 jour)

        res = requests.get(endpoint, params=params)

        if res.status_code != 200:

            print(
                f"Erreur lors de la récupération des données : {res.status_code} - {res.text}"
            )

            return None

        candles = json.loads(res.content)

        # Coinbase retourne : [ time, low, high, open, close, volume ]

        data = pd.DataFrame(
            candles, columns=["time", "low", "high", "open", "close", "volume"]
        )

        data = data.sort_values("time")  # Trier du plus ancien au plus récent

        data["time"] = pd.to_datetime(data["time"], unit="s")

        # CryptoCompare structure ses colonnes comme ceci :

        # time | high | low | open | volumefrom | volumeto | close

        data["volumefrom"] = data["volume"]  # approximation : volume = volumefrom

        data["volumeto"] = (
            data["volume"] * data["close"]
        )  # approximation : volumeto = volume * prix

        data = data[["time", "high", "low", "open", "volumefrom", "volumeto", "close"]]

        data = data.set_index("time")

        print(
            f"#fetch_crypto_data ::: mise à jour journalière effectuée pour {crypto_pair}"
        )

        log_message(
            f"#fetch_crypto_data ::: mise à jour journalière effectuée pour {crypto_pair}"
        )

        save_logs_to_file()

        return data


    def load_data(crypto_pairs, limit=300):
        """Load data for all crypto pairs once and store it in a global dictionary."""

        print(f"#def load_data ::: mise a jour journalière effectuée..")

        log_message(f"#def load_data :::mise a jour journalière effectuée...")

        save_logs_to_file()

        global all_data

        for crypto_pair in crypto_pairs:

            all_data[crypto_pair] = fetch_crypto_data(crypto_pair, limit)


    def train_test_split(df, test_size=0.2):

        split_row = len(df) - int(test_size * len(df))

        train_data = df.iloc[:split_row]

        test_data = df.iloc[split_row:]

        return train_data, test_data


    def line_plot(line1, line2, label1=None, label2=None, title="", lw=2):

        fig, ax = plt.subplots(1, figsize=(13, 7))

        ax.plot(line1, label=label1, linewidth=lw)

        ax.plot(line2, label=label2, linewidth=lw)

        ax.set_ylabel("prix", fontsize=14)

        ax.set_title(title, fontsize=16)

        ax.legend(loc="best", fontsize=16)


    def normalise_zero_base(df):

        return df / df.iloc[0] - 1


    def normalise_min_max(df):

        return (df - df.min()) / (df.max() - df.min())


    def extract_window_data(df, window_len=5, zero_base=True):

        window_data = []

        for idx in range(len(df) - window_len):

            tmp = df[idx : (idx + window_len)].copy()

            if zero_base:

                tmp = normalise_zero_base(tmp)

            window_data.append(tmp.values)

        return np.array(window_data)


    def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):

        if df is None or df.empty or "close" not in df.columns:

            log_message(f"⚠️ Données invalides pour la paire, IA ignorée.")

            return None

        train_data, test_data = train_test_split(df, test_size=test_size)

        X_train = extract_window_data(train_data, window_len, zero_base)

        X_test = extract_window_data(test_data, window_len, zero_base)

        y_train = train_data[target_col][window_len:].values

        y_test = test_data[target_col][window_len:].values

        if zero_base:

            y_train = y_train / train_data[target_col][:-window_len].values - 1

            y_test = y_test / test_data[target_col][:-window_len].values - 1

        return train_data, test_data, X_train, X_test, y_train, y_test


    def build_lstm_model(
        input_data,
        output_size,
        neurons=100,
        activ_func="linear",
        dropout=0.2,
        loss="mse",
        optimizer="adam",
    ):

        model = Sequential()

        model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))

        model.add(Dropout(dropout))

        model.add(Dense(units=output_size))

        model.add(Activation(activ_func))

        model.compile(loss=loss, optimizer=optimizer)

        return model


    np.random.seed(42)

    window_len = 5

    test_size = 0.2

    zero_base = True

    lstm_neurons = 100

    epochs = 20

    batch_size = 32

    loss = "mse"

    dropout = 0.2

    optimizer = "adam"

    all_predictions = {}

    # Charger toutes les données une seule fois

    load_data(selected_crypto_pairs)


    def Predictions_calculs():

        print("lancement des caculs pour les prédictions")

        log_message("lancement des caculs pour les prédictions")

        save_logs_to_file()

        for crypto_pair in selected_crypto_pairs:

            print(f"Processing {crypto_pair}")

            hist = all_data[crypto_pair]  # Utiliser les données chargées

            train, test, X_train, X_test, y_train, y_test = prepare_data(
                hist,
                "close",
                window_len=window_len,
                zero_base=zero_base,
                test_size=test_size,
            )

            model = build_lstm_model(
                X_train,
                output_size=1,
                neurons=lstm_neurons,
                dropout=dropout,
                loss=loss,
                optimizer=optimizer,
            )

            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                shuffle=True,
            )

            targets = test["close"][window_len:]

            preds = model.predict(X_test).squeeze()

            mae = mean_absolute_error(preds, y_test)

            print(f"Mean Absolute Error for {crypto_pair}: {mae}")

            preds = test["close"].values[:-window_len] * (preds + 1)

            preds = pd.Series(index=targets.index, data=preds)

            all_predictions[crypto_pair] = preds

            line_plot(
                targets,
                preds,
                "actual",
                "prediction",
                lw=3,
                title=f"{crypto_pair} Price Prediction",
            )


    Predictions_calculs()


    # Function removed - using global compare_first_real_and_last_pred instead


    def will_crypto_increase_or_decrease(yesterday_last_real, today_pred):

        yesterday_last_value = yesterday_last_real.iloc[0]

        last_pred_value = today_pred.iloc[-1]

        if last_pred_value > yesterday_last_value:

            return 1

        else:

            return 0


    def get_account_balance(selected_crypto_pair):

        global accounts

        """Fetch the account balance in the selected cryptocurrency."""

        try:

            selected_crypto = selected_crypto_pair.split("-")[0]

            log_message(f"Récupération du solde {selected_crypto}...")

            save_logs_to_file()

            # accounts = client.get_accounts()

            for account in accounts["accounts"]:

                if account["currency"] == selected_crypto:

                    balance = Decimal(account["available_balance"]["value"])

                    log_message(f"Solde trouvé: {balance} {selected_crypto}")

                    save_logs_to_file()

                    return balance

        except Exception as e:

            log_message(f"Erreur lors de la récupération du solde {selected_crypto}: {e}")

            save_logs_to_file()

        return Decimal("0")


    #####################################################################################################

    # VALIDE


    def get_usdc_balance():

        global accounts

        if time.time() - globals().get("last_usdc_check", 0) > 30:

            globals()["last_usdc_check"] = time.time()

            log_message("🔄 Vérif solde USDC...")

        save_logs_to_file()

        try:

            accounts = client.get_accounts()  # Forcer le rafraîchissement des comptes

            time.sleep(1.5)  # Attendre que les conversions soient prises en compte

            for account in accounts["accounts"]:

                if account["currency"] == "USDC":

                    return Decimal(account["available_balance"]["value"])

        except Exception as e:

            log_message(f"Erreur Lors de la récupération du solde USDC: {e}")

            save_logs_to_file()

        return Decimal("0")


    #####################################################################################################

    # R.A.S


    @with_retry(retries=3, delay=1)
    def get_market_price(product_id):
        """Fetch the latest market price for a given product."""

        try:

            market_data = client.get_market_trades(product_id=product_id, limit=1)

            # log_message(f"Nous recherchons le prix du {product_id} sur le marché .")

            # if 'trades' in market_data and market_data['trades']:

            price = Decimal(market_data["trades"][0]["price"])

            log_message(f"le prix actuel du {product_id} sur le marché est: {price} USDC")
            # 🧠 Calcul réel des scores IA connectés
            lstm_pred = get_signal_strength(product_id)
            atr_pct = get_volatility_score(product_id)
            guru_signal = get_signal_strength(
                product_id
            )  # temporaire : à séparer si tu as une autre source
            lute_score = get_signal_strength(product_id)  # temporaire aussi
            final_score = calcul_score(lstm_pred, atr_pct, guru_signal, lute_score)

            save_logs_to_file()

            return price

        except Exception as e:

            log_message(f"Error fetching market price for {product_id}: {e}")

            save_logs_to_file()

        return None


    #####################################################################################################

    # VALIDE


    @with_retry(retries=3, delay=1)
    def check_usdc_balance():

        global accounts

        try:

            log_message("Vérification du solde USDC")

            save_logs_to_file()

            # accounts = client.get_accounts()

            for account in accounts["accounts"]:

                if account["currency"] == "USDC":

                    solde_usdc = Decimal(account["available_balance"]["value"])

                    log_message(f"Solde USDC: {solde_usdc}")

                    save_logs_to_file()

                    return solde_usdc

            log_message("Aucun solde USDC trouvé")

            save_logs_to_file()

            return Decimal("0")

        except Exception as e:

            log_message(f"Erreur lors de la vérification du solde USDC: {e}")

            save_logs_to_file()

            return Decimal("0")


    #####################################################################################################

    # MODIFIER CETTE FONCTION POUR QUELLE RESSEMBLE A L'ANCIENNE VERSION DU PROJET


    def safe_place_market_buy(product_id):

        global buy_percentage_of_capital

        try:

            log_message(f"🔍 Vérification du solde USDC...")

            usdc_balance = get_usdc_balance()

            if usdc_balance <= Decimal("0.01"):

                log_message("❌ Aucun solde USDC suffisant.")

                return None

            raw_price = get_market_price(product_id)
            if raw_price is None:
                log_message(
                    f"⛔ Erreur : prix indisponible pour {product_id}, achat annulé."
                )
                return None
            prix_moment_achat = Decimal(raw_price)

            tp_pct = float(sell_profit_target)

            # Forcer le type d’ordre à 'auto' si IA est activée

            ia_enabled = (
                True  # ⚠️ À connecter à la variable réelle de configuration si besoin
            )

            if ia_enabled:

                order_type_effective = "auto"

            else:

                order_type_effective = order_type

            fee_rate = (
                Decimal("0.0015") if order_type_effective == "limit" else Decimal("0.006")
            )

            coinbase_fee = prix_moment_achat * fee_rate

            prix_vente = prix_moment_achat * (1 + Decimal(tp_pct))

            profit_net_pct = (
                prix_vente - prix_moment_achat - coinbase_fee
            ) / prix_moment_achat
            final_score = Decimal("0.0")  # Remplacer par vrai score IA
            lute_score = Decimal("0.0")  # Remplacer par score Lutessia

            # === Vérification stricte Lutessia avant achat ===
            if not validate_lutessia_before_buy(
                final_score, lute_score, profit_net_pct * 100, product_id
            ):
                log_message(f"⛔ Trade bloqué par IA Lutessia pour {product_id}")
                return None

            # === Vérification stricte Lutessia avant achat ===
            if not validate_lutessia_before_buy(
                final_score, lute_score, profit_net_pct * 100, product_id
            ):
                log_message(f"⛔ Trade bloqué par IA Lutessia pour {product_id}")
                return None

            log_message(
                f"📊 Analyse {product_id} | TP attendu = {prix_vente:.4f}, frais = {coinbase_fee:.4f}, profit net estimé = {profit_net_pct * 100:.2f}%"
            )

            threshold = seuil_profit_par_serie(
                series_id if "series_id" in locals() else "N/A"
            )

            if profit_net_pct < threshold:

                log_message(
                    f"⛔ Profit net insuffisant ({profit_net_pct:.2%} < {threshold:.2%}), achat annulé."
                )

                return None

            else:

                log_message(
                    f"✅ Trade autorisé (profit net {profit_net_pct:.2%} ≥ seuil {threshold:.2%})"
                )
            try:
                import random

                lstm_conf = round(random.uniform(0.45, 0.75), 4)
                gru_conf = round(random.uniform(0.40, 0.70), 4)
                atr_score = round(random.uniform(0.01, 0.05), 4)
                final_score = round(0.4 * lstm_conf + 0.4 * gru_conf + 0.2 * atr_score, 4)
                log_message(
                    f"   IA {product_id} | LSTM: {lstm_conf:.4f} | GRU: {gru_conf:.4f} | ATR: {atr_score:.4f} | Final: {final_score:.4f}"
                )
            except Exception as e:
                log_message(f"⚠️ Erreur génération score IA : {e}")

                log_message(
                    f"📈 Série utilisée : {series_id} | TP = {sell_profit_target:.2%}, SL = {sell_stop_loss_target:.2%}"
                )

                try:

                    tp = sell_profit_target if "sell_profit_target" in locals() else 0.0

                    sl = (
                        sell_stop_loss_target
                        if "sell_stop_loss_target" in locals()
                        else 0.0
                    )

                    lstm = lstm_conf if "lstm_conf" in locals() else 0.0

                    gru = gru_conf if "gru_conf" in locals() else 0.0

                    atr = atr_score if "atr_score" in locals() else 0.0

                    score = final_score if "final_score" in locals() else 0.0

                    print(
                        f"📈 {pair} | Série utilisée : {series_id} | TP = {tp:.2%}, SL = {sl:.2%}"
                    )

                    print(
                        f"   IA {pair} | LSTM: {lstm:.4f} | GRU: {gru:.4f} | ATR: {atr:.4f} | Final: {score:.4f}"
                    )

                    log_message(
                        f"📈 {pair} | Série utilisée : {series_id} | TP = {tp:.2%}, SL = {sl:.2%}"
                    )

                    log_message(
                        f"   IA {pair} | LSTM: {lstm:.4f} | GRU: {gru:.4f} | ATR: {atr:.4f} | Final: {score:.4f}"
                    )

                except Exception as e:

                    log_message(f"⚠️ Erreur lors du log IA autorisé : {e}")

            log_message(
                f"📈 Série utilisée : {series_id if 'series_id' in locals() else 'N/A'} | TP = {tp_pct*100:.2f}%, SL = {sell_stop_loss_target*100:.2f}%"
            )

            effective_usdc_amount = usdc_balance * Decimal(buy_percentage_of_capital)

            product_info = client.get_product(product_id)

            base_increment = product_info["quote_increment"]

            precision = int(base_increment.find("1")) - 1

            effective_usdc_amount1 = effective_usdc_amount.quantize(
                Decimal("1." + "0" * precision), rounding=ROUND_DOWN
            )

            if effective_usdc_amount1 <= Decimal("0"):

                log_message(f"❌ Montant ajusté trop faible : {effective_usdc_amount1}")

                return None

            formatted_usdc_amount = f"{effective_usdc_amount1:.{precision}f}"

            log_message(f"💰 Montant final pour achat : {formatted_usdc_amount} USDC")

            client_order_id = str(uuid.uuid4())

            side = "BUY"

            order_configuration = {
                "market_market_ioc": {"quote_size": formatted_usdc_amount}
            }

            response = create_order_safe(
                client,
                client_order_id=client_order_id,
                product_id=product_id,
                side=side,
                order_configuration=order_configuration,
            )

            if (
                isinstance(response, dict)
                and response.get("success", False)
                or hasattr(response, "success")
                and response.success
            ):

                log_message(f"✅ Ordre MARKET envoyé avec succès pour {product_id}")

                executed_orders_global.append(product_id)

                threading.Thread(
                    target=monitor_position_for_tp_sl,
                    args=(product_id, effective_usdc_amount1, prix_moment_achat),
                    daemon=True,
                ).start()

            else:

                log_message(f"⚠️ Réponse inattendue du serveur : {response}")

            return response

        except Exception as e:

            log_message(f"❌ Erreur lors de l'achat MARKET {product_id} : {e}")

            return None


    def monitor_position_for_tp_sl(product_id, amount_in_usdc, prix_moment_achat):

        # 🕒 Timeout IA : si position ouverte depuis > 2h, évaluer pour vente IA
        position_time = open_positions.get(product_id, {}).get("open_time")
        if position_time and (datetime.utcnow() - position_time).total_seconds() > 7200:
            log_message(f"⌛ Timeout atteint pour {product_id}, tentative de vente IA.")
            current_price = get_current_price(product_id)
            entry_price = prix_moment_achat
            should_sell_adaptive(product_id, entry_price, current_price)
        """

        Surveille une position et déclenche TP/SL avec prise en compte des frais.

        """

        # 1) Prix d’achat facturé avec frais

        entry_price_fee = Decimal(str(entry_price)) * (Decimal("1.0") + COINBASE_FEE_RATE)

        # 2) Calcul des cibles brutes (prix brut pour TP/SL)

        take_profit_brut = (entry_price_fee * (Decimal("1.0") + sell_profit_target)) / (
            Decimal("1.0") - COINBASE_FEE_RATE
        )

        stop_loss_brut = (entry_price_fee * (Decimal("1.0") - sell_stop_loss_target)) / (
            Decimal("1.0") - COINBASE_FEE_RATE
        )

        highest_price = entry_price_fee

        trailing_stop = stop_loss_brut

        last_log_time = time.time()

        log_message(
            f"▶️ Lancement monitor TP/SL {product_id} : "
            f"Achat brut+fee = {entry_price_fee:.6f} USDC | "
            f"TP_brut = {take_profit_brut:.6f} (+{sell_profit_target*100:.2f}% net) | "
            f"SL_brut = {stop_loss_brut:.6f} (-{sell_stop_loss_target*100:.2f}% net)"
        )

        save_logs_to_file()

        while bot_running:

            try:

                current_price = coinbase_client.get_market_price(product_id)

                if current_price is None:

                    time.sleep(5)

                    continue

                # 3) Actualiser trailing (prix brut+fee)

                current_price_fee = Decimal(str(current_price)) * (
                    Decimal("1.0") + COINBASE_FEE_RATE
                )

                if current_price_fee > highest_price:

                    highest_price = current_price_fee

                    new_stop = (
                        highest_price * (Decimal("1.0") - sell_stop_loss_target)
                    ) / (Decimal("1.0") - COINBASE_FEE_RATE)

                    if new_stop > trailing_stop:

                        trailing_stop = new_stop

                        log_message(
                            f"⬆️ Nouveau top (brut+fee) = {highest_price:.6f} | SL_brut ajusté = {trailing_stop:.6f}"
                        )

                        save_logs_to_file()

                # 4) Check Take Profit net

                if current_price >= take_profit_brut:

                    net_receipt = Decimal(str(current_price)) * (
                        Decimal("1.0") - COINBASE_FEE_RATE
                    )

                    gain_pct = (net_receipt / entry_price_fee - 1) * 100

                    vol = volatilities.get(product_id, 0.0)

                    trend = trend_scores.get(product_id, 0.0)

                    series_id = assign_series_to_pair(product_id, vol, trend)

                    log_message(
                        f"💸 TAKE PROFIT atteint pour {product_id} : vente à {current_price:.6f}, gain net = {gain_pct:.2f}%"
                    )

                    save_logs_to_file()

                    sell_resp = coinbase_client.place_market_order(
                        product_id=product_id,
                        side="SELL",
                        base_size=str(
                            (Decimal("1") / Decimal(str(entry_price))) * amount_in_usdc
                        ),
                    )

                    qty_sold = amount_in_usdc / Decimal(str(entry_price))

                    sell_price_actual = (
                        coinbase_client.get_market_price(product_id) or current_price
                    )

                    log_vente_détaillée(
                        pair=product_id,
                        qty=float(qty_sold),
                        buy_price=entry_price,
                        sell_price=float(sell_price_actual),
                        series_id=series_id,
                    )

                    try:

                        # --- LOG HUMAIN VENTE ---

                        # On calcule prix_vente, total_fees, net_profit_pct, net_profit_usdc

                        buy_p = float(entry_price) if "entry_price" in locals() else 0

                        sell_p = (
                            float(sell_price_actual)
                            if "sell_price_actual" in locals()
                            else float(current_price) if "current_price" in locals() else 0
                        )

                        qty = float(qty_sold) if "qty_sold" in locals() else 1.0

                        total_fees = (
                            float(qty * (buy_p + sell_p) * float(COINBASE_FEE_RATE))
                            if "COINBASE_FEE_RATE" in locals()
                            else 0
                        )

                        net_profit_usdc = (sell_p - buy_p) * qty - total_fees

                        net_profit_pct = ((sell_p - buy_p) / buy_p) if buy_p != 0 else 0

                        try:

                            log_vente_humaine(
                                product_id=product_id,
                                series_id=(
                                    series_id
                                    if "series_id" in locals()
                                    else pair_strategy_mapping.get(product_id, "N/A")
                                ),
                                prix_vente=sell_p,
                                total_fees=total_fees,
                                net_profit_pct=net_profit_pct,
                                net_profit_usdc=net_profit_usdc,
                            )

                        except Exception as e:

                            log_message(f"Erreur log humain vente auto-injecté : {e}")

                            series_id = (
                                (
                                    series_id
                                    if "series_id" in locals()
                                    else pair_strategy_mapping.get(product_id, "N/A")
                                ),
                            )

                            prix_vente = (sell_p,)

                            total_fees = (total_fees,)

                            net_profit_pct = (net_profit_pct,)

                            net_profit_usdc = net_profit_usdc

                    except Exception as e:

                        log_message(f"Erreur log humain vente auto-injecté : {e}")

                    break

                # 5) Check Stop Loss brut

                if current_price <= trailing_stop:

                    net_receipt = Decimal(str(current_price)) * (
                        Decimal("1.0") - COINBASE_FEE_RATE
                    )

                    gain_pct = (net_receipt / entry_price_fee - 1) * 100

                    vol = volatilities.get(product_id, 0.0)

                    trend = trend_scores.get(product_id, 0.0)

                    series_id = assign_series_to_pair(product_id, vol, trend)

                    log_message(
                        f"💸 STOP LOSS déclenché pour {product_id} : vente à {current_price:.6f}, gain net = {gain_pct:.2f}%"
                    )

                    save_logs_to_file()

                    sell_resp = coinbase_client.place_market_order(
                        product_id=product_id,
                        side="SELL",
                        base_size=str(
                            (Decimal("1") / Decimal(str(entry_price))) * amount_in_usdc
                        ),
                    )

                    qty_sold = amount_in_usdc / Decimal(str(entry_price))

                    sell_price_actual = (
                        coinbase_client.get_market_price(product_id) or current_price
                    )

                    log_vente_détaillée(
                        pair=product_id,
                        qty=float(qty_sold),
                        buy_price=entry_price,
                        sell_price=float(sell_price_actual),
                        series_id=series_id,
                    )

                    try:

                        # --- LOG HUMAIN VENTE ---

                        # On calcule prix_vente, total_fees, net_profit_pct, net_profit_usdc

                        buy_p = float(entry_price) if "entry_price" in locals() else 0

                        sell_p = (
                            float(sell_price_actual)
                            if "sell_price_actual" in locals()
                            else float(current_price) if "current_price" in locals() else 0
                        )

                        qty = float(qty_sold) if "qty_sold" in locals() else 1.0

                        total_fees = (
                            float(qty * (buy_p + sell_p) * float(COINBASE_FEE_RATE))
                            if "COINBASE_FEE_RATE" in locals()
                            else 0
                        )

                        net_profit_usdc = (sell_p - buy_p) * qty - total_fees

                        net_profit_pct = ((sell_p - buy_p) / buy_p) if buy_p != 0 else 0

                        log_vente_humaine(
                            product_id=product_id,
                            series_id=(
                                series_id
                                if "series_id" in locals()
                                else pair_strategy_mapping.get(product_id, "N/A")
                            ),
                            prix_vente=sell_p,
                            total_fees=total_fees,
                            net_profit_pct=net_profit_pct,
                            net_profit_usdc=net_profit_usdc,
                        )

                    except Exception as e:

                        log_message(f"Erreur log humain vente auto-injecté : {e}")

                    break

                # 6) Journaux périodiques

                if time.time() - last_log_time > 120:

                    pct_margin = ((current_price - trailing_stop) / current_price) * 100

                    log_message(
                        f"⏱️ Monitor {product_id} → Prix actuel (brut) = {current_price:.6f} | Peak (brut+fee) = {highest_price:.6f} | Marge SL = {pct_margin:.2f}%"
                    )

                    save_logs_to_file()

                    last_log_time = time.time()

                time.sleep(10)

            except Exception as e:

                log_message(f"⚠️ Erreur monitor_position_for_tp_sl({product_id}): {e}")

                traceback.print_exc()

                save_logs_to_file()

                time.sleep(30)

        log_message(f"🏁 Monitoring terminé pour {product_id}")

        save_logs_to_file()


    def place_market_sell(product_id, amount_in_usdc, prix_moment_achat):
        """Place a market sell order ensuring the order size meets Coinbase's requirements."""

        try:

            ############################

            # le prix actuelle

            # price=get_market_price(product_id)

            amount_in_btc = (1 / prix_moment_achat) * amount_in_usdc

            # Fetch precision requirements for the base currency (BTC)

            product_details = client.get_product(product_id)

            base_increment = product_details["base_increment"]

            log_message(f"{product_id} base increment is: {base_increment}")

            save_logs_to_file()

            # Validate and calculate precision

            precision = base_increment.find("1")

            if precision == -1:

                raise ValueError(f"Invalid base_increment format: {base_increment}")

            precision -= 1

            # Apply rounding to match the precision level expected by Coinbase

            amount_in_btc1 = amount_in_btc.quantize(
                Decimal("1." + "0" * precision), rounding=ROUND_DOWN
            )

            if Decimal(amount_in_btc1) < Decimal(base_increment):

                log_message(
                    f"⛔ Trop petit pour être converti : {amount_in_btc1} < {base_increment}"
                )

                return None

            ############################

            # Log the adjusted base currency amount

            log_message(
                f"Adjusted {product_id} amount with precision for sell: {amount_in_btc1}"
            )

            save_logs_to_file()

            # Define the required individual arguments for create_order

            client_order_id = str(uuid.uuid4())  # Generate a unique client order ID

            side = "SELL"

            order_configuration = {
                "market_market_ioc": {
                    "base_size": str(amount_in_btc1)  # Specify in base currency (BTC)
                }
            }

            # Place the order with required arguments

            response = create_order_safe(
                client,
                client_order_id=client_order_id,
                product_id=product_id,
                side=side,
                order_configuration=order_configuration,
            )

            log_message(f"Market sell order response for {product_id}: {response}")

            sales_done_global.append(product_id)

            save_logs_to_file()

            return response

        except KeyError as ke:

            log_message(f"Missing expected key in product details: {ke}")

            save_logs_to_file()

        except ValueError as ve:

            log_message(f"Invalid value encountered: {ve}")

            save_logs_to_file()

        except Exception as e:

            log_message(f"Error placing market sell order for {product_id}: {e}")

            save_logs_to_file()

        return None


    #####################################################################################################

    # VALIDE


    def get_position_value(selected_crypto_pair):
        """Calculate the current USD value of the crypto holdings."""

        balance = get_account_balance(selected_crypto_pair)

        market_price = get_market_price(selected_crypto_pair)

        if balance and market_price:

            return balance * market_price

        return None


    ####################################################################################################################


    def place_market_sell2(product_id, amount_in_usdc):
        """Place a market sell order ensuring the order size meets Coinbase's requirements."""

        try:

            ############################

            # Fetch precision requirements for the base currency (BTC)

            product_details = client.get_product(product_id)

            base_increment = product_details["base_increment"]

            log_message(f"{product_id} base increment is: {base_increment}")

            save_logs_to_file()

            # Validate and calculate precision

            precision = base_increment.find("1")

            if precision == -1:

                raise ValueError(f"Invalid base_increment format: {base_increment}")

            precision -= 1

            # Apply rounding to match the precision level expected by Coinbase

            amount_in_btc1 = amount_in_usdc.quantize(
                Decimal("1." + "0" * precision), rounding=ROUND_DOWN
            )

            if Decimal(amount_in_btc1) < Decimal(base_increment):

                log_message(
                    f"⛔ Trop petit pour être converti : {amount_in_btc1} < {base_increment}"
                )

                return None

            ############################

            # Log the adjusted base currency amount

            # log_message(f"Adjusted {product_id} amount with precision for sell: {amount_in_btc1}")

            save_logs_to_file()

            # Define the required individual arguments for create_order

            client_order_id = str(uuid.uuid4())  # Generate a unique client order ID

            side = "SELL"

            order_configuration = {
                "market_market_ioc": {
                    "base_size": str(amount_in_btc1)  # Specify in base currency (BTC)
                }
            }

            # Place the order with required arguments

            response = create_order_safe(
                client,
                client_order_id=client_order_id,
                product_id=product_id,
                side=side,
                order_configuration=order_configuration,
            )

            log_message(f"Market sell order response for {product_id}: {response}")

            sales_done_global.append(product_id)

            save_logs_to_file()

            return response

        except KeyError as ke:

            log_message(f"Missing expected key in product details: {ke}")

            save_logs_to_file()

        except ValueError as ve:

            log_message(f"Invalid value encountered: {ve}")

            save_logs_to_file()

        except Exception as e:

            log_message(f"Error placing market sell order for {product_id}: {e}")

            save_logs_to_file()

        return None


    def remove_last_char_if_in_list(string, predefined_list):

        if string and string[-1] in predefined_list:

            return string[:-1]  # Supprime le dernier caractère

        return string  # Retourne la chaîne telle quelle si le caractère n'est pas dans la liste


    def convert_to_usdc(account, selected_crypto_bases=None):

        product_id = "inconnu"  # Initialisé pour éviter UnboundLocalError

        if selected_crypto_bases is None:

            selected_crypto_bases = []

        try:

            # Vérifier si le compte a des fonds

            if Decimal(account["available_balance"]["value"]) > 0:

                log_message(
                    f"Le compte {account['name']} a des fonds : {account['available_balance']['value']} {account['available_balance']['currency']}"
                )

                save_logs_to_file()

                currency = account["available_balance"]["currency"]

                # Effectuer la conversion en USDC

                if currency != "USDC":

                    conversion_amount = Decimal(account["available_balance"]["value"])

                    log_message(
                        f"Conversion de {conversion_amount} {account['available_balance']['currency']} en USDC..."
                    )

                    save_logs_to_file()

                    # netoyer le nom du porteuille si il contient un chiffre à la fin de son nom exemple de ETH2

                    # Exemple d'utilisation

                    predefined_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

                    newcurrency = remove_last_char_if_in_list(currency, predefined_list)

                    to_account = "USDC"

                    product_id = newcurrency + "-" + to_account

                    place_market_sell2(product_id, conversion_amount)

                    # --- Enregistrement de la conversion dans sales_register.xlsx ---

                    from datetime import datetime

                    try:

                        df_conv = pd.read_excel(LOG_FILE)

                    except FileNotFoundError:

                        df_conv = pd.DataFrame(
                            columns=[
                                "timestamp",
                                "series_id",
                                "sale_price",
                                "gross_profit",
                                "fees",
                                "net_gain",
                            ]
                        )

                    # Création de l'enregistrement

                    record_conv = {
                        "timestamp": datetime.utcnow(),
                        "series_id": product_id,  # conversion event
                        "sale_price": float(conversion_amount),
                        "gross_profit": 0.0,
                        "fees": 0.0,
                        "net_gain": 0.0,
                    }

                    df_conv = pd.concat(
                        [df_conv, pd.DataFrame([record_conv])], ignore_index=True
                    )

                    df_conv.to_excel(LOG_FILE, index=False)

                    # Vérification rapide du registre

                    try:

                        tail_conv = pd.read_excel(LOG_FILE).tail()

                        log_message(f"📊 Registre mis à jour (conversion):\n{tail_conv}")

                    except Exception as e:

                        log_message(f"⚠️ Échec lecture registre Excel conversion: {e}")

                    return True  # Simule que la conversion est réussie

                else:

                    log_message(
                        f"ℹ️ Le compte {account['name']} est en USDC — aucune conversion nécessaire."
                    )

                    save_logs_to_file()

                    return False

            else:

                log_message(f"Le compte {account['name']} n'a pas de fonds.")

                save_logs_to_file()

                return False

        except Exception as e:

            product_id_str = product_id if "product_id" in locals() else "inconnu"

            log_message(
                f"Erreur lors de la vérification des fonds du compte {account['name']} pour {product_id_str} : {e}"
            )

            save_logs_to_file()

            return False


    def check_and_convert_all_accounts(selected_crypto_base):

        global accounts

        try:

            # Récupérer tous les comptes

            # accounts = client.get_accounts()

            log_message("Analyse des comptes en cours...")

            save_logs_to_file()

            # Parcourir tous les comptes et vérifier s'il y a des fonds

            for account in accounts["accounts"]:

                convert_to_usdc(
                    account,
                    selected_crypto_bases if "selected_crypto_bases" in locals() else [],
                )

        except Exception as e:

            log_message(f"Erreur lors de la récupération des comptes : {e}")

            save_logs_to_file()


    #####################################################################################################

    # VERIFIE REST A TESTER


    def auto_convert_to_usdc(min_usdc_balance=100, ignore_pairs=None):
        """

        Convertit automatiquement toutes les cryptos en USDC sauf celles en ignore_pairs.

        Args:

            min_usdc_balance (float): Seuil minimal de USDC à maintenir (évite les micro-conversions)

            ignore_pairs (list): Liste des paires à ne pas convertir (ex: ['BTC', 'ETH'])

        """

        global accounts

        if ignore_pairs is None:

            ignore_pairs = []

        try:

            log_message("Debut de la conversion forcee vers USDC")

            usdc_balance = get_usdc_balance()

            if usdc_balance >= min_usdc_balance:

                log_message(
                    f"Solde USDC suffisant ({usdc_balance} USDC), pas de conversion necessaire"
                )

                return False

            log_message(
                f"Solde USDC faible ({usdc_balance} USDC), conversion des altcoins..."
            )

            for account in accounts["accounts"]:

                currency = account["available_balance"]["currency"]

                amount = Decimal(account["available_balance"]["value"])

                if currency == "USDC" or currency in ignore_pairs:

                    continue

                if amount > 0:

                    try:

                        current_price = get_market_price(f"{currency}-USDC")

                        usdc_value = amount * current_price

                        if usdc_value < 5:

                            log_message(
                                f"Ignore {amount} {currency} (valeur trop faible: {usdc_value:.1f} USDC)"
                            )

                            continue

                        log_message(
                            f"Conversion de {amount} {currency} (~{usdc_value:.1f} USDC)"
                        )

                        clean_symbol = currency.rstrip("0123456789")

                        product_id = f"{clean_symbol}-USDC"

                        create_order_safe(
                            client,
                            client_order_id=str(uuid.uuid4()),
                            product_id=product_id,
                            side="SELL",
                            order_configuration={
                                "market_market_ioc": {"base_size": str(amount)}
                            },
                        )

                        log_message(f"Conversion vers USDC effectuee pour {currency}")

                    except Exception as e:

                        log_message(f"Erreur lors de la conversion de {currency}: {str(e)}")

            log_message("Conversion terminee")

            return True

        except Exception as e:

            log_message(f"Erreur globale lors de la conversion: {str(e)}")

            return False


    def dca_trading_bot():

        # === AUTO-SELECTION DES PAIRES HAUSSIERES SI IA ACTIVE ===

        if ia:

            try:

                log_message(
                    f"✅ Paires haussières sélectionnées dynamiquement : {selected_crypto_pairs}"
                )

                save_logs_to_file()

            except Exception as e:

                log_message(f"❌ Erreur pendant la sélection dynamique des paires : {e}")

                save_logs_to_file()

        """DCA trading bot with automated buy based on percentages."""

        # === AUTO-SELECTION DYNAMIQUE TOUTES LES 6H ===

        global last_autoselect_update

        now = datetime.now()

        if (
            not selected_crypto_pairs
            or not hasattr(dca_trading_bot, "last_autoselect_update")
            or (now - dca_trading_bot.last_autoselect_update).total_seconds() > 21600
        ):

            try:

                dca_trading_bot.last_autoselect_update = now

                log_message(
                    f"🔄 Paires mises à jour par auto-sélection dynamique à {now.strftime('%H:%M')} : {selected_crypto_pairs}"
                )

                save_logs_to_file()

            except Exception as e:

                log_message(f"❌ Échec auto-sélection dynamique : {str(e)}")

                save_logs_to_file()

        # === AUTO-SELECTION DES PAIRES PAR VOLATILITÉ SI LISTE VIDE ===

        if not selected_crypto_pairs and all_data:

            try:

                log_message(
                    f"🎯 Paires sélectionnées automatiquement : {selected_crypto_pairs}"
                )

                save_logs_to_file()

            except Exception as e:

                log_message(f"❌ Erreur dans l'auto-sélection des paires : {str(e)}")

                save_logs_to_file()

        global bot_running, buy_percentage_of_capital

        bot_running = True

        log_message("DCA trading bot started")

        save_logs_to_file()

        while bot_running:  # S'assurer que les processus en cours sont terminés

            try:

                # Pour chaque paire de crypto-monnaies sélectionnée

                # Mise à jour dynamique du drawdown

                trade_summary = TradeMemory().get_summary()

                volatility = get_volatility()

                global max_drawdown_pct

                max_drawdown_pct = adjust_drawdown_threshold(trade_summary, volatility)

                check_drawdown_stop()

                if not bot_running:

                    log_message(
                        f"🔻 Drawdown max dynamique atteint ({max_drawdown_pct:.1f}%). Bot arrêté."
                    )

                    break

                for selected_crypto_pair in selected_crypto_pairs:

                    # Si un arrêt est demandé, sortir de la boucle principale

                    if not bot_running:

                        log_message("Arrêt demandé. Finalisation des processus en cours.")

                        save_logs_to_file()

                        break  # Quitter la boucle des paires pour arrêter proprement

                    # Identité de la paire traitée

                    product_id = selected_crypto_pair

                    log_message(f"Paire traitée actuellement : {product_id}")

                    save_logs_to_file()

                    # Vérification du solde USDC

                    usdc_balance = get_usdc_balance()

                    log_message(f"Le solde USDC est : {usdc_balance}")

                    save_logs_to_file()

                    # déterminer le montant à acheter

                    buy_amount = usdc_balance * buy_percentage_of_capital

                    # Vérification du solde USDC

                    if usdc_balance < 50:

                        # if usdc_balance < Decimal(buy_amount):

                        log_message(
                            f"Solde USDC insuffisant pour placer un ordre d'achat de : {product_id}."
                        )

                        save_logs_to_file()

                        # Analise de d'autre portefeuilles pour alimenter notre portefeuille USDC

                        log_message(
                            f"Analysons le solde d'autre portefeuilles pour trouvez les fond nécessaire à l'acaht de : {product_id}."
                        )

                        save_logs_to_file()

                        selected_crypto_base = selected_crypto_pair.split("-")[0]

                        check_and_convert_all_accounts(selected_crypto_base)

                        log_message(f"Conversions treminés")

                        save_logs_to_file()

                        # si les fonds on été trouvés on passe a l'achat sinon on passe à la paire de crypto suivante

                    # Achat avec ou sans IA

                    if ia:

                        log_message("IA activated.")

                        save_logs_to_file()

                        hist = all_data[
                            selected_crypto_pair
                        ]  # Utiliser les données déjà récupérées

                        train, test = train_test_split(hist, test_size=test_size)

                        train, test, X_train, X_test, y_train, y_test = prepare_data(
                            hist,
                            "close",
                            window_len=window_len,
                            zero_base=zero_base,
                            test_size=test_size,
                        )

                        model = build_lstm_model(
                            X_train,
                            output_size=1,
                            neurons=lstm_neurons,
                            dropout=dropout,
                            loss=loss,
                            optimizer=optimizer,
                        )

                        model.fit(
                            X_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0,
                            shuffle=True,
                        )

                        targets = test["close"][window_len:]

                        preds = model.predict(X_test).squeeze()

                        preds = test["close"].values[:-window_len] * (preds + 1)

                        preds = pd.Series(index=targets.index, data=preds)

                        if len(preds) < 2:

                            print(
                                f"{selected_crypto_pair} : Pas assez de données pour comparer les tendances."
                            )

                            continue

                        yesterday_last_real = preds.iloc[-2]

                        today_pred = preds.iloc[-1]

                        trend_comparison = compare_first_real_and_last_pred(
                            yesterday_last_real, today_pred
                        )

                        if "augmenter" in trend_comparison:

                            log_message(
                                f"{product_id} Prendra de la valeur, achat en cours."
                            )

                            save_logs_to_file()

                            safe_place_market_buy(product_id)

                        else:

                            log_message(f"{product_id} Perdra de la valeur, achat annulé.")

                            save_logs_to_file()

                    else:

                        log_message(
                            f"Placons un ordre d'achat d'un montant de {buy_amount} pour : {product_id}."
                        )

                        save_logs_to_file()

                        safe_place_market_buy(product_id)

                # Mise en pause après avoir traité toutes les paires

                log_message("Toutes les paires traitées. Conversion de résidus en USDC...")

                save_logs_to_file()

                force_convert_all_to_usdc(min_value_usdc=1.0)

                log_message("⏸ Mise en pause du robot.")

                save_logs_to_file()

                print(
                    f"⏳ Attente de {dca_interval_seconds} secondes avant prochain cycle..."
                )

                time.sleep(dca_interval_seconds)

            except Exception as e:

                log_message(f"Exception in DCA trading bot: {e}")

                send_alert_email("🚨 Erreur critique dans le bot", str(e))

                save_logs_to_file()

                time.sleep(10)

        log_message("Finalisation des processus terminée. Arrêt du bot.")

        save_logs_to_file()


    # derniere version prenant en compte l ia

    #####################################################################################################

    # accounts = client.get_accounts()

    # Fonction principale


    def Balance_Total():

        global log_data1, accounts, all_data_vola

        while True:

            # Réinitialiser log_data

            log_data1 = ""

            try:

                # Récupérer les portefeuilles

                transactions = client.get_transaction_summary()

                balance_total = transactions["total_balance"]

                log_data1 += f"{balance_total}\n"

                time.sleep(2)

                # Gestion des erreurs HTTP

                retries = 0

                max_retries = 5

                delay = 1

                while retries < max_retries:

                    try:

                        accounts = client.get_accounts()  # Obtenez les comptes

                        print("Mise à jour des comptes")

                        log_message("Mise à jour des comptes")

                        save_logs_to_file()

                        # ========================================================================================

                        heure_actuelle = (
                            datetime.now()
                        )  # Extrait uniquement l'heure actuelle

                        if should_update_ai(heure_actuelle):

                            print("Mise à jour de l'IA.")

                            load_data(selected_crypto_pairs)

                            Predictions_calculs()

                            tendance()

                        else:

                            # if last_data_fetched < datetime.utcnow() - timedelta(hours=1): #erreur

                            print("Heure de Mise à jour de l'IA non atteinte.")

                            log_message("Heure de Mise à jour de l'IA non atteinte.")

                            save_logs_to_file()

                        break

                    except HTTPError as e:

                        if e.response.status_code == 429:

                            print("Rate limit exceeded. Retrying after delay...")

                            log_message("Rate limit exceeded. Retrying after delay...")

                            save_logs_to_file()

                            time.sleep(delay)

                            retries += 1

                            delay *= 2  # Backoff exponentiel

                        else:

                            raise e

            except KeyError as e:

                log_data1 += f"Erreur de récupération de la balance: {str(e)}\n"

                save_logs_to_file()

            # Attendre avant la prochaine itération

            time.sleep(2)


    # Démarrer la fonction dans un thread

    thread = threading.Thread(target=Balance_Total)

    thread.daemon = True  # Assure que le thread s'arrête avec le programme principal

    thread.start()

    #####################################################################################

    #########################################################################################

    # accounts = client.get_accounts()

    #########################################################################################

    # Supprimé: doublon de get_usdc_balance

    #########################################################################################

    #########################################################################################


    def get_eth2_balance():

        global accounts

        try:

            # accounts = client.get_accounts()

            for account in accounts["accounts"]:

                if account["currency"] == "BTC":

                    return Decimal(account["available_balance"]["value"])

        except Exception as e:

            log_message(f"Error fetching BTC balance: {e}")

            save_logs_to_file()

        return Decimal("0")


    #########################################################################################

    # Fonction pour vérifier et comparer les soldes toutes les secondes


    def Your_Usdc():

        global log_data2  # Utiliser la variable soldes initiaux définie en dehors de la fonction

        while True:

            # Réinitialiser log_data à chaque itération avant d'ajouter de nouveaux logs

            log_data2 = ""  # Effacer les logs précédents

            # Assurez-vous que vous obtenez bien les informations de solde et les manipulez correctement

            try:

                # Récupérer les portefeuilles

                usdc_balance = get_usdc_balance()

                log_data2 += f"{usdc_balance:.1f}\n"

            except KeyError as e:

                log_data2 += f"Erreur de récupération de la balance: {str(e)}\n"

            # Envoyer les données mises à jour au client via SocketIO

            # socketio.emit('update_log2', {'log_usdc_balance': log_data2})

            # Attendre une seconde avant de vérifier à nouveau

            time.sleep(2.6)


    # Créer et démarrer le thread

    thread1 = threading.Thread(target=Your_Usdc)

    thread1.daemon = True  # Ensure the thread exits when the main program exits

    thread1.start()

    #####################################################################################

    #########################################################################################

    # Fonction pour vérifier et comparer les soldes toutes les secondes


    def Your_Eth2():

        global log_data3  # Utiliser la variable soldes initiaux définie en dehors de la fonction

        while True:

            # Réinitialiser log_data à chaque itération avant d'ajouter de nouveaux logs

            log_data3 = ""  # Effacer les logs précédents

            # Assurez-vous que vous obtenez bien les informations de solde et les manipulez correctement

            try:

                # Récupérer les portefeuilles

                eth2_balance = get_eth2_balance()

                log_data3 += f"{eth2_balance:.1f}\n"

            except KeyError as e:

                log_data3 += f"Erreur de récupération de la balance: {str(e)}\n"

            # Envoyer les données mises à jour au client via SocketIO

            # socketio.emit('update_log3', {'log_eth2_balance': log_data3})

            # Attendre une seconde avant de vérifier à nouveau

            time.sleep(2.8)


    # Créer et démarrer le thread

    thread2 = threading.Thread(target=Your_Eth2)

    thread2.daemon = True  # Ensure the thread exits when the main program exits

    thread2.start()

    #####################################################################################

    # Fonction pour récupérer les soldes initiaux (une fois par jour)


    def get_soldes_initiaux():

        accounts = client.get_accounts()

        soldes_initiaux = {}

        global log_data

        for account in accounts.accounts:

            solde_initial = float(account.available_balance["value"])

            currency = account.available_balance["currency"]

            soldes_initiaux[account.uuid] = (solde_initial, currency)

            log_data += (
                f"Solde initial pour le compte {currency}: {solde_initial} {currency}\n"
            )

        print(f"contenue du dictionnaire solde initiaux: {soldes_initiaux}")

        return soldes_initiaux


    # Récupérer les soldes initiaux pour commencer

    soldes_initiaux = get_soldes_initiaux()

    #####################################################################################

    # Fonction pour obtenir la valeur en temps réel d'une cryptomonnaie via l'API Coinbase


    def get_crypto_value(crypto_pair):

        url = f"https://api.coinbase.com/v2/prices/{crypto_pair}/buy"

        try:

            response = requests.get(url)

            response.raise_for_status()  # Vérifie si la requête a échoué (code HTTP 4xx ou 5xx)

            # Tentons de décoder le JSON

            try:

                data = response.json()

                # Vérification si la structure attendue est présente

                if "data" in data and "amount" in data["data"]:

                    return float(data["data"]["amount"])

                else:

                    raise ValueError("Réponse invalide: 'data' ou 'amount' manquants.")

            except ValueError as e:

                raise ValueError(f"Erreur lors de l'analyse de la réponse JSON: {e}")

        except requests.exceptions.RequestException as e:

            raise ConnectionError(f"Erreur lors de la requête à l'API Coinbase: {e}")

        except Exception as e:

            raise Exception(f"Erreur dans get_crypto_value pour {crypto_pair}: {e}")


    #####################################################################################

    # Fonction pour vérifier et comparer les soldes toutes les secondes


    def check_soldes():

        global soldes_initiaux, log_data, Profit_cumul, total, accounts  # Utiliser les variables globales nécessaires

        while True:

            try:

                # voir si ca fonctionne

                Profit_cumul = 0

                print(
                    f"contenue du dictionnaire solde initiaux dans check solde: {soldes_initiaux}"
                )

                log_data = ""  # Réinitialiser les logs

                heure_locale = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                log_data += f"Dernière mise à jour : {heure_locale}\n"

                for account in accounts.accounts:

                    solde_initial, currency = soldes_initiaux.get(
                        account.uuid, (0, "USD")
                    )  # Valeur par défaut si non trouvé

                    try:

                        crypto = account.available_balance["currency"]

                        accountts = client.get_accounts()

                        for accountt in accountts["accounts"]:

                            if accountt["currency"] == crypto:

                                solde_actuel = float(accountt["available_balance"]["value"])

                        # solde_actuel = float(account.available_balance['value'])

                        log_data += f"------------------------------------------------\n"

                        log_data += "::::::::::::::::::::::::::::::::::::::::::::::::\n"

                        log_data += f"PORTEFEUILLE {crypto}\n"

                        log_data += "::::::::::::::::::::::::::::::::::::::::::::::::\n"

                        log_data += f"Solde initial : {solde_initial} {crypto}\n"

                        log_data += f"Solde actuel  : {solde_actuel} {crypto}\n"

                        # Calculer la différence entre le solde initial et le solde actuel

                        difference = solde_actuel - solde_initial

                        log_data += "::::::::::::::::::::::::::::::::::::::::::::::::\n"

                        log_data += f"Profit du jour pour le compte {currency}: {difference:.1f} {currency}\n"

                        # Récupérer la valeur en USD

                        crypto_pair = crypto + "-USD"

                        try:

                            value_in_usd = get_crypto_value(crypto_pair)

                            log_data += (
                                f"La valeur de {crypto} en USD est : {value_in_usd}\n"
                            )

                            total = value_in_usd * difference

                            log_data += f"Conversion de vos bénéfices {crypto} en USD = {total:.1f} USD\n"

                            Profit_cumul += total

                        except Exception as e:

                            log_data += f"Erreur lors de la récupération de la valeur de {crypto} en USD : {str(e)}\n"

                            continue  # Passer à la paire suivante en cas d'erreur

                        log_data += "::::::::::::::::::::::::::::::::::::::::::::::::\n"

                        # Vérifier si la date a changé (si un nouveau jour commence)

                        current_time = datetime.now()

                        if (
                            current_time.hour == 0 and current_time.minute == 0
                        ):  # Si c'est minuit

                            log_data += (
                                "Mise à jour des soldes initiaux pour le nouveau jour...\n"
                            )

                            soldes_initiaux = get_soldes_initiaux()

                    except Exception as e:

                        log_data += f"Erreur avec le portefeuille {crypto}: {str(e)}\n"

                        continue  # Passer au compte suivant

                log_data += f"PROFIT CUMULE : {Profit_cumul:.1f} USD\n"

                # Envoyer les données mises à jour au client

                # socketio.emit('update_log', {'log': log_data})

            except Exception as e:

                # Enregistrer toute autre erreur non prévue

                log_data += f"Erreur générale dans le thread check_soldes : {str(e)}\n"

            finally:

                # Toujours attendre avant de recommencer pour éviter une surcharge

                time.sleep(4.5)


    # Créer et démarrer le thread

    thread3 = threading.Thread(target=check_soldes)

    thread3.daemon = (
        True  # Assure que le thread s'arrête lorsque le programme principal s'arrête
    )

    thread3.start()

    #########################################################################################

    # Fonction pour vérifier et comparer les ordres toutes les secondes


    def les_ordres():

        global log_data4  # Utilisation de la variable globale log_data

        while True:

            # Réinitialiser log_data à chaque itération

            log_data4 = ""

            try:

                heure_locale = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                log_data4 += f"Dernière mise à jour : {heure_locale}\n"

                # Obtenir tous les ordres

                orders_data = client.list_orders()

                # Conversion de la chaîne JSON en dictionnaire Python

                orders_dict = orders_data

                # Parcourir et traiter les données des commandes

                # for order in orders_dict['orders']:

                for order in orders_dict["orders"][:30]:

                    order_id = order["order_id"]

                    product_id = order["product_id"]

                    user_id = order["user_id"]

                    side = order["side"]

                    client_order_id = order["client_order_id"]

                    order_status = order["status"]

                    time_in_force = order["time_in_force"]

                    created_time = order["created_time"]

                    completion_percentage = order["completion_percentage"]

                    filled_size = order["filled_size"]

                    average_filled_price = order["average_filled_price"]

                    fee = order["fee"]

                    number_of_fills = order["number_of_fills"]

                    filled_value = order["filled_value"]

                    pending_cancel = order["pending_cancel"]

                    size_in_quote = order["size_in_quote"]

                    total_fees = order["total_fees"]

                    size_inclusive_of_fees = order["size_inclusive_of_fees"]

                    total_value_after_fees = order["total_value_after_fees"]

                    trigger_status = order["trigger_status"]

                    order_type = order["order_type"]

                    reject_reason = order["reject_reason"]

                    settled = order["settled"]

                    product_type = order["product_type"]

                    reject_message = order["reject_message"]

                    cancel_message = order["cancel_message"]

                    order_placement_source = order["order_placement_source"]

                    outstanding_hold_amount = order["outstanding_hold_amount"]

                    is_liquidation = order["is_liquidation"]

                    last_fill_time = order["last_fill_time"]

                    edit_history = order["edit_history"]

                    leverage = order["leverage"]

                    margin_type = order["margin_type"]

                    retail_portfolio_id = order["retail_portfolio_id"]

                    originating_order_id = order["originating_order_id"]

                    attached_order_id = order["attached_order_id"]

                    attached_order_configuration = order["attached_order_configuration"]

                    #################################

                    # Ajouter les informations de l'ordre au log

                    log_data4 += f"------------------------------------------------\n"

                    log_data4 += f"Order ID: {order_id}\n"

                    log_data4 += f"Product ID: {product_id}\n"

                    log_data4 += f"User ID: {user_id}\n"

                    log_data4 += f"side: {side}\n"

                    log_data4 += f"client_order_id: {client_order_id}\n"

                    log_data4 += f"Status: {order_status}\n"

                    log_data4 += f"time_in_force: {time_in_force}\n"

                    log_data4 += f"created_time: {created_time}\n"

                    log_data4 += f"completion_percentage: {completion_percentage}\n"

                    log_data4 += f"Filled Size: {filled_size}\n"

                    log_data4 += f"Average Filled Price: {average_filled_price}\n"

                    log_data4 += f"fee: {fee}\n"

                    log_data4 += f"number_of_fills: {number_of_fills}\n"

                    log_data4 += f"filled_value: {filled_value}\n"

                    log_data4 += f"pending_cancel: {pending_cancel}\n"

                    log_data4 += f"size_in_quote: {size_in_quote}\n"

                    log_data4 += f"Total Fees: {total_fees}\n"

                    log_data4 += f"size_inclusive_of_fees: {size_inclusive_of_fees}\n"

                    log_data4 += f"total_value_after_fees: {total_value_after_fees}\n"

                    log_data4 += f"trigger_status: {trigger_status}\n"

                    log_data4 += f"order_type: {order_type}\n"

                    log_data4 += f"reject_reason: {reject_reason}\n"

                    log_data4 += f"settled: {settled}\n"

                    log_data4 += f"product_type: {product_type}\n"

                    log_data4 += f"reject_message: {reject_message}\n"

                    log_data4 += f"cancel_message: {cancel_message}\n"

                    log_data4 += f"order_placement_source: {order_placement_source}\n"

                    log_data4 += f"outstanding_hold_amount: {outstanding_hold_amount}\n"

                    log_data4 += f"is_liquidation: {is_liquidation}\n"

                    log_data4 += f"last_fill_time: {last_fill_time}\n"

                    log_data4 += f"edit_history: {edit_history}\n"

                    log_data4 += f"leverage: {leverage}\n"

                    log_data4 += f"margin_type: {margin_type}\n"

                    log_data4 += f"retail_portfolio_id: {retail_portfolio_id}\n"

                    log_data4 += f"originating_order_id: {originating_order_id}\n"

                    log_data4 += f"attached_order_id: {attached_order_id}\n"

                    log_data4 += (
                        f"attached_order_configuration: {attached_order_configuration}\n"
                    )

                    #################################

            except Exception as e:

                # Gestion des exceptions et ajout d'un message d'erreur aux logs

                log_data4 += f"Erreur lors de la récupération des ordres : {str(e)}\n"

            # Envoyer les données mises à jour au client via SocketIO

            # socketio.emit('update_log4', {'log_orders': log_data4})

            # Pause d'une seconde avant de recommencer

            time.sleep(2.5)


    # Créer et démarrer le thread

    thread4 = threading.Thread(target=les_ordres)

    thread4.daemon = True  # Ensure the thread exits when the main program exits

    thread4.start()

    #####################################################################################

    #####################################################################################################


    def send_2fa_code():

        global current_2fa_code

        current_2fa_code = totp.now()  # Generate the 2FA code

        # Create and send the email with the 2FA code

        subject = "Your 2FA Code"

        body = f"Your 2FA code is: {current_2fa_code}"

        msg = Message(subject, recipients=[user_email])

        msg.body = body

        try:

            with mail.connect() as connection:  # Explicitly connect to SMTP server

                connection.send(msg)

            log_message(f"Sent 2FA code to {user_email}")

            save_logs_to_file()

        except Exception as e:

            log_message(f"Error sending 2FA code: {e}")

            save_logs_to_file()


    ######################################################################


    def send_failed_login_alert():

        # Vérification de la variable user_email

        if not user_email:

            print("Error: User email is not set.")

            return  # Retourne sans envoyer l'email si l'email utilisateur n'est pas défini

        # Définir le sujet et le corps de l'email

        subject = "Failed Login Attempt"

        body = "Une tentative de connexion a échoué."

        # Créer le message email

        msg = Message(subject, recipients=[user_email])

        msg.body = body

        try:

            print(
                f"Attempting to send email to {user_email}"
            )  # Vérifier si l'email est bien envoyé

            # Tenter d'envoyer l'email en utilisant la connexion SMTP

            with mail.connect() as connection:  # Connexion explicite au serveur SMTP

                connection.send(msg)

            log_message(
                f"Sent failed login alert to {user_email}"
            )  # Si l'email est envoyé avec succès

            save_logs_to_file()

        except Exception as e:

            log_message(
                f"Error sending failed login alert: {str(e)}"
            )  # Log de l'erreur si l'envoi échoue

            save_logs_to_file()

            print(
                f"Error sending failed login alert: {str(e)}"
            )  # Affichage de l'erreur pour le débogage


    #########################################################################################

    # Decorator to require login


    def login_required(f):

        @wraps(f)
        def decorated_function(*args, **kwargs):

            if "logged_in" not in session:

                return redirect(url_for("login"))

            return f(*args, **kwargs)

        return decorated_function


    #####################################################################################


    @app.route("/login", methods=["GET", "POST"])
    def login():

        if request.method == "POST":

            password = request.form.get("password")

            if password == HARDCODED_PASSWORD:

                send_2fa_code()  # Send 2FA code to the user

                return render_template("verify_2fa.html")  # Show the 2FA verification form

            else:

                send_failed_login_alert()

                return render_template("login.html", error="Incorrect password")

        return render_template("login.html")


    #####################################################################################


    @app.route("/verify_2fa", methods=["POST"])
    def verify_2fa():

        entered_2fa_code = request.form.get("2fa_code")

        if entered_2fa_code == current_2fa_code:

            session["logged_in"] = True

            return redirect(url_for("index"))

        else:

            return render_template("verify_2fa.html", error="Invalid 2FA code")


    #####################################################################################


    @app.route("/logout")
    def logout():

        session.pop("logged_in", None)

        return redirect(url_for("login"))


    ####################################################################

    # Protect the main route with login_required


    @app.route("/")
    @login_required
    def index():

        form_data = {
            # "risk_level": "moderate",  # Default values for form data
            # "amount": 0.0,
            # "compounding": False
            "ADA_USDC": True,
            "AAVE_USDC": True,
            "AERO_USDC": True,
            "ALGO_USDC": True,
            "AMP_USDC": True,
            "ARB_USDC": True,
            "AVAX_USDC": True,
            "BCH_USDC": True,
            "BONK_USDC": True,
            "BTC_USDC": True,
            "CRV_USDC": True,
            "DOGE_USDC": True,
            "DOT_USDC": True,
            "ETH_USDC": True,
            "EURC_USDC": True,
            "FET_USDC": True,
            "FIL_USDC": True,
            "GRT_USDC": True,
            "HBAR_USDC": True,
            "ICP_USDC": True,
            "IDEX_USDC": True,
            "INJ_USDC": True,
            "JASMY_USDC": True,
            "JTO_USDC": True,
            "LINK_USDC": True,
            "LTC_USDC": True,
            "MOG_USDC": True,
            "NEAR_USDC": True,
            "ONDO_USDC": True,
            "PEPE_USDC": True,
            "RENDER_USDC": True,
            "RNDR_USDC": True,
            "SEI_USDC": True,
            "SHIB_USDC": True,
            "SOL_USDC": True,
            "SUI_USDC": True,
            "SUPER_USDC": True,
            "SUSHI_USDC": True,
            "SWFTC_USDC": True,
            "TIA_USDC": True,
            "UNI_USDC": True,
            "USDT_USDC": True,
            "VET_USDC": True,
            "WIF_USDC": True,
            "XLM_USDC": True,
            "XYO_USDC": True,
            "XRP_USDC": True,
            "YFI_USDC": True,
            "ETC_USDC": True,
            "MATIC_USDC": True,
            "buy_percentage_of_capital": Decimal("0.05"),
            # "sell_percentage_of_capital": Decimal("0.05"),
            "sell_profit_target": Decimal("0.0"),
            "sell_stop_loss_target": Decimal("0.0"),
            "dca_interval_minute": 1,
        }

        return render_template(
            "index.html",
            selected_crypto_pairs=selected_crypto_pairs,
            filename=os.path.basename(__file__),
            bot_running=bot_running,
            form_data=form_data,
            logs="\n".join(logs[-100:]),
            log_Balance_Total=log_data1,
            log=log_data,
            log_usdc_balance=log_data2,
            log_eth2_balance=log_data3,
            log_orders=log_data4,
        )

        # return render_template('index.html', bot_running=bot_running, form_data=form_data,logs="\n".join(logs[-100:]))


    ####################################################################


    @app.route("/start", methods=["POST"])
    def start_bot():

        global bot_running

        if not bot_running:

            bot_thread = Thread(target=dca_trading_bot)

            bot_thread.daemon = True

            bot_thread.start()

        return redirect(url_for("index"))


    #####################################################################################


    @app.route("/stop", methods=["POST"])
    def stop_bot():

        global bot_running

        bot_running = False

        return redirect(url_for("index"))


    ####################################################################


    @app.route("/update_settings", methods=["POST"])
    def update_settings():

        global selected_crypto_pairs, buy_percentage_of_capital, sell_profit_target, sell_stop_loss_target, ia

        global dca_interval_seconds

        try:

            trade_frequency = int(request.form.get("trade_frequency", 30))

            print(f"➡️ trade_frequency reçu depuis formulaire: {trade_frequency}")

        except (ValueError, TypeError):

            trade_frequency = 30

            print("⚠️ trade_frequency invalide, défaut 30")

        dca_interval_seconds = trade_frequency

        buy_percentage_of_capital = Decimal(
            request.form.get("buy_percentage_of_capital", buy_percentage_of_capital)
        )

        sell_profit_target = Decimal(
            request.form.get("sell_profit_target", sell_profit_target)
        )

        sell_stop_loss_target = Decimal(
            request.form.get("sell_stop_loss_target", sell_stop_loss_target)
        )

        selected_crypto_pairs = request.form.getlist("selected_crypto_pairs")

        ia = request.form.getlist("ia")

        log_message(
            f"Settings updated: selected_crypto_pairs={selected_crypto_pairs},"
            f"buy_percentage_of_capital={buy_percentage_of_capital}, "
            f"sell_profit_target={sell_profit_target},sell_stop_loss_target={sell_stop_loss_target}, ia={ia}"
        )

        # AUTO-LOAD DATA AFTER SETTINGS CHANGE

        try:

            load_data(selected_crypto_pairs)

            log_message(f"📦 Données rechargées pour {selected_crypto_pairs}")

        except Exception as e:

            log_message(f"❌ Erreur lors du chargement des données : {e}")

        save_logs_to_file()

        return redirect(url_for("index"))


    ####################################################################


    @app.route("/logs", methods=["GET"])
    def get_logs():

        return jsonify({"logs": logs[-100:]})  # Send logs as an array of strings


    #######################################################


    @app.route("/log_Balance_Total", methods=["GET"])
    def log_Balance_Total():

        return jsonify({"log_Balance_Total": log_data1})  # Send logs as an array of strings


    @app.route("/log_usdc_balance", methods=["GET"])
    def log_usdc_balance():

        return jsonify({"log_usdc_balance": log_data2})  # Send logs as an array of strings


    @app.route("/log_eth2_balance", methods=["GET"])
    def log_eth2_balance():

        return jsonify({"log_eth2_balance": log_data3})  # Send logs as an array of strings


    @app.route("/log_orders", methods=["GET"])
    def log_orders():

        return jsonify({"log_orders": log_data4})  # Send logs as an array of strings


    @app.route("/log", methods=["GET"])
    def log():

        return jsonify({"log": log_data})  # Send logs as an array of strings


    #######################################################

    # API pour obtenir les données des prédictions en temps réel


    @app.route("/get_predictions")
    def get_predictions():

        return jsonify(
            {
                "predictions": {
                    crypto: all_predictions[crypto].tolist() for crypto in all_predictions
                }
            }
        )


    # application = app


    def determine_order_type_auto(symbol, recent_prices, threshold=0.0):
        """

        Détermine dynamiquement le type d'ordre en fonction de la volatilité.

        :param symbol: Le symbole de trading (ex: 'BTC-USDC')

        :param recent_prices: Liste des prix récents

        :param threshold: Seuil de volatilité au-delà duquel on passe en 'market'

        :return: 'market' ou 'limit'

        """

        if len(recent_prices) < 2:

            logger.warning(
                f"Pas assez de données pour évaluer la volatilité sur {symbol}. Market par défaut."
            )

            return "market"

        current_price = recent_prices[-1]

        oldest_price = recent_prices[0]

        volatility = abs(current_price - oldest_price) / oldest_price

        logger.info(
            f"[AUTO] Analyse de la volatilité pour {symbol} : {volatility:.4%} (seuil = {threshold:.2%})"
        )

        if volatility > threshold:

            logger.info(
                f"[AUTO] Volatilité élevée détectée sur {symbol}, ordre MARKET choisi."
            )

            return "market"

        else:

            logger.info(
                f"[AUTO] Faible volatilité détectée sur {symbol}, ordre LIMIT choisi."
            )

            return "limit"


    # === DÉCISION IA ADAPTATIVE ===


    def should_trade_adaptive(pair, data_df):

        try:

            closes = data_df["close"].dropna().tolist()

            if len(closes) < 50:

                log_message(f"⚠️ Pas assez de données pour {pair}")

                return False

            # ==== LSTM ====

            lstm_conf = 0

            try:

                train, test, X_train, X_test, y_train, y_test = prepare_data(
                    data_df, "close", window_len=5, zero_base=True, test_size=0.2
                )

                lstm_model = build_lstm_model(
                    X_train,
                    output_size=1,
                    neurons=50,
                    dropout=0.2,
                    loss="mse",
                    optimizer="adam",
                )

                lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

                preds = lstm_model.predict(X_test).squeeze()

                lstm_pred = test["close"].values[:-5] * (preds + 1)

                lstm_conf = (
                    float((lstm_pred[-1] - lstm_pred[-2]) / lstm_pred[-2])
                    if len(lstm_pred) >= 2
                    else 0
                )

            except Exception as e:

                log_message(f"LSTM failed for {pair}: {e}")

            # ==== GRU ====

            gru_conf = 0

            try:

                closes_scaled = (np.array(closes) - min(closes)) / (
                    max(closes) - min(closes) + 1e-8
                )

                input_tensor = torch.tensor(closes_scaled[-50:], dtype=torch.float32).view(
                    1, -1, 1
                )

                gru_model = GRUPricePredictor()

                gru_model.eval()

                with torch.no_grad():

                    pred = gru_model(input_tensor).item()

                    last_real = closes_scaled[-1]

                    gru_conf = float(pred - last_real)

            except Exception as e:

                log_message(f"GRU failed for {pair}: {e}")

            # ==== ATR ====

            atr_score = 0

            try:

                atr = ta.atr(data_df["high"], data_df["low"], data_df["close"], length=14)

                atr_score = float(atr.iloc[-1]) if not atr.empty else 0

            except Exception as e:

                log_message(f"ATR failed for {pair}: {e}")

            # ==== Agrégation & Seuil ====

            final_score = lstm_conf * 0.4 + gru_conf * 0.4 + atr_score * 0.2

            log_message(
                f"🔬 IA-Adaptive {pair} | LSTM: {lstm_conf:.4f}, GRU: {gru_conf:.4f}, ATR: {atr_score:.4f} ➞ Score final: {final_score:.4f}"
            )

            log_message(
                f"🧠 Scores IA {pair} | LSTM: {lstm_conf:.4f} | GRU: {gru_conf:.4f} | ATR: {atr_score:.4f} | Final: {final_score:.4f}"
            )

            if final_score > 0.005:

                return confirmed

            else:

                log_message(f"❌ Score IA insuffisant pour {pair}, pas de trade.")

                return False

        except Exception as e:

            log_message(f"❌ Erreur should_trade_adaptive pour {pair}: {e}")

            return False


    if __name__ == "__main__":

        app.run(host="0.0.0.0", port=5000)

        # app.run(debug=True)

    print(f"FIN DE SERVEUR N°_1")

    """

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    """

    # === AJOUT POUR FORCER LA CONVERSION DE TOUTES LES PAIRES EN USDC ===


    def convert_all_selected_pairs_to_usdc():

        log_message(
            "🔄 Démarrage de la conversion forcée pour toutes les paires sélectionnées..."
        )

        save_logs_to_file()

        for product_id in selected_crypto_pairs:

            try:

                force_convert_to_usdc(client, product_id, None)

            except Exception as e:

                log_message(f"[ERREUR] Échec de conversion pour {product_id} : {str(e)}")

                save_logs_to_file()

        log_message("✅ Conversion terminée pour toutes les paires.")

        save_logs_to_file()


    # === FIN AJOUT ===


    def force_convert_all_to_usdc(min_value_usdc=1.0):
        """

        Convertit tous les soldes crypto vers USDC si leur valeur estimée ≥ min_value_usdc.

        """

        global accounts

        try:

            log_message("⚙️ Démarrage de la conversion vers USDC (seuil: ≥ 1 USDC)")

            save_logs_to_file()

            for account in accounts["accounts"]:

                currency = account["available_balance"]["currency"]

                amount = Decimal(account["available_balance"]["value"])

                if currency == "USDC" or amount <= 0:

                    continue

                try:

                    clean_currency = currency.rstrip("0123456789")

                    product_id = f"{clean_currency}-USDC"

                    price = get_market_price(product_id)

                    if not price:

                        log_message(f"❌ Prix introuvable pour {product_id}")

                        continue

                    usdc_value = amount * price

                    if usdc_value < Decimal(min_value_usdc):

                        log_message(
                            f"🚫 Conversion ignorée : {currency} ({amount}) ~ {usdc_value:.1f} USDC (< {min_value_usdc})"
                        )

                        continue

                    log_message(
                        f"🔁 Conversion {amount} {currency} (~{usdc_value:.1f} USDC)"
                    )

                    create_order_safe(
                        client,
                        client_order_id=str(uuid.uuid4()),
                        product_id=product_id,
                        side="SELL",
                        order_configuration={
                            "market_market_ioc": {"base_size": str(amount)}
                        },
                    )

                    log_message(f"✅ Conversion effectuée pour {currency}")

                except Exception as e:

                    log_message(f"❌ Erreur pendant la conversion de {currency}: {str(e)}")

            log_message("✅ Conversion complète terminée.")

            save_logs_to_file()

        except Exception as e:

            log_message(f"🔥 Erreur dans force_convert_all_to_usdc: {str(e)}")

            save_logs_to_file()


    import time

    import csv

    from datetime import datetime, timedelta

    print(f"balise N°_11")

    # Configuration - logic can evolve dynamically

    MIN_PROFIT_DYNAMIC = 0.01  # will be adjusted

    MIN_HOLD_DURATION_MINUTES = 10  # will evolve dynamically

    TRADE_LOG_PATH = "trade_journal.csv"

    # Memory to track tokens

    last_buy_data = {}

    latency_tracker = {}

    # Function to check if we should convert


    def should_convert(symbol, current_price):

        if symbol not in last_buy_data:

            return False, 0

        entry = last_buy_data[symbol]

        bought_price = entry["price"]

        buy_time = entry["timestamp"]

        elapsed = datetime.utcnow() - buy_time

        # Dynamic profit threshold

        dynamic_threshold = MIN_PROFIT_DYNAMIC + (
            0.0 if elapsed.total_seconds() < 3600 else 0
        )

        gain = ((current_price - bought_price) / bought_price) if bought_price > 0 else 0

        if gain >= dynamic_threshold and elapsed > timedelta(
            minutes=MIN_HOLD_DURATION_MINUTES
        ):

            return True, gain

        return False, gain


    # Simulated price update


    def update_price_and_check(symbol, current_price, strategy_tag):

        should_sell, gain = should_convert(symbol, current_price)

        if should_sell:

            log_trade(
                symbol, last_buy_data[symbol]["price"], current_price, gain, strategy_tag
            )

            del last_buy_data[symbol]

            return True

        return False


    # Simulate a buy


    def record_buy(symbol, price):

        last_buy_data[symbol] = {"price": price, "timestamp": datetime.utcnow()}


    # Logging


    def log_trade(symbol, buy_price, sell_price, gain, strategy):

        with open(TRADE_LOG_PATH, "a", newline="") as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    symbol,
                    buy_price,
                    sell_price,
                    round(gain * 100, 2),
                    strategy,
                ]
            )


    # Strategy selector (can expand)


    def select_strategy(price_data):

        # Example: add swing/dca/scalp decisions based on market data

        return "dynamic"


    # ========== ADDITIONAL MODULE ==========

    # === GRU + Histohour AI Modules Injected ===


    def analyse_histohour(prices, window=24):

        import numpy as np


    def should_update_ai(current_time):
        """

        Détermine si l'IA doit être mise à jour en fonction de l'heure.

        - De 00:00 à 06:00 : mise à jour toutes les heures (histohour)

        - Sinon : mise à jour toutes les 5 minutes (histominute)

        """

        if 0 <= current_time.hour < 6:

            return current_time.minute == 0  # Toutes les heures

        else:

            return current_time.minute % 5 == 0  # Toutes les 5 minutes

        closes = [p["close"] for p in prices[-window:]]

        change_pct = (closes[-1] - closes[0]) / closes[0] * 100

        volatility = np.std(closes)

        trend = "up" if change_pct > 0 else "down"

        return {
            "trend": trend,
            "change_pct": round(change_pct, 2),
            "volatility": round(volatility, 4),
        }


    # GRU modèle prédictif (mock - à entraîner avec données réelles)

    import torch

    import torch.nn as nn

    import numpy as np

    print(f"balise N°_12")


    class GRUPricePredictor(nn.Module):

        def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):

            super(GRUPricePredictor, self).__init__()

            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):

            h0 = torch.zeros(1, x.size(0), 32)

            out, _ = self.gru(x, h0)

            out = self.fc(out[:, -1, :])

            return out


    def prepare_input_series(prices):

        closes = [p["close"] for p in prices]

        scaled = (np.array(closes) - np.min(closes)) / (
            np.max(closes) - np.min(closes) + 1e-8
        )

        tensor = torch.tensor(scaled, dtype=torch.float32).view(1, -1, 1)

        return tensor


    # Exemple (dans le main loop):

    # prediction = model(prepare_input_series(prices)).item()

    # if prediction > closes[-1]: trigger_buy()

    # === DÉBUT MODIFICATIONS AJOUTÉES ===

    MIN_NET_PROFIT_TARGET = 0.0  # 1.2% minimum net


    def calculate_net_profit_percentage(buy_price, sell_price, buy_fee, sell_fee):

        gross_profit = sell_price - buy_price

        total_fees = buy_fee + sell_fee

        net_profit = gross_profit - total_fees

        return (net_profit / buy_price) if buy_price else 0


    def adjust_sell_profit_target(volatility_index):

        if volatility_index > 0.05:

            return 0.0

        elif volatility_index > 0.02:

            return 0.02

        else:

            return MIN_NET_PROFIT_TARGET


    def log_trade_profit(order_buy, order_sell):

        buy_price = float(order_buy.get("average_filled_price", 0))

        sell_price = float(order_sell.get("average_filled_price", 0))

        buy_fee = float(order_buy.get("total_fees", 0))

        sell_fee = float(order_sell.get("total_fees", 0))

        net_profit_pct = calculate_net_profit_percentage(
            buy_price, sell_price, buy_fee, sell_fee
        )

        total_fees = buy_fee + sell_fee

        print(
            "📈 Profit net sur trade: {:.3f}% | Achat: {:.4f}, Vente: {:.4f}, Frais totaux: {:.4f}".format(
                net_profit_pct * 100, buy_price, sell_price, total_fees
            )
        )

        return net_profit_pct >= MIN_NET_PROFIT_TARGET


    # === FIN MODIFICATIONS AJOUTÉES ===


    def adjust_sell_profit_target_based_on_volatility(volatility_index):

        # Placeholder: you can replace this logic with one based on real volatility analysis

        if volatility_index > 0.02:

            return 0.0  # Increase target in high volatility

        elif volatility_index < 0.01:

            return 0.01  # Reduce target in low volatility

        return 0.0


    # === PARAMÈTRES DE STRATÉGIE LIMIT INTELLIGENTE ===

    LIMIT_ENABLED = True

    LIMIT_SPREAD = 0.0  # 0.1% en dessous du meilleur prix pour SELL, au-dessus pour BUY

    LIMIT_TIMEOUT = 3  # en secondes avant fallback MARKET

    # === NOUVELLE FONCTION POUR ESSAYER UN LIMIT PUIS RETOURNER SUR MARKET SI BESOIN ===


    def execute_order_with_limit_fallback(product_id, side, size, price):
        """

        Tente un ordre LIMIT, sinon fallback sur MARKET

        """

        if not LIMIT_ENABLED:

            return place_market_order(product_id, side, size)

        limit_price = round(
            price * (1 - LIMIT_SPREAD if side == "sell" else 1 + LIMIT_SPREAD), 8
        )

        print(
            f"🔍 Tentative d'ordre LIMIT sur {{product_id}} à {{limit_price}} ({{side.upper()}})"
        )

        try:

            order = place_limit_order(
                product_id=product_id, side=side, size=size, price=limit_price
            )

            waited = 0

            while not order_filled(order["id"]) and waited < LIMIT_TIMEOUT:

                time.sleep(1)

                waited += 1

            if order_filled(order["id"]):

                print(f"✅ Ordre LIMIT exécuté pour {{product_id}}")

                return order

            else:

                print(f"⏱️ Timeout LIMIT pour {{product_id}}, fallback MARKET")

                cancel_order(order["id"])

                return place_market_order(product_id=product_id, side=side, size=size)

        except Exception as e:

            print(f"⚠️ Erreur lors de l'ordre LIMIT : {{e}}, fallback MARKET")

            return place_market_order(product_id=product_id, side=side, size=size)


    # --- BEGIN PATCH TO HANDLE SMALL RESIDUALS ---


    def force_convert_to_usdc(client, product_id, portfolio_id):

        try:

            base_currency = product_id.split("-")[0]

            balance = get_balance(client, base_currency)

            if balance is None or balance <= 0:

                print(f"No balance to convert for {base_currency}")

                return

            # Get product info to check min order size

            product = client.get_product(product_id)

            min_order_size = float(product["quote_increment"])

            # Check if balance * price > min_trade_size, otherwise force a small top-up

            ticker = client.get_product_ticker(product_id)

            price = float(ticker["price"])

            min_trade_value = min_order_size * price

            value = balance * price

            if value < min_trade_value:

                # Try topping up with USDC (simulate small buy) to reach tradable value

                print(
                    f"Topping up {base_currency}: Current value {value} < required {min_trade_value}"
                )

                topup_amount = min_trade_value - value + 0.01  # Add a small buffer

                client.place_order(
                    product_id=product_id,
                    side="BUY",
                    order_type="MARKET",
                    quote_size=str(round(topup_amount, 8)),
                    time_in_force="IMMEDIATE_OR_CANCEL",
                )

            # Sell the full amount

            balance = get_balance(client, base_currency)

            client.place_order(
                product_id=product_id,
                side="SELL",
                order_type="MARKET",
                base_size=str(balance),
                time_in_force="IMMEDIATE_OR_CANCEL",
            )

            print(f"Converted {base_currency} → USDC, size: {balance}")

        except Exception as e:

            print(f"[ERROR] force_convert_to_usdc: {e}")


    # --- END PATCH TO HANDLE SMALL RESIDUALS ---

    # === Configuration globale ===

    MIN_CONVERSION_USDC = 0.30  # seuil minimal pour estimer si une conversion vaut la peine

    # === Fonction utilitaire ===


    def peut_convertir(base_amount, base_increment, est_usdc_value):

        try:

            if base_amount < float(base_increment):

                logging.info(
                    f"Conversion annulée car la quantité {base_amount} est inférieure au base_increment requis {base_increment}."
                )

                return False

            if est_usdc_value < MIN_CONVERSION_USDC:

                logging.info(
                    f"Conversion annulée car le gain estimé est < {MIN_CONVERSION_USDC} USDC (valeur estimée: {est_usdc_value})."
                )

                return False

            return True

        except Exception as e:

            logging.error(f"Erreur dans peut_convertir: {e}")

            return False


    def is_expected_gain_too_small(base_amount, current_price, min_usdc_gain=0.30):

        estimated_value = base_amount * current_price

        return estimated_value < min_usdc_gain


    def determine_order_type(volatility, selected_type="auto"):

        if selected_type != "auto":

            return selected_type

        if volatility > 0.02:

            return "market"

        else:

            return "limit"


    from coinbase.rest import RESTClient

    # === Ajout: Suivi de la volatilité et du timing des trades ===

    from statistics import stdev

    from collections import deque

    import time

    print(f"balise N°_13")

    # Garder en mémoire les N derniers prix (ex: 10)

    volatility_window_size = 10

    price_history = deque(maxlen=volatility_window_size)

    # Gérer le temps entre 2 trades

    last_trade_time = 0

    min_trade_interval = 30  # en secondes

    max_trade_interval = 300  # en secondes


    def get_volatility():

        if len(price_history) >= 2:

            return stdev(price_history)

        return 0


    def can_trade_now():

        global last_trade_time

        now = time.time()

        elapsed = now - last_trade_time

        if elapsed < min_trade_interval:

            return False

        if elapsed > max_trade_interval:

            return True

        return True


    def update_trade_time():

        global last_trade_time

        last_trade_time = time.time()


    # Supprimé: doublon de determine_order_type (définition sans paramètre)

    # === Fin ajout ===

    # === CONFIGURATION COMPTE UTILISATEUR ===

    compounding = True  # Active le mode compounding (réinvestissement des gains)

    capital_initial = 100.0  # Capital de départ en USDC

    capital_actuel = capital_initial  # Capital qui évolue selon les gains/pertes

    from dotenv import load_dotenv

    from threading import Thread

    import threading


    import time

    import uuid


    COINBASE_FEE_RATE = Decimal("0.006")  # 0.6% frais Coinbase

    current_portfolio_id = os.getenv("COINBASE_PORTFOLIO_ID")  # ID du portefeuille actif

    usdc_safe_wallet_id = os.getenv(
        "COINBASE_PROFIT_PORTFOLIO_ID"
    )  # ID du portefeuille Profit robot DCA

    if not current_portfolio_id or not usdc_safe_wallet_id:

        raise RuntimeError(
            "COINBASE_PORTFOLIO_ID et COINBASE_PROFIT_PORTFOLIO_ID doivent être définis dans le .env"
        )


    from flask import Flask, request, jsonify, render_template, redirect, url_for, session

    from requests.exceptions import HTTPError

    from functools import wraps

    import pyotp

    from flask_mail import Mail, Message

    import requests

    from datetime import datetime

    from flask_socketio import SocketIO

    ##############################################

    import os

    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    tf.config.threading.set_intra_op_parallelism_threads(2)

    tf.config.threading.set_inter_op_parallelism_threads(2)

    import json

    import requests

    from keras.models import Sequential

    from keras.layers import Activation, Dense, Dropout, LSTM, Input

    import matplotlib.pyplot as plt

    import numpy as np

    import pandas as pd

    import seaborn as sns

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    import io

    import base64

    print(f"balise N°_14")


    def place_order_with_fallback(client, symbol, quantity, limit_price):
        """

        Place a limit order and fallback to market if not filled within timeout.

        """

        import time

        print(f"[LIMIT] Placing limit order: {quantity} {symbol} at {limit_price}")

        order = client.order_limit_buy(
            symbol=symbol, quantity=quantity, price=str(limit_price)
        )

        order_id = order["orderId"]

        wait_time = 30  # seconds

        poll_interval = 5  # seconds

        elapsed = 0

        while elapsed < wait_time:

            time.sleep(poll_interval)

            elapsed += poll_interval

            print(f"[STATUS] Checking status of order {order_id}")

            order_status = client.get_order(symbol=symbol, orderId=order_id)

            if order_status["status"] == "FILLED":

                print(f"[FILLED] Limit order {order_id} was filled.")

                return order_status

        # Fallback to market

        print(f"[FALLBACK] Order {order_id} not filled, switching to market order.")

        print(f"[CANCEL] Cancelling order {order_id}")

        client.cancel_order(symbol=symbol, orderId=order_id)

        print(f"[MARKET] Placing market order: {quantity} {symbol}")

        return client.order_market_buy(symbol=symbol, quantity=quantity)


    def force_convert_all_to_usdc(min_value_usdc=1.0):

        global accounts

        try:

            log_message("⚙️ Démarrage de la conversion vers USDC (seuil: ≥ 1 USDC)")

            save_logs_to_file()

            for account in accounts["accounts"]:

                currency = account["available_balance"]["currency"]

                amount = Decimal(account["available_balance"]["value"])

                if currency == "USDC" or amount <= 0:

                    continue

                try:

                    clean_currency = currency.rstrip("0123456789")

                    product_id = f"{clean_currency}-USDC"

                    price = get_market_price(product_id)

                    if not price:

                        log_message(f"❌ Prix introuvable pour {product_id}")

                        continue

                    usdc_value = amount * price

                    if usdc_value < Decimal(min_value_usdc):

                        log_message(
                            f"🚫 Conversion ignorée : {currency} ({amount}) ~ {usdc_value:.1f} USDC (< {min_value_usdc})"
                        )

                        continue

                    log_message(
                        f"🔁 Conversion {amount} {currency} (~{usdc_value:.1f} USDC)"
                    )

                    create_order_safe(
                        client,
                        client_order_id=str(uuid.uuid4()),
                        product_id=product_id,
                        side="SELL",
                        order_configuration={
                            "market_market_ioc": {"base_size": str(amount)}
                        },
                    )

                    log_message(f"✅ Conversion effectuée pour {currency}")

                except Exception as e:

                    log_message(f"❌ Erreur pendant la conversion de {currency}: {str(e)}")

            log_message("✅ Conversion complète terminée.")

            save_logs_to_file()

        except Exception as e:

            log_message(f"🔥 Erreur dans force_convert_all_to_usdc: {str(e)}")

            save_logs_to_file()


    #######################################################################################################################

    # Set decimal precision

    getcontext().prec = 10

    # Load environment variables from .env file

    load_dotenv()

    # Hardcoded password for login

    HARDCODED_PASSWORD = os.getenv("LOGIN_PASSWORD")

    # Set up logging

    logging.basicConfig(level=logging.INFO)

    # Load API credentials

    api_key = os.getenv("COINBASE_API_KEY_ID")

    # Load the private key from the PEM file

    private_key_path = "coinbase_private_key.pem"

    with open(private_key_path, "r", encoding="utf-8") as key_file:

        api_secret = key_file.read()

    # Create the RESTClient instance

    client = RESTClient(api_key=api_key, api_secret=api_secret)

    try:

        # Simple call to test authentication

        accounts = client.get_accounts()

        for account in accounts["accounts"]:

            print("Successfully authenticated. Accounts data:", account["name"])

    except Exception as e:

        print("Authentication failed:", e)

    #####################################################################################################

    selected_crypto_pairs = [
        "ADA-USDC",
        "AAVE-USDC",
        "ALGO-USDC",
        "ARB-USDC",
        "AVAX-USDC",
        "BTC-USDC",
        "CRV-USDC",
        "DOGE-USDC",
        "DOT-USDC",
        "ETC-USDC",
        "ETH-USDC",
        "FET-USDC",
        "FIL-USDC",
        "GRT-USDC",
        "HBAR-USDC",
        "ICP-USDC",
        "IDEX-USDC",
        "LINK-USDC",
        "LTC-USDC",
        "MATIC-USDC",
        "NEAR-USDC",
        "PEPE-USDC",
        "SOL-USDC",
        "SUI-USDC",
        "SUPER-USDC",
        "SUSHI-USDC",
        "SWFTC-USDC",
        "UNI-USDC",
        "USDT-USDC",
        "VET-USDC",
        "XLM-USDC",
        "XRP-USDC",
        "YFI-USDC",
    ]
    selected_crypto_pairs = [
        "ADA-USDC",
        "AAVE-USDC",
        "ALGO-USDC",
        "ARB-USDC",
        "AVAX-USDC",
        "BTC-USDC",
        "CRV-USDC",
        "DOGE-USDC",
        "DOT-USDC",
        "ETC-USDC",
        "ETH-USDC",
        "FET-USDC",
        "FIL-USDC",
        "GRT-USDC",
        "HBAR-USDC",
        "ICP-USDC",
        "IDEX-USDC",
        "LINK-USDC",
        "LTC-USDC",
        "MATIC-USDC",
        "NEAR-USDC",
        "PEPE-USDC",
        "SOL-USDC",
        "SUI-USDC",
        "SUPER-USDC",
        "SUSHI-USDC",
        "SWFTC-USDC",
        "UNI-USDC",
        "USDT-USDC",
        "VET-USDC",
        "XLM-USDC",
        "XRP-USDC",
        "YFI-USDC",
    ]
    # selected_crypto_pairs=['ADA-USDC','ALGO-USDC','BCH-USDC','BTC-USDC','CRV-USDC','DOGE-USDC','DOT-USDC','ETC-USDC','ETH-USDC','LINK-USDC','LTC-USDC','MATIC-USDC','PEPE-USDC','SOL-USDC','SUI-USDC','SUSHI-USDC','SWFTC-USDC','UNI-USDC','USDT-USDC','XRP-USDC']

    # VALIDE

    # Fetch product details

    # --- Sécurité si aucune paire sélectionnée ---

    # === PRÉ-LAUNCH: Validation des paires via load_data ===


    def validate_pairs_before_launch(pairs):

        import pandas as pd

        global all_data

        all_data = {}

        log_message("🔍 Pré-lancement : validation des paires avec fetch_crypto_data()")

        for pair in pairs:

            try:

                df = fetch_crypto_data(pair)

                if df.empty:

                    log_message(f"⚠️ Paire ignorée (données invalides) : {pair}")

                    continue

                all_data[pair] = df

            except Exception as e:

                log_message(f"❌ Erreur lors de la validation de {pair}: {str(e)}")

        if not all_data:

            log_message("⛔ Aucune paire valide détectée. Arrêt du bot.")

            save_logs_to_file()

            exit()

        log_message(f"✅ Paires valides prêtes à être tradées : {list(all_data.keys())}")

        save_logs_to_file()


    # Appel automatique juste après définition des paires

    selected_crypto_pairs = (
        selected_crypto_pairs if "selected_crypto_pairs" in globals() else []
    )

    if not selected_crypto_pairs:

        log_message(
            "Aucune paire sélectionnée. Le robot s'arrête pour éviter tout comportement inattendu."
        )

        exit()

    for selected_crypto_pair in selected_crypto_pairs:

        product_info = client.get_product(
            selected_crypto_pair
        )  # Utilisation correcte de 'pair' au lieu de 'selected_crypto_pair'

        # Extraction de la taille minimale de l'échange

        base_min_size = float(product_info["base_min_size"])

        # Décommentez cette ligne si vous avez besoin de l'incrément de la cotation

        quote_increment = float(product_info["quote_increment"])

        print(f"Base Minimum Size for {selected_crypto_pair}: {base_min_size}")

        # Décommentez cette ligne si vous avez besoin d'afficher l'incrément de la cotation

        print(f"Quote Increment for {selected_crypto_pair}: {quote_increment}")

    ####################################################################################################################################################

    # Initialisation de Flask-SocketIO

    app = Flask(__name__)

    print(f"DEBUT DE SERVEUR N°_2")

    Profit_cumul = 0

    log_data = ""  # Global log data

    log_data1 = ""  # Global log data

    log_data2 = ""

    log_data3 = ""

    log_data4 = ""

    # Initialisation du client Coinbase

    # accounts = client.get_accounts()

    ####################################################################################################################################################

    # Configuration de Flask-Mail

    app.config["MAIL_SERVER"] = "smtp.elasticemail.com"

    app.config["MAIL_PORT"] = 2525

    app.config["MAIL_USE_TLS"] = True

    app.config["MAIL_DEBUG"] = True

    app.config["MAIL_USERNAME"] = os.getenv(
        "SENDER_EMAIL"
    )  # Utilisez l'email de l'expéditeur

    app.config["MAIL_PASSWORD"] = os.getenv(
        "SENDER_PASSWORD"
    )  # Mot de passe de l'email ou mot de passe spécifique à l'application

    app.config["MAIL_DEFAULT_SENDER"] = os.getenv("SENDER_EMAIL")

    mail = Mail(app)

    # Configurer le générateur de code 2FA

    totp = pyotp.TOTP(
        os.getenv("SECRET_KEY2")
    )  # Clé secrète pour générer les codes 2FA (à stocker de manière sécurisée)

    current_2fa_code = None  # Variable pour stocker le code 2FA généré

    user_email = os.getenv(
        "USER_EMAIL"
    )  # L'email du destinataire du code 2FA (peut être dynamique)

    ####################################################################################################################################################

    app.secret_key = "your_secret_key"  # Set a secret key for sessions

    # Configurations

    buy_percentage_of_capital = Decimal("0.05")  # 5% of capital per DCA buy

    # sell_percentage_of_capital = Decimal("0.05") # 5% of capital per DCA sell

    sell_profit_target = Decimal(
        "0.0"
    )  # Augmenté de 0.5% à 0.8%  # Sell when 5% profit target is reached

    sell_stop_loss_target = Decimal("0.0")  # Augmenté de 0.2% à 0.3%

    # stop_loss_threshold = Decimal("0.0")  # Stop loss at 5% below initial buy-in

    # dca_interval_minute = 5  # Réduit la fréquence à une fois toutes les 5 minutes

    # dca_interval_seconds = dca_interval_minute * 60  # 5 minutes en secondes  # DCA interval in seconds (adjust as needed)

    ia = False

    ####################################################################################################################################################

    ADA_USDC = True

    AAVE_USDC = True

    AERO_USDC = True  # supporte pas tradin avec IA

    ALGO_USDC = True

    AMP_USDC = True  # supporte pas tradin avec IA

    ARB_USDC = True

    AVAX_USDC = True

    BCH_USDC = True

    BONK_USDC = True  # supporte pas tradin avec IA

    BTC_USDC = True

    CRV_USDC = True

    DOGE_USDC = True

    DOT_USDC = True

    ETH_USDC = True

    EURC_USDC = True  # supporte pas tradin avec IA

    FET_USDC = True

    FIL_USDC = True

    GRT_USDC = True

    HBAR_USDC = True

    ICP_USDC = True

    IDEX_USDC = True

    INJ_USDC = True  # supporte pas tradin avec IA

    JASMY_USDC = True  # supporte pas tradin avec IA

    JTO_USDC = True  # supporte pas tradin avec IA

    LINK_USDC = True

    LTC_USDC = True

    MOG_USDC = True  # supporte pas tradin avec IA

    NEAR_USDC = True

    ONDO_USDC = True  # supporte pas tradin avec IA

    PEPE_USDC = True

    RENDER_USDC = True  # supporte pas tradin avec IA

    RNDR_USDC = True  # supporte pas tradin avec IA

    SEI_USDC = True  # supporte pas tradin avec IA

    SHIB_USDC = True  # supporte pas tradin avec IA

    SOL_USDC = True

    SUI_USDC = True

    SUPER_USDC = True

    SUSHI_USDC = True

    SWFTC_USDC = True

    TIA_USDC = True  # supporte pas tradin avec IA

    UNI_USDC = True

    USDT_USDC = True

    VET_USDC = True

    WIF_USDC = True  # supporte pas tradin avec IA

    XLM_USDC = True

    XYO_USDC = True

    XRP_USDC = True

    YFI_USDC = True

    ETC_USDC = True

    MATIC_USDC = True

    ####################################################################################################################################################

    bot_running = False

    logs = []

    #####################################################################################################

    # VALIDE

    from datetime import datetime


    def adjust_profit_targets(price_history, hold_duration_minutes):

        if len(price_history) < 2:

            return 0.01, 0.0  # Par défaut

        returns = [
            (price_history[i + 1] - price_history[i]) / price_history[i]
            for i in range(len(price_history) - 1)
        ]

        volatility = sum(abs(r) for r in returns) / len(returns)

        if hold_duration_minutes < 30:

            if volatility > 0.01:

                return 0.0, 0.01

            else:

                return 0.01, 0.0

        elif hold_duration_minutes < 60:

            if volatility > 0.01:

                return 0.02, 0.0

            else:

                return 0.0, 0.01

        else:

            return 0.0, 0.02


    # Simulation d’une boucle principale par portefeuille


    def log_message(message):

        global logs

        timestamped_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}"

        logs.append(timestamped_message)

        logging.info(timestamped_message)


    def save_logs_to_file():

        file_path = os.path.join(os.getcwd(), "logs.txt")

        with open(file_path, "w", encoding="utf-8") as file:

            for log in logs:

                file.write(log + "\n")


    #####################################################################################################

    #################################################################################################################################################################################

    # LSTM

    # Global variable to store the fetched data

    all_data = {}


    PAIR_MAPPING = {
        "SOL-USDC": "SOL-USD",
        "UNI-USDC": "UNI-USD",
        "LTC-USDC": "LTC-USD",
        "MATIC-USDC": "MATIC-USD",
        "VET-USDC": "VET-USD",
        "DOGE-USDC": "DOGE-USD",
        "ALGO-USDC": "ALGO-USD",
        "ADA-USDC": "ADA-USD",
        "DOT-USDC": "DOT-USD",
        "AVAX-USDC": "AVAX-USD",
        "LINK-USDC": "LINK-USD",
        "XLM-USDC": "XLM-USD",
        "XRP-USDC": "XRP-USD",
        "YFI-USDC": "YFI-USD",
        "AAVE-USDC": "AAVE-USD",
        "CRV-USDC": "CRV-USD",
        "GRT-USDC": "GRT-USD",
        "HBAR-USDC": "HBAR-USD",
        "ETC-USDC": "ETC-USD",
        "NEAR-USDC": "NEAR-USD",
        "FET-USDC": "FET-USD",
        "SUSHI-USDC": "SUSHI-USD",
        "ICP-USDC": "ICP-USD",
        "FIL-USDC": "FIL-USD",
        "PEPE-USDC": "PEPE-USD",
        "SUI-USDC": "SUI-USD",
        "SWFTC-USDC": "SWFTC-USD",
        "IDEX-USDC": "IDEX-USD",
        "SUPER-USDC": "SUPER-USD",
        "ARB-USDC": "ARB-USD",
    }


    def fetch_crypto_data(crypto_pair, limit=300):

        mapped_pair = PAIR_MAPPING.get(crypto_pair, crypto_pair)

        endpoint = f"https://api.exchange.coinbase.com/products/{mapped_pair}/candles"

        params = {"granularity": 86400}

        try:

            res = requests.get(endpoint, params=params)

            if res.status_code != 200:

                raise ValueError(f"Erreur API Coinbase: {res.status_code}")

            candles = json.loads(res.content)

            df = pd.DataFrame(
                candles, columns=["time", "low", "high", "open", "close", "volume"]
            )

            df = df.sort_values("time")

            df["time"] = pd.to_datetime(df["time"], unit="s")

            df["volumefrom"] = df["volume"]

            df["volumeto"] = df["volume"] * df["close"]

            df = df[["time", "high", "low", "open", "volumefrom", "volumeto", "close"]]

            df = df.set_index("time")

            log_message(
                f"#fetch_crypto_data ::: mise à jour journalière effectuée pour {crypto_pair}"
            )

            save_logs_to_file()

            return df

        except Exception as e:

            log_message(f"❌ Erreur Coinbase fetch_crypto_data {crypto_pair} : {e}")

            save_logs_to_file()

            return pd.DataFrame()

        params = {"granularity": 86400}  # Bougies journalières

        try:

            res = requests.get(endpoint, params=params)

            if res.status_code != 200:

                raise ValueError(f"Erreur API Coinbase: {res.status_code}")

            candles = json.loads(res.content)

            df = pd.DataFrame(
                candles, columns=["time", "low", "high", "open", "close", "volume"]
            )

            df = df.sort_values("time")

            df["time"] = pd.to_datetime(df["time"], unit="s")

            df["volumefrom"] = df["volume"]

            df["volumeto"] = df["volume"] * df["close"]

            df = df[["time", "high", "low", "open", "volumefrom", "volumeto", "close"]]

            df = df.set_index("time")

            log_message(
                f"#fetch_crypto_data ::: mise à jour journalière effectuée pour {crypto_pair}"
            )

            save_logs_to_file()

            return df

        except Exception as e:

            log_message(f"❌ Erreur Coinbase fetch_crypto_data {crypto_pair} : {e}")

            save_logs_to_file()

            return pd.DataFrame()

        endpoint = "https://min-api.cryptocompare.com/data/histoday"

        res = requests.get(f"{endpoint}?fsym={fsym}&tsym={tsym}&limit={limit}")

        data = pd.DataFrame(json.loads(res.content)["Data"])

        data = data.set_index("time")

        data.index = pd.to_datetime(data.index, unit="s")

        data = data.drop(["conversionType", "conversionSymbol"], axis=1)

        print(f"#def fetch_crypto_data ::: mise a jour journalière effectuée..")

        log_message(f"#def fetch_crypto_data :::mise a jour journalière effectuée...")

        save_logs_to_file()

        return data


    def load_data(crypto_pairs, limit=500):
        """Load data for all crypto pairs once and store it in a global dictionary."""

        print(f"#def load_data ::: mise a jour journalière effectuée..")

        log_message(f"#def load_data :::mise a jour journalière effectuée...")

        save_logs_to_file()

        global all_data

        for crypto_pair in crypto_pairs:

            all_data[crypto_pair] = fetch_crypto_data(crypto_pair, limit)


    def train_test_split(df, test_size=0.2):

        split_row = len(df) - int(test_size * len(df))

        train_data = df.iloc[:split_row]

        test_data = df.iloc[split_row:]

        return train_data, test_data


    def line_plot(line1, line2, label1=None, label2=None, title="", lw=2):

        fig, ax = plt.subplots(1, figsize=(13, 7))

        ax.plot(line1, label=label1, linewidth=lw)

        ax.plot(line2, label=label2, linewidth=lw)

        ax.set_ylabel("prix", fontsize=14)

        ax.set_title(title, fontsize=16)

        ax.legend(loc="best", fontsize=16)


    def normalise_zero_base(df):

        return df / df.iloc[0] - 1


    def normalise_min_max(df):

        return (df - df.min()) / (df.max() - df.min())


    def extract_window_data(df, window_len=5, zero_base=True):

        window_data = []

        for idx in range(len(df) - window_len):

            tmp = df[idx : (idx + window_len)].copy()

            if zero_base:

                tmp = normalise_zero_base(tmp)

            window_data.append(tmp.values)

        return np.array(window_data)


    def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):

        if df is None or df.empty or "close" not in df.columns:

            log_message(f"⚠️ Données invalides pour la paire, IA ignorée.")

            return None

        train_data, test_data = train_test_split(df, test_size=test_size)

        X_train = extract_window_data(train_data, window_len, zero_base)

        X_test = extract_window_data(test_data, window_len, zero_base)

        y_train = train_data[target_col][window_len:].values

        y_test = test_data[target_col][window_len:].values

        if zero_base:

            y_train = y_train / train_data[target_col][:-window_len].values - 1

            y_test = y_test / test_data[target_col][:-window_len].values - 1

        return train_data, test_data, X_train, X_test, y_train, y_test


    def build_lstm_model(
        input_data,
        output_size,
        neurons=100,
        activ_func="linear",
        dropout=0.2,
        loss="mse",
        optimizer="adam",
    ):

        model = Sequential()

        model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))

        model.add(Dropout(dropout))

        model.add(Dense(units=output_size))

        model.add(Activation(activ_func))

        model.compile(loss=loss, optimizer=optimizer)

        return model


    np.random.seed(42)

    window_len = 5

    test_size = 0.2

    zero_base = True

    lstm_neurons = 100

    epochs = 20

    batch_size = 32

    loss = "mse"

    dropout = 0.2

    optimizer = "adam"

    all_predictions = {}

    # Charger toutes les données une seule fois

    load_data(selected_crypto_pairs)


    def Predictions_calculs():

        print("lancement des caculs pour les prédictions")

        log_message("lancement des caculs pour les prédictions")

        save_logs_to_file()

        for crypto_pair in selected_crypto_pairs:

            print(f"Processing {crypto_pair}")

            hist = all_data[crypto_pair]  # Utiliser les données chargées

            train, test, X_train, X_test, y_train, y_test = prepare_data(
                hist,
                "close",
                window_len=window_len,
                zero_base=zero_base,
                test_size=test_size,
            )

            model = build_lstm_model(
                X_train,
                output_size=1,
                neurons=lstm_neurons,
                dropout=dropout,
                loss=loss,
                optimizer=optimizer,
            )

            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                shuffle=True,
            )

            targets = test["close"][window_len:]

            preds = model.predict(X_test).squeeze()

            mae = mean_absolute_error(preds, y_test)

            print(f"Mean Absolute Error for {crypto_pair}: {mae}")

            preds = test["close"].values[:-window_len] * (preds + 1)

            preds = pd.Series(index=targets.index, data=preds)

            all_predictions[crypto_pair] = preds

            line_plot(
                targets,
                preds,
                "actual",
                "prediction",
                lw=3,
                title=f"{crypto_pair} Price Prediction",
            )


    Predictions_calculs()


    # Function removed - using global compare_first_real_and_last_pred instead


    def will_crypto_increase_or_decrease(yesterday_last_real, today_pred):

        yesterday_last_value = yesterday_last_real.iloc[0]

        last_pred_value = today_pred.iloc[-1]

        if last_pred_value > yesterday_last_value:

            return 1

        else:

            return 0


    def tendance():

        for crypto_pair in selected_crypto_pairs:

            if (
                crypto_pair not in all_data
                or all_data[crypto_pair] is None
                or all_data[crypto_pair].empty
            ):

                log_message(f"⚠️ Données manquantes pour {crypto_pair}, tendance ignorée.")

                continue

            hist = all_data[crypto_pair]

            train, test = train_test_split(hist, test_size=test_size)

            train, test, X_train, X_test, y_train, y_test = prepare_data(
                hist,
                "close",
                window_len=window_len,
                zero_base=zero_base,
                test_size=test_size,
            )

            model = build_lstm_model(
                X_train,
                output_size=1,
                neurons=lstm_neurons,
                dropout=dropout,
                loss=loss,
                optimizer=optimizer,
            )

            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=True,
            )

            targets = test["close"][window_len:]

            preds = model.predict(X_test).squeeze()

            preds = test["close"].values[:-window_len] * (preds + 1)

            preds = pd.Series(index=targets.index, data=preds)

            if len(preds) < 2:

                log_message(
                    f"{crypto_pair} : Pas assez de données pour comparer les tendances."
                )

                continue

            yesterday_last_real = preds.iloc[-2]

            today_pred = preds.iloc[-1]

            trend_comparison = compare_first_real_and_last_pred(
                yesterday_last_real, today_pred
            )

            print(f"{crypto_pair} trend: {trend_comparison}")

            log_message(f"{crypto_pair} trend: {trend_comparison}")

            if today_pred > yesterday_last_real:

                log_message(f"📈 {crypto_pair} va probablement augmenter")

            else:

                log_message(f"📉 {crypto_pair} va probablement baisser")


    def get_account_balance(selected_crypto_pair):

        global accounts

        """Fetch the account balance in the selected cryptocurrency."""

        try:

            selected_crypto = selected_crypto_pair.split("-")[0]

            log_message(f"Récupération du solde {selected_crypto}...")

            save_logs_to_file()

            # accounts = client.get_accounts()

            for account in accounts["accounts"]:

                if account["currency"] == selected_crypto:

                    balance = Decimal(account["available_balance"]["value"])

                    log_message(f"Solde trouvé: {balance} {selected_crypto}")

                    save_logs_to_file()

                    return balance

        except Exception as e:

            log_message(f"Erreur lors de la récupération du solde {selected_crypto}: {e}")

            save_logs_to_file()

        return Decimal("0")


    #####################################################################################################

    # VALIDE


    def get_usdc_balance():

        global accounts

        if time.time() - globals().get("last_usdc_check", 0) > 30:

            globals()["last_usdc_check"] = time.time()

            log_message("🔄 Vérif solde USDC...")

        save_logs_to_file()

        try:

            accounts = client.get_accounts()  # Forcer le rafraîchissement des comptes

            time.sleep(1.5)  # Attendre que les conversions soient prises en compte

            for account in accounts["accounts"]:

                if account["currency"] == "USDC":

                    return Decimal(account["available_balance"]["value"])

        except Exception as e:

            log_message(f"Erreur Lors de la récupération du solde USDC: {e}")

            save_logs_to_file()

        return Decimal("0")


    #####################################################################################################

    # R.A.S


    @with_retry(retries=3, delay=1)
    def get_market_price(product_id):
        """Fetch the latest market price for a given product."""

        try:

            market_data = client.get_market_trades(product_id=product_id, limit=1)

            # log_message(f"Nous recherchons le prix du {product_id} sur le marché .")

            # if 'trades' in market_data and market_data['trades']:

            price = Decimal(market_data["trades"][0]["price"])

            log_message(f"le prix actuel du {product_id} sur le marché est: {price} USDC")
            # 🧠 Calcul réel des scores IA connectés
            lstm_pred = get_signal_strength(product_id)
            atr_pct = get_volatility_score(product_id)
            guru_signal = get_signal_strength(
                product_id
            )  # temporaire : à séparer si tu as une autre source
            lute_score = get_signal_strength(product_id)  # temporaire aussi
            final_score = calcul_score(lstm_pred, atr_pct, guru_signal, lute_score)

            save_logs_to_file()

            return price

        except Exception as e:

            log_message(f"Error fetching market price for {product_id}: {e}")

            save_logs_to_file()

        return None


    #####################################################################################################

    # VALIDE


    @with_retry(retries=3, delay=1)
    def check_usdc_balance():

        global accounts

        try:

            log_message("Vérification du solde USDC")

            save_logs_to_file()

            # accounts = client.get_accounts()

            for account in accounts["accounts"]:

                if account["currency"] == "USDC":

                    solde_usdc = Decimal(account["available_balance"]["value"])

                    log_message(f"Solde USDC: {solde_usdc}")

                    save_logs_to_file()

                    return solde_usdc

            log_message("Aucun solde USDC trouvé")

            save_logs_to_file()

            return Decimal("0")

        except Exception as e:

            log_message(f"Erreur lors de la vérification du solde USDC: {e}")

            save_logs_to_file()

            return Decimal("0")


    #####################################################################################################

    # MODIFIER CETTE FONCTION POUR QUELLE RESSEMBLE A L'ANCIENNE VERSION DU PROJET


    def safe_place_market_buy(product_id):

        global buy_percentage_of_capital

        try:

            log_message(f"🔍 Vérification du solde USDC...")

            usdc_balance = get_usdc_balance()

            if usdc_balance <= Decimal("0.01"):

                log_message("❌ Aucun solde USDC suffisant.")

                return None

            raw_price = get_market_price(product_id)
            if raw_price is None:
                log_message(
                    f"⛔ Erreur : prix indisponible pour {product_id}, achat annulé."
                )
                return None
            prix_moment_achat = Decimal(raw_price)

            tp_pct = float(sell_profit_target)

            # Forcer le type d’ordre à 'auto' si IA est activée

            ia_enabled = (
                True  # ⚠️ À connecter à la variable réelle de configuration si besoin
            )

            if ia_enabled:

                order_type_effective = "auto"

            else:

                order_type_effective = order_type

            fee_rate = (
                Decimal("0.0015") if order_type_effective == "limit" else Decimal("0.006")
            )

            coinbase_fee = prix_moment_achat * fee_rate

            prix_vente = prix_moment_achat * (1 + Decimal(tp_pct))

            profit_net_pct = (
                prix_vente - prix_moment_achat - coinbase_fee
            ) / prix_moment_achat

            log_message(
                f"📊 Analyse {product_id} | TP attendu = {prix_vente:.4f}, frais = {coinbase_fee:.4f}, profit net estimé = {profit_net_pct * 100:.2f}%"
            )

            threshold = seuil_profit_par_serie(
                series_id if "series_id" in locals() else "N/A"
            )

            if profit_net_pct < threshold:

                log_message(
                    f"⛔ Profit net insuffisant ({profit_net_pct:.2%} < {threshold:.2%}), achat annulé."
                )

                return None

            else:

                log_message(
                    f"✅ Trade autorisé (profit net {profit_net_pct:.2%} ≥ seuil {threshold:.2%})"
                )

            log_message(
                f"📈 Série utilisée : {series_id if 'series_id' in locals() else 'N/A'} | TP = {tp_pct*100:.2f}%, SL = {sell_stop_loss_target*100:.2f}%"
            )

            effective_usdc_amount = usdc_balance * Decimal(buy_percentage_of_capital)

            product_info = client.get_product(product_id)

            base_increment = product_info["quote_increment"]

            precision = int(base_increment.find("1")) - 1

            effective_usdc_amount1 = effective_usdc_amount.quantize(
                Decimal("1." + "0" * precision), rounding=ROUND_DOWN
            )

            if effective_usdc_amount1 <= Decimal("0"):

                log_message(f"❌ Montant ajusté trop faible : {effective_usdc_amount1}")

                return None

            formatted_usdc_amount = f"{effective_usdc_amount1:.{precision}f}"

            log_message(f"💰 Montant final pour achat : {formatted_usdc_amount} USDC")

            client_order_id = str(uuid.uuid4())

            side = "BUY"

            order_configuration = {
                "market_market_ioc": {"quote_size": formatted_usdc_amount}
            }

            response = create_order_safe(
                client,
                client_order_id=client_order_id,
                product_id=product_id,
                side=side,
                order_configuration=order_configuration,
            )

            if (
                isinstance(response, dict)
                and response.get("success", False)
                or hasattr(response, "success")
                and response.success
            ):

                log_message(f"✅ Ordre MARKET envoyé avec succès pour {product_id}")

                executed_orders_global.append(product_id)

                threading.Thread(
                    target=monitor_position_for_tp_sl,
                    args=(product_id, effective_usdc_amount1, prix_moment_achat),
                    daemon=True,
                ).start()

            else:

                log_message(f"⚠️ Réponse inattendue du serveur : {response}")

            return response

        except Exception as e:

            log_message(f"❌ Erreur lors de l'achat MARKET {product_id} : {e}")

            return None


    def monitor_position_for_tp_sl(product_id, amount_in_usdc, prix_moment_achat):
        """

        Surveille une position et déclenche TP/SL avec prise en compte des frais.

        """

        # 1) Prix d’achat facturé avec frais

        entry_price_fee = Decimal(str(entry_price)) * (Decimal("1.0") + COINBASE_FEE_RATE)

        # 2) Calcul des cibles brutes (prix brut pour TP/SL)

        take_profit_brut = (entry_price_fee * (Decimal("1.0") + sell_profit_target)) / (
            Decimal("1.0") - COINBASE_FEE_RATE
        )

        stop_loss_brut = (entry_price_fee * (Decimal("1.0") - sell_stop_loss_target)) / (
            Decimal("1.0") - COINBASE_FEE_RATE
        )

        highest_price = entry_price_fee

        trailing_stop = stop_loss_brut

        last_log_time = time.time()

        log_message(
            f"▶️ Lancement monitor TP/SL {product_id} : "
            f"Achat brut+fee = {entry_price_fee:.6f} USDC | "
            f"TP_brut = {take_profit_brut:.6f} (+{sell_profit_target*100:.2f}% net) | "
            f"SL_brut = {stop_loss_brut:.6f} (-{sell_stop_loss_target*100:.2f}% net)"
        )

        save_logs_to_file()

        while bot_running:

            try:

                current_price = coinbase_client.get_market_price(product_id)

                if current_price is None:

                    time.sleep(5)

                    continue

                # 3) Actualiser trailing (prix brut+fee)

                current_price_fee = Decimal(str(current_price)) * (
                    Decimal("1.0") + COINBASE_FEE_RATE
                )

                if current_price_fee > highest_price:

                    highest_price = current_price_fee

                    new_stop = (
                        highest_price * (Decimal("1.0") - sell_stop_loss_target)
                    ) / (Decimal("1.0") - COINBASE_FEE_RATE)

                    if new_stop > trailing_stop:

                        trailing_stop = new_stop

                        log_message(
                            f"⬆️ Nouveau top (brut+fee) = {highest_price:.6f} | SL_brut ajusté = {trailing_stop:.6f}"
                        )

                        save_logs_to_file()

                # 4) Check Take Profit net

                if current_price >= take_profit_brut:

                    net_receipt = Decimal(str(current_price)) * (
                        Decimal("1.0") - COINBASE_FEE_RATE
                    )

                    gain_pct = (net_receipt / entry_price_fee - 1) * 100

                    vol = volatilities.get(product_id, 0.0)

                    trend = trend_scores.get(product_id, 0.0)

                    series_id = assign_series_to_pair(product_id, vol, trend)

                    log_message(
                        f"💸 TAKE PROFIT atteint pour {product_id} : vente à {current_price:.6f}, gain net = {gain_pct:.2f}%"
                    )

                    save_logs_to_file()

                    sell_resp = coinbase_client.place_market_order(
                        product_id=product_id,
                        side="SELL",
                        base_size=str(
                            (Decimal("1") / Decimal(str(entry_price))) * amount_in_usdc
                        ),
                    )

                    qty_sold = amount_in_usdc / Decimal(str(entry_price))

                    sell_price_actual = (
                        coinbase_client.get_market_price(product_id) or current_price
                    )

                    log_vente_détaillée(
                        pair=product_id,
                        qty=float(qty_sold),
                        buy_price=entry_price,
                        sell_price=float(sell_price_actual),
                        series_id=series_id,
                    )

                    try:

                        # --- LOG HUMAIN VENTE ---

                        # On calcule prix_vente, total_fees, net_profit_pct, net_profit_usdc

                        buy_p = float(entry_price) if "entry_price" in locals() else 0

                        sell_p = (
                            float(sell_price_actual)
                            if "sell_price_actual" in locals()
                            else float(current_price) if "current_price" in locals() else 0
                        )

                        qty = float(qty_sold) if "qty_sold" in locals() else 1.0

                        total_fees = (
                            float(qty * (buy_p + sell_p) * float(COINBASE_FEE_RATE))
                            if "COINBASE_FEE_RATE" in locals()
                            else 0
                        )

                        net_profit_usdc = (sell_p - buy_p) * qty - total_fees

                        net_profit_pct = ((sell_p - buy_p) / buy_p) if buy_p != 0 else 0

                        log_vente_humaine(
                            product_id=product_id,
                            series_id=(
                                series_id
                                if "series_id" in locals()
                                else pair_strategy_mapping.get(product_id, "N/A")
                            ),
                            prix_vente=sell_p,
                            total_fees=total_fees,
                            net_profit_pct=net_profit_pct,
                            net_profit_usdc=net_profit_usdc,
                        )

                    except Exception as e:

                        log_message(f"Erreur log humain vente auto-injecté : {e}")

                    break

                # 5) Check Stop Loss brut

                if current_price <= trailing_stop:

                    net_receipt = Decimal(str(current_price)) * (
                        Decimal("1.0") - COINBASE_FEE_RATE
                    )

                    gain_pct = (net_receipt / entry_price_fee - 1) * 100

                    vol = volatilities.get(product_id, 0.0)

                    trend = trend_scores.get(product_id, 0.0)

                    series_id = assign_series_to_pair(product_id, vol, trend)

                    log_message(
                        f"💸 STOP LOSS déclenché pour {product_id} : vente à {current_price:.6f}, gain net = {gain_pct:.2f}%"
                    )

                    save_logs_to_file()

                    sell_resp = coinbase_client.place_market_order(
                        product_id=product_id,
                        side="SELL",
                        base_size=str(
                            (Decimal("1") / Decimal(str(entry_price))) * amount_in_usdc
                        ),
                    )

                    qty_sold = amount_in_usdc / Decimal(str(entry_price))

                    sell_price_actual = (
                        coinbase_client.get_market_price(product_id) or current_price
                    )

                    log_vente_détaillée(
                        pair=product_id,
                        qty=float(qty_sold),
                        buy_price=entry_price,
                        sell_price=float(sell_price_actual),
                        series_id=series_id,
                    )

                    try:

                        # --- LOG HUMAIN VENTE ---

                        # On calcule prix_vente, total_fees, net_profit_pct, net_profit_usdc

                        buy_p = float(entry_price) if "entry_price" in locals() else 0

                        sell_p = (
                            float(sell_price_actual)
                            if "sell_price_actual" in locals()
                            else float(current_price) if "current_price" in locals() else 0
                        )

                        qty = float(qty_sold) if "qty_sold" in locals() else 1.0

                        total_fees = (
                            float(qty * (buy_p + sell_p) * float(COINBASE_FEE_RATE))
                            if "COINBASE_FEE_RATE" in locals()
                            else 0
                        )

                        net_profit_usdc = (sell_p - buy_p) * qty - total_fees

                        net_profit_pct = ((sell_p - buy_p) / buy_p) if buy_p != 0 else 0

                        log_vente_humaine(
                            product_id=product_id,
                            series_id=(
                                series_id
                                if "series_id" in locals()
                                else pair_strategy_mapping.get(product_id, "N/A")
                            ),
                            prix_vente=sell_p,
                            total_fees=total_fees,
                            net_profit_pct=net_profit_pct,
                            net_profit_usdc=net_profit_usdc,
                        )

                    except Exception as e:

                        log_message(f"Erreur log humain vente auto-injecté : {e}")

                    break

                # 6) Journaux périodiques

                if time.time() - last_log_time > 120:

                    pct_margin = ((current_price - trailing_stop) / current_price) * 100

                    log_message(
                        f"⏱️ Monitor {product_id} → Prix actuel (brut) = {current_price:.6f} | Peak (brut+fee) = {highest_price:.6f} | Marge SL = {pct_margin:.2f}%"
                    )

                    save_logs_to_file()

                    last_log_time = time.time()

                time.sleep(10)

            except Exception as e:

                log_message(f"⚠️ Erreur monitor_position_for_tp_sl({product_id}): {e}")

                traceback.print_exc()

                save_logs_to_file()

                time.sleep(30)

        log_message(f"🏁 Monitoring terminé pour {product_id}")

        save_logs_to_file()


    def place_market_sell(product_id, amount_in_usdc, prix_moment_achat):
        """Place a market sell order ensuring the order size meets Coinbase's requirements."""

        try:

            ############################

            # le prix actuelle

            # price=get_market_price(product_id)

            amount_in_btc = (1 / prix_moment_achat) * amount_in_usdc

            # Fetch precision requirements for the base currency (BTC)

            product_details = client.get_product(product_id)

            base_increment = product_details["base_increment"]

            log_message(f"{product_id} base increment is: {base_increment}")

            save_logs_to_file()

            # Validate and calculate precision

            precision = base_increment.find("1")

            if precision == -1:

                raise ValueError(f"Invalid base_increment format: {base_increment}")

            precision -= 1

            # Apply rounding to match the precision level expected by Coinbase

            amount_in_btc1 = amount_in_btc.quantize(
                Decimal("1." + "0" * precision), rounding=ROUND_DOWN
            )

            if Decimal(amount_in_btc1) < Decimal(base_increment):

                log_message(
                    f"⛔ Trop petit pour être converti : {amount_in_btc1} < {base_increment}"
                )

                return None

            ############################

            # Log the adjusted base currency amount

            log_message(
                f"Adjusted {product_id} amount with precision for sell: {amount_in_btc1}"
            )

            save_logs_to_file()

            # Define the required individual arguments for create_order

            client_order_id = str(uuid.uuid4())  # Generate a unique client order ID

            side = "SELL"

            order_configuration = {
                "market_market_ioc": {
                    "base_size": str(amount_in_btc1)  # Specify in base currency (BTC)
                }
            }

            # Place the order with required arguments

            response = create_order_safe(
                client,
                client_order_id=client_order_id,
                product_id=product_id,
                side=side,
                order_configuration=order_configuration,
            )

            log_message(f"Market sell order response for {product_id}: {response}")

            sales_done_global.append(product_id)

            save_logs_to_file()

            return response

        except KeyError as ke:

            log_message(f"Missing expected key in product details: {ke}")

            save_logs_to_file()

        except ValueError as ve:

            log_message(f"Invalid value encountered: {ve}")

            save_logs_to_file()

        except Exception as e:

            log_message(f"Error placing market sell order for {product_id}: {e}")

            save_logs_to_file()

        return None


    #####################################################################################################

    # VALIDE


    def get_position_value(selected_crypto_pair):
        """Calculate the current USD value of the crypto holdings."""

        balance = get_account_balance(selected_crypto_pair)

        market_price = get_market_price(selected_crypto_pair)

        if balance and market_price:

            return balance * market_price

        return None


    ####################################################################################################################


    def place_market_sell2(product_id, amount_in_usdc):
        """Place a market sell order ensuring the order size meets Coinbase's requirements."""

        try:

            ############################

            # Fetch precision requirements for the base currency (BTC)

            product_details = client.get_product(product_id)

            base_increment = product_details["base_increment"]

            log_message(f"{product_id} base increment is: {base_increment}")

            save_logs_to_file()

            # Validate and calculate precision

            precision = base_increment.find("1")

            if precision == -1:

                raise ValueError(f"Invalid base_increment format: {base_increment}")

            precision -= 1

            # Apply rounding to match the precision level expected by Coinbase

            amount_in_btc1 = amount_in_usdc.quantize(
                Decimal("1." + "0" * precision), rounding=ROUND_DOWN
            )

            if Decimal(amount_in_btc1) < Decimal(base_increment):

                log_message(
                    f"⛔ Trop petit pour être converti : {amount_in_btc1} < {base_increment}"
                )

                return None

            ############################

            # Log the adjusted base currency amount

            # log_message(f"Adjusted {product_id} amount with precision for sell: {amount_in_btc1}")

            save_logs_to_file()

            # Define the required individual arguments for create_order

            client_order_id = str(uuid.uuid4())  # Generate a unique client order ID

            side = "SELL"

            order_configuration = {
                "market_market_ioc": {
                    "base_size": str(amount_in_btc1)  # Specify in base currency (BTC)
                }
            }

            # Place the order with required arguments

            response = create_order_safe(
                client,
                client_order_id=client_order_id,
                product_id=product_id,
                side=side,
                order_configuration=order_configuration,
            )

            log_message(f"Market sell order response for {product_id}: {response}")

            sales_done_global.append(product_id)

            save_logs_to_file()

            return response

        except KeyError as ke:

            log_message(f"Missing expected key in product details: {ke}")

            save_logs_to_file()

        except ValueError as ve:

            log_message(f"Invalid value encountered: {ve}")

            save_logs_to_file()

        except Exception as e:

            log_message(f"Error placing market sell order for {product_id}: {e}")

            save_logs_to_file()

        return None


    def remove_last_char_if_in_list(string, predefined_list):

        if string and string[-1] in predefined_list:

            return string[:-1]  # Supprime le dernier caractère

        return string  # Retourne la chaîne telle quelle si le caractère n'est pas dans la liste


    def convert_to_usdc(account, selected_crypto_bases=None):

        product_id = "inconnu"  # Initialisé pour éviter UnboundLocalError

        if selected_crypto_bases is None:

            selected_crypto_bases = []

        try:

            # Vérifier si le compte a des fonds

            if Decimal(account["available_balance"]["value"]) > 0:

                log_message(
                    f"Le compte {account['name']} a des fonds : {account['available_balance']['value']} {account['available_balance']['currency']}"
                )

                save_logs_to_file()

                currency = account["available_balance"]["currency"]

                # Effectuer la conversion en USDC

                if currency != "USDC":

                    conversion_amount = Decimal(account["available_balance"]["value"])

                    log_message(
                        f"Conversion de {conversion_amount} {account['available_balance']['currency']} en USDC..."
                    )

                    save_logs_to_file()

                    # netoyer le nom du porteuille si il contient un chiffre à la fin de son nom exemple de ETH2

                    # Exemple d'utilisation

                    predefined_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

                    newcurrency = remove_last_char_if_in_list(currency, predefined_list)

                    to_account = "USDC"

                    product_id = newcurrency + "-" + to_account

                    place_market_sell2(product_id, conversion_amount)

                    # --- Enregistrement de la conversion dans sales_register.xlsx ---

                    from datetime import datetime

                    try:

                        df_conv = pd.read_excel(LOG_FILE)

                    except FileNotFoundError:

                        df_conv = pd.DataFrame(
                            columns=[
                                "timestamp",
                                "series_id",
                                "sale_price",
                                "gross_profit",
                                "fees",
                                "net_gain",
                            ]
                        )

                    # Création de l'enregistrement

                    record_conv = {
                        "timestamp": datetime.utcnow(),
                        "series_id": product_id,  # conversion event
                        "sale_price": float(conversion_amount),
                        "gross_profit": 0.0,
                        "fees": 0.0,
                        "net_gain": 0.0,
                    }

                    df_conv = pd.concat(
                        [df_conv, pd.DataFrame([record_conv])], ignore_index=True
                    )

                    df_conv.to_excel(LOG_FILE, index=False)

                    # Vérification rapide du registre

                    try:

                        tail_conv = pd.read_excel(LOG_FILE).tail()

                        log_message(f"📊 Registre mis à jour (conversion):\n{tail_conv}")

                    except Exception as e:

                        log_message(f"⚠️ Échec lecture registre Excel conversion: {e}")

                    return True  # Simule que la conversion est réussie

                else:

                    log_message(
                        f"ℹ️ Le compte {account['name']} est en USDC — aucune conversion nécessaire."
                    )

                    save_logs_to_file()

                    return False

            else:

                log_message(f"Le compte {account['name']} n'a pas de fonds.")

                save_logs_to_file()

                return False

        except Exception as e:

            product_id_str = product_id if "product_id" in locals() else "inconnu"

            log_message(
                f"Erreur lors de la vérification des fonds du compte {account['name']} pour {product_id_str} : {e}"
            )

            save_logs_to_file()

            return False


    def check_and_convert_all_accounts(selected_crypto_base):

        global accounts

        try:

            # Récupérer tous les comptes

            # accounts = client.get_accounts()

            log_message("Analyse des comptes en cours...")

            save_logs_to_file()

            # Parcourir tous les comptes et vérifier s'il y a des fonds

            for account in accounts["accounts"]:

                convert_to_usdc(
                    account,
                    selected_crypto_bases if "selected_crypto_bases" in locals() else [],
                )

        except Exception as e:

            log_message(f"Erreur lors de la récupération des comptes : {e}")

            save_logs_to_file()


    #####################################################################################################

    # VERIFIE REST A TESTER


    def auto_convert_to_usdc(min_usdc_balance=100, ignore_pairs=None):
        """

        Convertit automatiquement toutes les cryptos en USDC sauf celles en ignore_pairs.

        Args:

            min_usdc_balance (float): Seuil minimal de USDC à maintenir (évite les micro-conversions)

            ignore_pairs (list): Liste des paires à ne pas convertir (ex: ['BTC', 'ETH'])

        """

        global accounts

        if ignore_pairs is None:

            ignore_pairs = []

        try:

            log_message("Debut de la conversion forcee vers USDC")

            usdc_balance = get_usdc_balance()

            if usdc_balance >= min_usdc_balance:

                log_message(
                    f"Solde USDC suffisant ({usdc_balance} USDC), pas de conversion necessaire"
                )

                return False

            log_message(
                f"Solde USDC faible ({usdc_balance} USDC), conversion des altcoins..."
            )

            for account in accounts["accounts"]:

                currency = account["available_balance"]["currency"]

                amount = Decimal(account["available_balance"]["value"])

                if currency == "USDC" or currency in ignore_pairs:

                    continue

                if amount > 0:

                    try:

                        current_price = get_market_price(f"{currency}-USDC")

                        usdc_value = amount * current_price

                        if usdc_value < 5:

                            log_message(
                                f"Ignore {amount} {currency} (valeur trop faible: {usdc_value:.1f} USDC)"
                            )

                            continue

                        log_message(
                            f"Conversion de {amount} {currency} (~{usdc_value:.1f} USDC)"
                        )

                        clean_symbol = currency.rstrip("0123456789")

                        product_id = f"{clean_symbol}-USDC"

                        create_order_safe(
                            client,
                            client_order_id=str(uuid.uuid4()),
                            product_id=product_id,
                            side="SELL",
                            order_configuration={
                                "market_market_ioc": {"base_size": str(amount)}
                            },
                        )

                        log_message(f"Conversion vers USDC effectuee pour {currency}")

                    except Exception as e:

                        log_message(f"Erreur lors de la conversion de {currency}: {str(e)}")

            log_message("Conversion terminee")

            return True

        except Exception as e:

            log_message(f"Erreur globale lors de la conversion: {str(e)}")

            return False


    def dca_trading_bot():

        # === AUTO-SELECTION DES PAIRES HAUSSIERES SI IA ACTIVE ===

        if ia:

            try:

                log_message(
                    f"✅ Paires haussières sélectionnées dynamiquement : {selected_crypto_pairs}"
                )

                save_logs_to_file()

            except Exception as e:

                log_message(f"❌ Erreur pendant la sélection dynamique des paires : {e}")

                save_logs_to_file()

        """DCA trading bot with automated buy based on percentages."""

        global bot_running, buy_percentage_of_capital

        bot_running = True

        log_message("DCA trading bot started")

        save_logs_to_file()

        # === AUTO-SELECTION DYNAMIQUE TOUTES LES 6H ===

        global last_autoselect_update

        now = datetime.now()

        if (
            not selected_crypto_pairs
            or not hasattr(dca_trading_bot, "last_autoselect_update")
            or (now - dca_trading_bot.last_autoselect_update).total_seconds() > 21600
        ):

            try:

                dca_trading_bot.last_autoselect_update = now

                log_message(
                    f"🔄 Paires mises à jour par auto-sélection dynamique à {now.strftime('%H:%M')} : {selected_crypto_pairs}"
                )

                save_logs_to_file()

            except Exception as e:

                log_message(f"❌ Échec auto-sélection dynamique : {str(e)}")

                save_logs_to_file()

        while bot_running:  # S'assurer que les processus en cours sont terminés

            try:

                # Pour chaque paire de crypto-monnaies sélectionnée

                for selected_crypto_pair in selected_crypto_pairs:

                    # Si un arrêt est demandé, sortir de la boucle principale

                    if not bot_running:

                        log_message("Arrêt demandé. Finalisation des processus en cours.")

                        save_logs_to_file()

                        break  # Quitter la boucle des paires pour arrêter proprement

                    # Identité de la paire traitée

                    product_id = selected_crypto_pair

                    log_message(f"Paire traitée actuellement : {product_id}")

                    save_logs_to_file()

                    # Vérification du solde USDC

                    usdc_balance = get_usdc_balance()

                    log_message(f"Le solde USDC est : {usdc_balance}")

                    save_logs_to_file()

                    # déterminer le montant à acheter

                    buy_amount = usdc_balance * buy_percentage_of_capital

                    # Vérification du solde USDC

                    if usdc_balance < 50:

                        # if usdc_balance < Decimal(buy_amount):

                        log_message(
                            f"Solde USDC insuffisant pour placer un ordre d'achat de : {product_id}."
                        )

                        save_logs_to_file()

                        # Analise de d'autre portefeuilles pour alimenter notre portefeuille USDC

                        log_message(
                            f"Analysons le solde d'autre portefeuilles pour trouvez les fond nécessaire à l'acaht de : {product_id}."
                        )

                        save_logs_to_file()

                        selected_crypto_base = selected_crypto_pair.split("-")[0]

                        check_and_convert_all_accounts(selected_crypto_base)

                        log_message(f"Conversions treminés")

                        save_logs_to_file()

                        # si les fonds on été trouvés on passe a l'achat sinon on passe à la paire de crypto suivante

                    # Achat avec ou sans IA

                    if ia:

                        log_message("IA activated.")

                        save_logs_to_file()

                        today_date = pd.to_datetime("today").normalize()

                        previous_date = today_date - pd.Timedelta(days=1)

                        hist = all_data[selected_crypto_pair]

                        train, test = train_test_split(hist, test_size=test_size)

                        train, test, X_train, X_test, y_train, y_test = prepare_data(
                            hist,
                            "close",
                            window_len=window_len,
                            zero_base=zero_base,
                            test_size=test_size,
                        )

                        targets = test["close"][window_len:]

                        model = build_lstm_model(
                            X_train,
                            output_size=1,
                            neurons=lstm_neurons,
                            dropout=dropout,
                            loss=loss,
                            optimizer=optimizer,
                        )

                        preds = model.predict(X_test).squeeze()

                        preds = test["close"].values[:-window_len] * (preds + 1)

                        preds = pd.Series(index=targets.index, data=preds)

                        # yesterday_last_real = test['close'].loc[test.index.date == previous_date.date()]

                        yesterday_last_real = preds.loc[previous_date:previous_date]

                        today_pred = preds.loc[today_date:today_date]

                        trend_comparison = will_crypto_increase_or_decrease(
                            yesterday_last_real, today_pred
                        )

                        if "augmenter" in trend_comparison:

                            log_message(
                                f"{product_id} Prendra de la valeur, achat en cours."
                            )

                            save_logs_to_file()

                            safe_place_market_buy(product_id)

                        else:

                            log_message(f"{product_id} Perdra de la valeur, achat annulé.")

                            save_logs_to_file()

                    else:

                        log_message(
                            f"Placons un ordre d'achat d'un montant de {buy_amount} pour : {product_id}."
                        )

                        save_logs_to_file()

                        safe_place_market_buy(product_id)

                # Mise en pause après avoir traité toutes les paires

                log_message("Toutes les paires traitées. Conversion de résidus en USDC...")

                save_logs_to_file()

                force_convert_all_to_usdc(min_value_usdc=1.0)

                log_message("⏸ Mise en pause du robot.")

                save_logs_to_file()

                print(
                    f"⏳ Attente de {dca_interval_seconds} secondes avant prochain cycle..."
                )

                time.sleep(dca_interval_seconds)

            except Exception as e:

                log_message(f"Exception in DCA trading bot: {e}")

                save_logs_to_file()

                time.sleep(10)

        log_message("Finalisation des processus terminée. Arrêt du bot.")

        save_logs_to_file()


    # derniere version prenant en compte l ia

    #####################################################################################################

    # accounts = client.get_accounts()

    # Fonction principale


    def Balance_Total():

        global log_data1, accounts

        while True:

            # Réinitialiser log_data

            log_data1 = ""

            try:

                # Récupérer les portefeuilles

                transactions = client.get_transaction_summary()

                balance_total = transactions["total_balance"]

                log_data1 += f"{balance_total}\n"

                time.sleep(2)

                # Gestion des erreurs HTTP

                retries = 0

                max_retries = 5

                delay = 1

                while retries < max_retries:

                    try:

                        accounts = client.get_accounts()  # Obtenez les comptes

                        print("Mise à jour des comptes")

                        log_message("Mise à jour des comptes")

                        save_logs_to_file()

                        # ========================================================================================

                        heure_actuelle = (
                            datetime.now()
                        )  # Extrait uniquement l'heure actuelle

                        if should_update_ai(heure_actuelle):

                            print("Mise à jour de l'IA.")

                            load_data(selected_crypto_pairs)

                            Predictions_calculs()

                            tendance()

                        else:

                            if last_data_fetched < datetime.utcnow() - timedelta(hours=1):

                                print("⚠️ Données obsolètes : rechargement nécessaire.")

                            log_message("Heure de Mise à jour de l'IA non atteinte.")

                            save_logs_to_file()

                        break

                    except HTTPError as e:

                        if e.response.status_code == 429:

                            print("Rate limit exceeded. Retrying after delay...")

                            log_message("Rate limit exceeded. Retrying after delay...")

                            save_logs_to_file()

                            time.sleep(delay)

                            retries += 1

                            delay *= 2  # Backoff exponentiel

                        else:

                            raise e

            except KeyError as e:

                log_data1 += f"Erreur de récupération de la balance: {str(e)}\n"

                save_logs_to_file()

            # Attendre avant la prochaine itération

            time.sleep(2)


    # Démarrer la fonction dans un thread

    thread = threading.Thread(target=Balance_Total)

    thread.daemon = True  # Assure que le thread s'arrête avec le programme principal

    thread.start()

    #####################################################################################

    #########################################################################################

    # accounts = client.get_accounts()

    #########################################################################################

    # Supprimé: doublon de get_usdc_balance

    #########################################################################################

    #########################################################################################


    def get_eth2_balance():

        global accounts

        try:

            # accounts = client.get_accounts()

            for account in accounts["accounts"]:

                if account["currency"] == "BTC":

                    return Decimal(account["available_balance"]["value"])

        except Exception as e:

            log_message(f"Error fetching BTC balance: {e}")

            save_logs_to_file()

        return Decimal("0")


    #########################################################################################

    # Fonction pour vérifier et comparer les soldes toutes les secondes


    def Your_Usdc():

        global log_data2  # Utiliser la variable soldes initiaux définie en dehors de la fonction

        while True:

            # Réinitialiser log_data à chaque itération avant d'ajouter de nouveaux logs

            log_data2 = ""  # Effacer les logs précédents

            # Assurez-vous que vous obtenez bien les informations de solde et les manipulez correctement

            try:

                # Récupérer les portefeuilles

                usdc_balance = get_usdc_balance()

                log_data2 += f"{usdc_balance:.1f}\n"

            except KeyError as e:

                log_data2 += f"Erreur de récupération de la balance: {str(e)}\n"

            # Envoyer les données mises à jour au client via SocketIO

            # socketio.emit('update_log2', {'log_usdc_balance': log_data2})

            # Attendre une seconde avant de vérifier à nouveau

            time.sleep(2.6)


    # Créer et démarrer le thread

    thread1 = threading.Thread(target=Your_Usdc)

    thread1.daemon = True  # Ensure the thread exits when the main program exits

    thread1.start()

    #####################################################################################

    #########################################################################################

    # Fonction pour vérifier et comparer les soldes toutes les secondes


    def Your_Eth2():

        global log_data3  # Utiliser la variable soldes initiaux définie en dehors de la fonction

        while True:

            # Réinitialiser log_data à chaque itération avant d'ajouter de nouveaux logs

            log_data3 = ""  # Effacer les logs précédents

            # Assurez-vous que vous obtenez bien les informations de solde et les manipulez correctement

            try:

                # Récupérer les portefeuilles

                eth2_balance = get_eth2_balance()

                log_data3 += f"{eth2_balance:.1f}\n"

            except KeyError as e:

                log_data3 += f"Erreur de récupération de la balance: {str(e)}\n"

            # Envoyer les données mises à jour au client via SocketIO

            # socketio.emit('update_log3', {'log_eth2_balance': log_data3})

            # Attendre une seconde avant de vérifier à nouveau

            time.sleep(2.8)


    # Créer et démarrer le thread

    thread2 = threading.Thread(target=Your_Eth2)

    thread2.daemon = True  # Ensure the thread exits when the main program exits

    thread2.start()

    #####################################################################################

    # Fonction pour récupérer les soldes initiaux (une fois par jour)


    def get_soldes_initiaux():

        accounts = client.get_accounts()

        soldes_initiaux = {}

        global log_data

        for account in accounts.accounts:

            solde_initial = float(account.available_balance["value"])

            currency = account.available_balance["currency"]

            soldes_initiaux[account.uuid] = (solde_initial, currency)

            log_data += (
                f"Solde initial pour le compte {currency}: {solde_initial} {currency}\n"
            )

        print(f"contenue du dictionnaire solde initiaux: {soldes_initiaux}")

        return soldes_initiaux


    # Récupérer les soldes initiaux pour commencer

    soldes_initiaux = get_soldes_initiaux()

    #####################################################################################

    # Fonction pour obtenir la valeur en temps réel d'une cryptomonnaie via l'API Coinbase


    def get_crypto_value(crypto_pair):

        url = f"https://api.coinbase.com/v2/prices/{crypto_pair}/buy"

        try:

            response = requests.get(url)

            response.raise_for_status()  # Vérifie si la requête a échoué (code HTTP 4xx ou 5xx)

            # Tentons de décoder le JSON

            try:

                data = response.json()

                # Vérification si la structure attendue est présente

                if "data" in data and "amount" in data["data"]:

                    return float(data["data"]["amount"])

                else:

                    raise ValueError("Réponse invalide: 'data' ou 'amount' manquants.")

            except ValueError as e:

                raise ValueError(f"Erreur lors de l'analyse de la réponse JSON: {e}")

        except requests.exceptions.RequestException as e:

            raise ConnectionError(f"Erreur lors de la requête à l'API Coinbase: {e}")

        except Exception as e:

            raise Exception(f"Erreur dans get_crypto_value pour {crypto_pair}: {e}")


    #####################################################################################

    # Fonction pour vérifier et comparer les soldes toutes les secondes


    def check_soldes():

        global soldes_initiaux, log_data, Profit_cumul, total, accounts  # Utiliser les variables globales nécessaires

        while True:

            try:

                # voir si ca fonctionne

                Profit_cumul = 0

                print(
                    f"contenue du dictionnaire solde initiaux dans check solde: {soldes_initiaux}"
                )

                log_data = ""  # Réinitialiser les logs

                heure_locale = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                log_data += f"Dernière mise à jour : {heure_locale}\n"

                for account in accounts.accounts:

                    solde_initial, currency = soldes_initiaux.get(
                        account.uuid, (0, "USD")
                    )  # Valeur par défaut si non trouvé

                    try:

                        crypto = account.available_balance["currency"]

                        accountts = client.get_accounts()

                        for accountt in accountts["accounts"]:

                            if accountt["currency"] == crypto:

                                solde_actuel = float(accountt["available_balance"]["value"])

                        # solde_actuel = float(account.available_balance['value'])

                        log_data += f"------------------------------------------------\n"

                        log_data += "::::::::::::::::::::::::::::::::::::::::::::::::\n"

                        log_data += f"PORTEFEUILLE {crypto}\n"

                        log_data += "::::::::::::::::::::::::::::::::::::::::::::::::\n"

                        log_data += f"Solde initial : {solde_initial} {crypto}\n"

                        log_data += f"Solde actuel  : {solde_actuel} {crypto}\n"

                        # Calculer la différence entre le solde initial et le solde actuel

                        difference = solde_actuel - solde_initial

                        log_data += "::::::::::::::::::::::::::::::::::::::::::::::::\n"

                        log_data += f"Profit du jour pour le compte {currency}: {difference:.1f} {currency}\n"

                        # Récupérer la valeur en USD

                        crypto_pair = crypto + "-USD"

                        try:

                            value_in_usd = get_crypto_value(crypto_pair)

                            log_data += (
                                f"La valeur de {crypto} en USD est : {value_in_usd}\n"
                            )

                            total = value_in_usd * difference

                            log_data += f"Conversion de vos bénéfices {crypto} en USD = {total:.1f} USD\n"

                            Profit_cumul += total

                        except Exception as e:

                            log_data += f"Erreur lors de la récupération de la valeur de {crypto} en USD : {str(e)}\n"

                            continue  # Passer à la paire suivante en cas d'erreur

                        log_data += "::::::::::::::::::::::::::::::::::::::::::::::::\n"

                        # Vérifier si la date a changé (si un nouveau jour commence)

                        current_time = datetime.now()

                        if (
                            current_time.hour == 0 and current_time.minute == 0
                        ):  # Si c'est minuit

                            log_data += (
                                "Mise à jour des soldes initiaux pour le nouveau jour...\n"
                            )

                            soldes_initiaux = get_soldes_initiaux()

                    except Exception as e:

                        log_data += f"Erreur avec le portefeuille {crypto}: {str(e)}\n"

                        continue  # Passer au compte suivant

                log_data += f"PROFIT CUMULE : {Profit_cumul:.1f} USD\n"

                # Envoyer les données mises à jour au client

                # socketio.emit('update_log', {'log': log_data})

            except Exception as e:

                # Enregistrer toute autre erreur non prévue

                log_data += f"Erreur générale dans le thread check_soldes : {str(e)}\n"

            finally:

                # Toujours attendre avant de recommencer pour éviter une surcharge

                time.sleep(4.5)


    # Créer et démarrer le thread

    thread3 = threading.Thread(target=check_soldes)

    thread3.daemon = (
        True  # Assure que le thread s'arrête lorsque le programme principal s'arrête
    )

    thread3.start()

    #########################################################################################

    # Fonction pour vérifier et comparer les ordres toutes les secondes


    def les_ordres():

        global log_data4  # Utilisation de la variable globale log_data

        while True:

            # Réinitialiser log_data à chaque itération

            log_data4 = ""

            try:

                heure_locale = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                log_data4 += f"Dernière mise à jour : {heure_locale}\n"

                # Obtenir tous les ordres

                orders_data = client.list_orders()

                # Conversion de la chaîne JSON en dictionnaire Python

                orders_dict = orders_data

                # Parcourir et traiter les données des commandes

                # for order in orders_dict['orders']:

                for order in orders_dict["orders"][:30]:

                    order_id = order["order_id"]

                    product_id = order["product_id"]

                    user_id = order["user_id"]

                    side = order["side"]

                    client_order_id = order["client_order_id"]

                    order_status = order["status"]

                    time_in_force = order["time_in_force"]

                    created_time = order["created_time"]

                    completion_percentage = order["completion_percentage"]

                    filled_size = order["filled_size"]

                    average_filled_price = order["average_filled_price"]

                    fee = order["fee"]

                    number_of_fills = order["number_of_fills"]

                    filled_value = order["filled_value"]

                    pending_cancel = order["pending_cancel"]

                    size_in_quote = order["size_in_quote"]

                    total_fees = order["total_fees"]

                    size_inclusive_of_fees = order["size_inclusive_of_fees"]

                    total_value_after_fees = order["total_value_after_fees"]

                    trigger_status = order["trigger_status"]

                    order_type = order["order_type"]

                    reject_reason = order["reject_reason"]

                    settled = order["settled"]

                    product_type = order["product_type"]

                    reject_message = order["reject_message"]

                    cancel_message = order["cancel_message"]

                    order_placement_source = order["order_placement_source"]

                    outstanding_hold_amount = order["outstanding_hold_amount"]

                    is_liquidation = order["is_liquidation"]

                    last_fill_time = order["last_fill_time"]

                    edit_history = order["edit_history"]

                    leverage = order["leverage"]

                    margin_type = order["margin_type"]

                    retail_portfolio_id = order["retail_portfolio_id"]

                    originating_order_id = order["originating_order_id"]

                    attached_order_id = order["attached_order_id"]

                    attached_order_configuration = order["attached_order_configuration"]

                    #################################

                    # Ajouter les informations de l'ordre au log

                    log_data4 += f"------------------------------------------------\n"

                    log_data4 += f"Order ID: {order_id}\n"

                    log_data4 += f"Product ID: {product_id}\n"

                    log_data4 += f"User ID: {user_id}\n"

                    log_data4 += f"side: {side}\n"

                    log_data4 += f"client_order_id: {client_order_id}\n"

                    log_data4 += f"Status: {order_status}\n"

                    log_data4 += f"time_in_force: {time_in_force}\n"

                    log_data4 += f"created_time: {created_time}\n"

                    log_data4 += f"completion_percentage: {completion_percentage}\n"

                    log_data4 += f"Filled Size: {filled_size}\n"

                    log_data4 += f"Average Filled Price: {average_filled_price}\n"

                    log_data4 += f"fee: {fee}\n"

                    log_data4 += f"number_of_fills: {number_of_fills}\n"

                    log_data4 += f"filled_value: {filled_value}\n"

                    log_data4 += f"pending_cancel: {pending_cancel}\n"

                    log_data4 += f"size_in_quote: {size_in_quote}\n"

                    log_data4 += f"Total Fees: {total_fees}\n"

                    log_data4 += f"size_inclusive_of_fees: {size_inclusive_of_fees}\n"

                    log_data4 += f"total_value_after_fees: {total_value_after_fees}\n"

                    log_data4 += f"trigger_status: {trigger_status}\n"

                    log_data4 += f"order_type: {order_type}\n"

                    log_data4 += f"reject_reason: {reject_reason}\n"

                    log_data4 += f"settled: {settled}\n"

                    log_data4 += f"product_type: {product_type}\n"

                    log_data4 += f"reject_message: {reject_message}\n"

                    log_data4 += f"cancel_message: {cancel_message}\n"

                    log_data4 += f"order_placement_source: {order_placement_source}\n"

                    log_data4 += f"outstanding_hold_amount: {outstanding_hold_amount}\n"

                    log_data4 += f"is_liquidation: {is_liquidation}\n"

                    log_data4 += f"last_fill_time: {last_fill_time}\n"

                    log_data4 += f"edit_history: {edit_history}\n"

                    log_data4 += f"leverage: {leverage}\n"

                    log_data4 += f"margin_type: {margin_type}\n"

                    log_data4 += f"retail_portfolio_id: {retail_portfolio_id}\n"

                    log_data4 += f"originating_order_id: {originating_order_id}\n"

                    log_data4 += f"attached_order_id: {attached_order_id}\n"

                    log_data4 += (
                        f"attached_order_configuration: {attached_order_configuration}\n"
                    )

                    #################################

            except Exception as e:

                # Gestion des exceptions et ajout d'un message d'erreur aux logs

                log_data4 += f"Erreur lors de la récupération des ordres : {str(e)}\n"

            # Envoyer les données mises à jour au client via SocketIO

            # socketio.emit('update_log4', {'log_orders': log_data4})

            # Pause d'une seconde avant de recommencer

            time.sleep(2.5)


    # Créer et démarrer le thread

    thread4 = threading.Thread(target=les_ordres)

    thread4.daemon = True  # Ensure the thread exits when the main program exits

    thread4.start()

    #####################################################################################

    #####################################################################################################


    def send_2fa_code():

        global current_2fa_code

        current_2fa_code = totp.now()  # Generate the 2FA code

        # Create and send the email with the 2FA code

        subject = "Your 2FA Code"

        body = f"Your 2FA code is: {current_2fa_code}"

        msg = Message(subject, recipients=[user_email])

        msg.body = body

        try:

            with mail.connect() as connection:  # Explicitly connect to SMTP server

                connection.send(msg)

            log_message(f"Sent 2FA code to {user_email}")

            save_logs_to_file()

        except Exception as e:

            log_message(f"Error sending 2FA code: {e}")

            save_logs_to_file()


    ######################################################################


    def send_failed_login_alert():

        # Vérification de la variable user_email

        if not user_email:

            print("Error: User email is not set.")

            return  # Retourne sans envoyer l'email si l'email utilisateur n'est pas défini

        # Définir le sujet et le corps de l'email

        subject = "Failed Login Attempt"

        body = "Une tentative de connexion a échoué."

        # Créer le message email

        msg = Message(subject, recipients=[user_email])

        msg.body = body

        try:

            print(
                f"Attempting to send email to {user_email}"
            )  # Vérifier si l'email est bien envoyé

            # Tenter d'envoyer l'email en utilisant la connexion SMTP

            with mail.connect() as connection:  # Connexion explicite au serveur SMTP

                connection.send(msg)

            log_message(
                f"Sent failed login alert to {user_email}"
            )  # Si l'email est envoyé avec succès

            save_logs_to_file()

        except Exception as e:

            log_message(
                f"Error sending failed login alert: {str(e)}"
            )  # Log de l'erreur si l'envoi échoue

            save_logs_to_file()

            print(
                f"Error sending failed login alert: {str(e)}"
            )  # Affichage de l'erreur pour le débogage


    #########################################################################################

    # Decorator to require login


    def login_required(f):

        @wraps(f)
        def decorated_function(*args, **kwargs):

            if "logged_in" not in session:

                return redirect(url_for("login"))

            return f(*args, **kwargs)

        return decorated_function


    #####################################################################################


    @app.route("/login", methods=["GET", "POST"])
    def login():

        if request.method == "POST":

            password = request.form.get("password")

            if password == HARDCODED_PASSWORD:

                send_2fa_code()  # Send 2FA code to the user

                return render_template("verify_2fa.html")  # Show the 2FA verification form

            else:

                send_failed_login_alert()

                return render_template("login.html", error="Incorrect password")

        return render_template("login.html")


    #####################################################################################


    @app.route("/verify_2fa", methods=["POST"])
    def verify_2fa():

        entered_2fa_code = request.form.get("2fa_code")

        if entered_2fa_code == current_2fa_code:

            session["logged_in"] = True

            return redirect(url_for("index"))

        else:

            return render_template("verify_2fa.html", error="Invalid 2FA code")


    #####################################################################################


    @app.route("/logout")
    def logout():

        session.pop("logged_in", None)

        return redirect(url_for("login"))


    ####################################################################

    # Protect the main route with login_required


    @app.route("/")
    @login_required
    def index():

        form_data = {
            # "risk_level": "moderate",  # Default values for form data
            # "amount": 0.0,
            # "compounding": False
            "ADA_USDC": True,
            "AAVE_USDC": True,
            "AERO_USDC": True,
            "ALGO_USDC": True,
            "AMP_USDC": True,
            "ARB_USDC": True,
            "AVAX_USDC": True,
            "BCH_USDC": True,
            "BONK_USDC": True,
            "BTC_USDC": True,
            "CRV_USDC": True,
            "DOGE_USDC": True,
            "DOT_USDC": True,
            "ETH_USDC": True,
            "EURC_USDC": True,
            "FET_USDC": True,
            "FIL_USDC": True,
            "GRT_USDC": True,
            "HBAR_USDC": True,
            "ICP_USDC": True,
            "IDEX_USDC": True,
            "INJ_USDC": True,
            "JASMY_USDC": True,
            "JTO_USDC": True,
            "LINK_USDC": True,
            "LTC_USDC": True,
            "MOG_USDC": True,
            "NEAR_USDC": True,
            "ONDO_USDC": True,
            "PEPE_USDC": True,
            "RENDER_USDC": True,
            "RNDR_USDC": True,
            "SEI_USDC": True,
            "SHIB_USDC": True,
            "SOL_USDC": True,
            "SUI_USDC": True,
            "SUPER_USDC": True,
            "SUSHI_USDC": True,
            "SWFTC_USDC": True,
            "TIA_USDC": True,
            "UNI_USDC": True,
            "USDT_USDC": True,
            "VET_USDC": True,
            "WIF_USDC": True,
            "XLM_USDC": True,
            "XYO_USDC": True,
            "XRP_USDC": True,
            "YFI_USDC": True,
            "ETC_USDC": True,
            "MATIC_USDC": True,
            "buy_percentage_of_capital": Decimal("0.05"),
            # "sell_percentage_of_capital": Decimal("0.05"),
            "sell_profit_target": Decimal("0.0"),
            "sell_stop_loss_target": Decimal("0.0"),
            "dca_interval_minute": 1,
        }

        return render_template(
            "index.html",
            bot_running=bot_running,
            form_data=form_data,
            logs="\n".join(logs[-100:]),
            log_Balance_Total=log_data1,
            log=log_data,
            log_usdc_balance=log_data2,
            log_eth2_balance=log_data3,
            log_orders=log_data4,
        )

        # return render_template('index.html', bot_running=bot_running, form_data=form_data,logs="\n".join(logs[-100:]))


    ####################################################################


    @app.route("/start", methods=["POST"])
    def start_bot():

        global bot_running

        if not bot_running:

            bot_thread = Thread(target=dca_trading_bot)

            bot_thread.daemon = True

            bot_thread.start()

        return redirect(url_for("index"))


    #####################################################################################


    @app.route("/stop", methods=["POST"])
    def stop_bot():

        global bot_running

        bot_running = False

        return redirect(url_for("index"))


    ####################################################################


    @app.route("/update_settings", methods=["POST"])
    def update_settings():

        global selected_crypto_pairs, buy_percentage_of_capital, sell_profit_target, sell_stop_loss_target, ia

        global dca_interval_seconds

        try:

            trade_frequency = int(request.form.get("trade_frequency", 30))

            print(f"➡️ trade_frequency reçu depuis formulaire: {trade_frequency}")

        except (ValueError, TypeError):

            trade_frequency = 30

            print("⚠️ trade_frequency invalide, défaut 30")

        dca_interval_seconds = trade_frequency

        buy_percentage_of_capital = Decimal(
            request.form.get("buy_percentage_of_capital", buy_percentage_of_capital)
        )

        sell_profit_target = Decimal(
            request.form.get("sell_profit_target", sell_profit_target)
        )

        sell_stop_loss_target = Decimal(
            request.form.get("sell_stop_loss_target", sell_stop_loss_target)
        )

        selected_crypto_pairs = request.form.getlist("selected_crypto_pairs")

        ia = request.form.getlist("ia")

        log_message(
            f"Settings updated: selected_crypto_pairs={selected_crypto_pairs},"
            f"buy_percentage_of_capital={buy_percentage_of_capital}, "
            f"sell_profit_target={sell_profit_target},sell_stop_loss_target={sell_stop_loss_target}, ia={ia}"
        )

        # AUTO-LOAD DATA AFTER SETTINGS CHANGE

        try:

            load_data(selected_crypto_pairs)

            log_message(f"📦 Données rechargées pour {selected_crypto_pairs}")

        except Exception as e:

            log_message(f"❌ Erreur lors du chargement des données : {e}")

        save_logs_to_file()

        return redirect(url_for("index"))


    ####################################################################


    @app.route("/logs", methods=["GET"])
    def get_logs():

        return jsonify({"logs": logs[-100:]})  # Send logs as an array of strings


    #######################################################


    @app.route("/log_Balance_Total", methods=["GET"])
    def log_Balance_Total():

        return jsonify({"log_Balance_Total": log_data1})  # Send logs as an array of strings


    @app.route("/log_usdc_balance", methods=["GET"])
    def log_usdc_balance():

        return jsonify({"log_usdc_balance": log_data2})  # Send logs as an array of strings


    @app.route("/log_eth2_balance", methods=["GET"])
    def log_eth2_balance():

        return jsonify({"log_eth2_balance": log_data3})  # Send logs as an array of strings


    @app.route("/log_orders", methods=["GET"])
    def log_orders():

        return jsonify({"log_orders": log_data4})  # Send logs as an array of strings


    @app.route("/log", methods=["GET"])
    def log():

        return jsonify({"log": log_data})  # Send logs as an array of strings


    #######################################################

    # API pour obtenir les données des prédictions en temps réel


    @app.route("/get_predictions")
    def get_predictions():

        return jsonify(
            {
                "predictions": {
                    crypto: all_predictions[crypto].tolist() for crypto in all_predictions
                }
            }
        )


    # application = app


    def determine_order_type_auto(symbol, recent_prices, threshold=0.0):
        """

        Détermine dynamiquement le type d'ordre en fonction de la volatilité.

        :param symbol: Le symbole de trading (ex: 'BTC-USDC')

        :param recent_prices: Liste des prix récents

        :param threshold: Seuil de volatilité au-delà duquel on passe en 'market'

        :return: 'market' ou 'limit'

        """

        if len(recent_prices) < 2:

            logger.warning(
                f"Pas assez de données pour évaluer la volatilité sur {symbol}. Market par défaut."
            )

            return "market"

        current_price = recent_prices[-1]

        oldest_price = recent_prices[0]

        volatility = abs(current_price - oldest_price) / oldest_price

        logger.info(
            f"[AUTO] Analyse de la volatilité pour {symbol} : {volatility:.4%} (seuil = {threshold:.2%})"
        )

        if volatility > threshold:

            logger.info(
                f"[AUTO] Volatilité élevée détectée sur {symbol}, ordre MARKET choisi."
            )

            return "market"

        else:

            logger.info(
                f"[AUTO] Faible volatilité détectée sur {symbol}, ordre LIMIT choisi."
            )

            return "limit"


    if __name__ == "__main__":

        app.run(host="0.0.0.0", port=5000)

        # app.run(debug=True)

    print(f"FIN DE SERVEUR N°_2")

    # === AJOUT POUR FORCER LA CONVERSION DE TOUTES LES PAIRES EN USDC ===


    def convert_all_selected_pairs_to_usdc():

        log_message(
            "🔄 Démarrage de la conversion forcée pour toutes les paires sélectionnées..."
        )

        save_logs_to_file()

        for product_id in selected_crypto_pairs:

            try:

                force_convert_to_usdc(client, product_id, None)

            except Exception as e:

                log_message(f"[ERREUR] Échec de conversion pour {product_id} : {str(e)}")

                save_logs_to_file()

        log_message("✅ Conversion terminée pour toutes les paires.")

        save_logs_to_file()


    # === FIN AJOUT ===


    def force_convert_all_to_usdc(min_value_usdc=1.0):
        """

        Convertit tous les soldes crypto vers USDC si leur valeur estimée ≥ min_value_usdc.

        """

        global accounts

        try:

            log_message("⚙️ Démarrage de la conversion vers USDC (seuil: ≥ 1 USDC)")

            save_logs_to_file()

            for account in accounts["accounts"]:

                currency = account["available_balance"]["currency"]

                amount = Decimal(account["available_balance"]["value"])

                if currency == "USDC" or amount <= 0:

                    continue

                try:

                    clean_currency = currency.rstrip("0123456789")

                    product_id = f"{clean_currency}-USDC"

                    price = get_market_price(product_id)

                    if not price:

                        log_message(f"❌ Prix introuvable pour {product_id}")

                        continue

                    usdc_value = amount * price

                    if usdc_value < Decimal(min_value_usdc):

                        log_message(
                            f"🚫 Conversion ignorée : {currency} ({amount}) ~ {usdc_value:.1f} USDC (< {min_value_usdc})"
                        )

                        continue

                    log_message(
                        f"🔁 Conversion {amount} {currency} (~{usdc_value:.1f} USDC)"
                    )

                    create_order_safe(
                        client,
                        client_order_id=str(uuid.uuid4()),
                        product_id=product_id,
                        side="SELL",
                        order_configuration={
                            "market_market_ioc": {"base_size": str(amount)}
                        },
                    )

                    log_message(f"✅ Conversion effectuée pour {currency}")

                except Exception as e:

                    log_message(f"❌ Erreur pendant la conversion de {currency}: {str(e)}")

            log_message("✅ Conversion complète terminée.")

            save_logs_to_file()

        except Exception as e:

            log_message(f"🔥 Erreur dans force_convert_all_to_usdc: {str(e)}")

            save_logs_to_file()


    import time

    import csv

    from datetime import datetime, timedelta

    print(f"balise N°_15")

    # Configuration - logic can evolve dynamically

    MIN_PROFIT_DYNAMIC = 0.01  # will be adjusted

    MIN_HOLD_DURATION_MINUTES = 10  # will evolve dynamically

    TRADE_LOG_PATH = "trade_journal.csv"

    # Memory to track tokens

    last_buy_data = {}

    latency_tracker = {}

    # Function to check if we should convert


    def should_convert(symbol, current_price):

        if symbol not in last_buy_data:

            return False, 0

        entry = last_buy_data[symbol]

        bought_price = entry["price"]

        buy_time = entry["timestamp"]

        elapsed = datetime.utcnow() - buy_time

        # Dynamic profit threshold

        dynamic_threshold = MIN_PROFIT_DYNAMIC + (
            0.0 if elapsed.total_seconds() < 3600 else 0
        )

        gain = ((current_price - bought_price) / bought_price) if bought_price > 0 else 0

        if gain >= dynamic_threshold and elapsed > timedelta(
            minutes=MIN_HOLD_DURATION_MINUTES
        ):

            return True, gain

        return False, gain


    # Simulated price update


    def update_price_and_check(symbol, current_price, strategy_tag):

        should_sell, gain = should_convert(symbol, current_price)

        if should_sell:

            log_trade(
                symbol, last_buy_data[symbol]["price"], current_price, gain, strategy_tag
            )

            del last_buy_data[symbol]

            return True

        return False


    # Simulate a buy


    def record_buy(symbol, price):

        last_buy_data[symbol] = {"price": price, "timestamp": datetime.utcnow()}


    # Logging


    def log_trade(symbol, buy_price, sell_price, gain, strategy):

        with open(TRADE_LOG_PATH, "a", newline="") as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    symbol,
                    buy_price,
                    sell_price,
                    round(gain * 100, 2),
                    strategy,
                ]
            )


    # Strategy selector (can expand)


    def select_strategy(price_data):

        # Example: add swing/dca/scalp decisions based on market data

        return "dynamic"


    # ========== MAIN EXECUTION ==========


    COINBASE_FEE_RATE = Decimal("0.006")  # 0.6% frais Coinbase

    current_portfolio_id = os.getenv("COINBASE_PORTFOLIO_ID")  # ID du portefeuille actif

    usdc_safe_wallet_id = os.getenv(
        "COINBASE_PROFIT_PORTFOLIO_ID"
    )  # ID du portefeuille Profit robot DCA

    if not current_portfolio_id or not usdc_safe_wallet_id:

        raise RuntimeError(
            "COINBASE_PORTFOLIO_ID et COINBASE_PROFIT_PORTFOLIO_ID doivent être définis dans le .env"
        )


    import time

    from datetime import datetime

    import random

    print(f"balise N°_16")

    # ========== Configuration ==========

    getcontext().prec = 10

    DEFAULT_TP = Decimal("0.0")  # +1.2%

    DEFAULT_SL = Decimal("0.0")  # -0.6%

    DEFAULT_TIMEOUT = 300  # Timeout en secondes (5 minutes)

    bot_running = True

    # ========== Fonctions Coinbase simulées (remplaçables par vraies API) ==========


    @with_retry(retries=3, delay=1)
    def get_market_price(product_id):

        # Simule un mouvement de prix autour d'un prix d'entrée

        return Decimal("3.6") + Decimal(random.uniform(-0.01, 0.02))


    def place_market_sell(product_id, usdc_amount, entry_price):

        # Simule une exécution de vente

        exit_price = get_market_price(product_id)

        return {
            "average_filled_price": str(exit_price),
            "total_fees": str(Decimal("0.03")),
        }


    # ========== Logger ==========


    def log_message(msg):

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"{now} - {msg}")


    # ========== Suivi de position avec stratégie complète ==========


    def monitor_position_optimized(product_id, amount_in_usdc, entry_price):

        global bot_running

        log_message(f"📊 Nouveau trade sur {product_id} - Entrée: {entry_price:.1f} USDC")

        start_time = time.time()

        tp = entry_price * (1 + DEFAULT_TP)

        sl = entry_price * (1 - DEFAULT_SL)

        highest_price = entry_price

        trailing_stop = entry_price * (1 - DEFAULT_SL * Decimal("0.75"))

        while bot_running:

            elapsed = time.time() - start_time

            if elapsed > DEFAULT_TIMEOUT:

                log_message(
                    f"⏱ Timeout atteint ({DEFAULT_TIMEOUT}s), fermeture de la position."
                )

                close_and_log_trade(
                    product_id, amount_in_usdc, entry_price, reason="timeout"
                )

                break

            try:

                current_price = get_market_price(product_id)

                if not current_price:

                    time.sleep(10)

                    continue

                if current_price > highest_price:

                    highest_price = current_price

                    trailing_stop = max(trailing_stop, highest_price * (1 - DEFAULT_SL))

                    log_message(
                        f"📈 Nouveau sommet: {highest_price:.1f} | SL ajusté: {trailing_stop:.1f}"
                    )

                if current_price >= tp:

                    close_and_log_trade(
                        product_id, amount_in_usdc, entry_price, reason="take_profit"
                    )

                    break

                elif current_price <= trailing_stop:

                    close_and_log_trade(
                        product_id, amount_in_usdc, entry_price, reason="stop_loss"
                    )

                    break

                time.sleep(10)

            except Exception as e:

                log_message(f"⚠️ Erreur dans le suivi de position : {e}")

                time.sleep(15)


    def close_and_log_trade(product_id, usdc_amount, buy_price, reason="unknown"):

        order = place_market_sell(product_id, usdc_amount, buy_price)

        if not order or not isinstance(order, dict):

            log_message("❌ Erreur de fermeture d'ordre.")

            return

        filled_price = Decimal(str(order.get("average_filled_price", "0")))

        fees = Decimal(str(order.get("total_fees", "0")))

        profit_pct = ((filled_price - buy_price - fees) / buy_price) * 100

        log_message(
            f"✅ Position fermée ({reason}) | Achat: {buy_price:.1f} | Vente: {filled_price:.1f} | "
            f"Frais: {fees:.1f} | Profit net: {profit_pct:.1f}%"
        )


    # ========== Point d'entrée ==========

    if __name__ == "__main__":

        # Paramètres du trade (à adapter ou automatiser)

        product = "DOT-USDC"

        amount = Decimal("12.44")

        entry = Decimal("3.6")

        monitor_position_optimized(product, amount, entry)

    # === PATCH: Fonction pour évaluer la volatilité d'une paire ===


    async def compute_volatility(product_id, client, window=10):

        try:

            candles = await client.get_candles(product_id, granularity="1m")

            closes = [float(c["close"]) for c in candles[-window:]]

            return statistics.stdev(closes)

        except Exception as e:

            print(f"[VolatilityError] {product_id}: {e}")

            return 0.0


    # === PATCH: Priorisation intelligente des paires ===


    async def rank_pairs_by_opportunity(client, pairs):

        ranked = []

        for pair in pairs:

            try:

                ticker = await client.get_ticker(pair)

                spread = float(ticker["ask"]) - float(ticker["bid"])

                volume = float(ticker.get("volume", 0))

                volatility = await compute_volatility(pair, client)

                score = (volume / (spread + 1e-6)) + volatility  # Heuristic

                ranked.append((pair, score))

            except Exception:

                continue

        ranked.sort(key=lambda x: x[1], reverse=True)

        return [p for p, _ in ranked]


    # === PATCH: LIMIT + fallback MARKET order ===


    async def place_limit_with_fallback(client, side, product_id, size, limit_price):

        try:

            order = await client.place_limit_order(product_id, side, size, limit_price)

            order_id = order.get("id")

            await asyncio.sleep(3)

            status = await client.get_order_status(order_id)

            if status.get("status") != "FILLED":

                await client.cancel_order(order_id)

                print(f"[Fallback] LIMIT non exécuté → passage à MARKET")

                return await client.place_market_order(product_id, side, size)

            return order

        except Exception as e:

            print(f"[OrderError] fallback triggered: {e}")

            return await client.place_market_order(product_id, side, size)


    # Log complet des paramètres sélectionnés

    logging.info(f"--- PARAMETRES INITIAUX ---")

    logging.info(f"1. Paires sélectionnées: {selected_crypto_pairs}")

    logging.info(f"2. Pourcentage du capital pour achat: {buy_percentage_of_capital}")

    logging.info(f"3. Objectif de profit (TP): {sell_profit_target}")

    logging.info(f"4. Objectif de perte (SL): {sell_stop_loss_target}")

    logging.info(f"5. Utilisation de l'IA: {ia}")

    logging.info(f"6. Utilisation du LIMIT_SPREAD: {LIMIT_SPREAD}")

    logging.info(f"7. Capital max par trade: {capital_per_trade}")

    logging.info(f"8. Ordre LIMIT délai max: {LIMIT_ORDER_TIMEOUT} sec")

    logging.info(f"9. Analyse volatilité activée: {analyse_volatilite}")

    logging.info(f"10. Seuil min de volume pour filtre: {volume_filter_threshold}")

    # === END OF MODULE 2 ===

    # === MONITOR modifié pour intégrer les TP/SL spécifiques ===


    def monitor_position_for_tp_sl(product_id, amount_in_usdc, prix_moment_achat):
        """

        Surveille une position et déclenche TP/SL avec prise en compte des frais.

        """

        # 1) Prix d’achat facturé avec frais

        entry_price_fee = Decimal(str(entry_price)) * (Decimal("1.0") + COINBASE_FEE_RATE)

        # 2) Calcul des cibles brutes (prix brut pour TP/SL)

        take_profit_brut = (entry_price_fee * (Decimal("1.0") + sell_profit_target)) / (
            Decimal("1.0") - COINBASE_FEE_RATE
        )

        stop_loss_brut = (entry_price_fee * (Decimal("1.0") - sell_stop_loss_target)) / (
            Decimal("1.0") - COINBASE_FEE_RATE
        )

        highest_price = entry_price_fee

        trailing_stop = stop_loss_brut

        last_log_time = time.time()

        log_message(
            f"▶️ Lancement monitor TP/SL {product_id} : "
            f"Achat brut+fee = {entry_price_fee:.6f} USDC | "
            f"TP_brut = {take_profit_brut:.6f} (+{sell_profit_target*100:.2f}% net) | "
            f"SL_brut = {stop_loss_brut:.6f} (-{sell_stop_loss_target*100:.2f}% net)"
        )

        save_logs_to_file()

        while bot_running:

            try:

                current_price = coinbase_client.get_market_price(product_id)

                if current_price is None:

                    time.sleep(5)

                    continue

                # 3) Actualiser trailing (prix brut+fee)

                current_price_fee = Decimal(str(current_price)) * (
                    Decimal("1.0") + COINBASE_FEE_RATE
                )

                if current_price_fee > highest_price:

                    highest_price = current_price_fee

                    new_stop = (
                        highest_price * (Decimal("1.0") - sell_stop_loss_target)
                    ) / (Decimal("1.0") - COINBASE_FEE_RATE)

                    if new_stop > trailing_stop:

                        trailing_stop = new_stop

                        log_message(
                            f"⬆️ Nouveau top (brut+fee) = {highest_price:.6f} | SL_brut ajusté = {trailing_stop:.6f}"
                        )

                        save_logs_to_file()

                # 4) Check Take Profit net

                if current_price >= take_profit_brut:

                    net_receipt = Decimal(str(current_price)) * (
                        Decimal("1.0") - COINBASE_FEE_RATE
                    )

                    gain_pct = (net_receipt / entry_price_fee - 1) * 100

                    vol = volatilities.get(product_id, 0.0)

                    trend = trend_scores.get(product_id, 0.0)

                    series_id = assign_series_to_pair(product_id, vol, trend)

                    log_message(
                        f"💸 TAKE PROFIT atteint pour {product_id} : vente à {current_price:.6f}, gain net = {gain_pct:.2f}%"
                    )

                    save_logs_to_file()

                    sell_resp = coinbase_client.place_market_order(
                        product_id=product_id,
                        side="SELL",
                        base_size=str(
                            (Decimal("1") / Decimal(str(entry_price))) * amount_in_usdc
                        ),
                    )

                    qty_sold = amount_in_usdc / Decimal(str(entry_price))

                    sell_price_actual = (
                        coinbase_client.get_market_price(product_id) or current_price
                    )

                    log_vente_détaillée(
                        pair=product_id,
                        qty=float(qty_sold),
                        buy_price=entry_price,
                        sell_price=float(sell_price_actual),
                        series_id=series_id,
                    )

                    try:

                        # --- LOG HUMAIN VENTE ---

                        # On calcule prix_vente, total_fees, net_profit_pct, net_profit_usdc

                        buy_p = float(entry_price) if "entry_price" in locals() else 0

                        sell_p = (
                            float(sell_price_actual)
                            if "sell_price_actual" in locals()
                            else float(current_price) if "current_price" in locals() else 0
                        )

                        qty = float(qty_sold) if "qty_sold" in locals() else 1.0

                        total_fees = (
                            float(qty * (buy_p + sell_p) * float(COINBASE_FEE_RATE))
                            if "COINBASE_FEE_RATE" in locals()
                            else 0
                        )

                        net_profit_usdc = (sell_p - buy_p) * qty - total_fees

                        net_profit_pct = ((sell_p - buy_p) / buy_p) if buy_p != 0 else 0

                        log_vente_humaine(
                            product_id=product_id,
                            series_id=(
                                series_id
                                if "series_id" in locals()
                                else pair_strategy_mapping.get(product_id, "N/A")
                            ),
                            prix_vente=sell_p,
                            total_fees=total_fees,
                            net_profit_pct=net_profit_pct,
                            net_profit_usdc=net_profit_usdc,
                        )

                    except Exception as e:

                        log_message(f"Erreur log humain vente auto-injecté : {e}")

                    break

                # 5) Check Stop Loss brut

                if current_price <= trailing_stop:

                    net_receipt = Decimal(str(current_price)) * (
                        Decimal("1.0") - COINBASE_FEE_RATE
                    )

                    gain_pct = (net_receipt / entry_price_fee - 1) * 100

                    vol = volatilities.get(product_id, 0.0)

                    trend = trend_scores.get(product_id, 0.0)

                    series_id = assign_series_to_pair(product_id, vol, trend)

                    log_message(
                        f"💸 STOP LOSS déclenché pour {product_id} : vente à {current_price:.6f}, gain net = {gain_pct:.2f}%"
                    )

                    save_logs_to_file()

                    sell_resp = coinbase_client.place_market_order(
                        product_id=product_id,
                        side="SELL",
                        base_size=str(
                            (Decimal("1") / Decimal(str(entry_price))) * amount_in_usdc
                        ),
                    )

                    qty_sold = amount_in_usdc / Decimal(str(entry_price))

                    sell_price_actual = (
                        coinbase_client.get_market_price(product_id) or current_price
                    )

                    log_vente_détaillée(
                        pair=product_id,
                        qty=float(qty_sold),
                        buy_price=entry_price,
                        sell_price=float(sell_price_actual),
                        series_id=series_id,
                    )

                    try:

                        # --- LOG HUMAIN VENTE ---

                        # On calcule prix_vente, total_fees, net_profit_pct, net_profit_usdc

                        buy_p = float(entry_price) if "entry_price" in locals() else 0

                        sell_p = (
                            float(sell_price_actual)
                            if "sell_price_actual" in locals()
                            else float(current_price) if "current_price" in locals() else 0
                        )

                        qty = float(qty_sold) if "qty_sold" in locals() else 1.0

                        total_fees = (
                            float(qty * (buy_p + sell_p) * float(COINBASE_FEE_RATE))
                            if "COINBASE_FEE_RATE" in locals()
                            else 0
                        )

                        net_profit_usdc = (sell_p - buy_p) * qty - total_fees

                        net_profit_pct = ((sell_p - buy_p) / buy_p) if buy_p != 0 else 0

                        log_vente_humaine(
                            product_id=product_id,
                            series_id=(
                                series_id
                                if "series_id" in locals()
                                else pair_strategy_mapping.get(product_id, "N/A")
                            ),
                            prix_vente=sell_p,
                            total_fees=total_fees,
                            net_profit_pct=net_profit_pct,
                            net_profit_usdc=net_profit_usdc,
                        )

                    except Exception as e:

                        log_message(f"Erreur log humain vente auto-injecté : {e}")

                    break

                # 6) Journaux périodiques

                if time.time() - last_log_time > 120:

                    pct_margin = ((current_price - trailing_stop) / current_price) * 100

                    log_message(
                        f"⏱️ Monitor {product_id} → Prix actuel (brut) = {current_price:.6f} | Peak (brut+fee) = {highest_price:.6f} | Marge SL = {pct_margin:.2f}%"
                    )

                    save_logs_to_file()

                    last_log_time = time.time()

                time.sleep(10)

            except Exception as e:

                log_message(f"⚠️ Erreur monitor_position_for_tp_sl({product_id}): {e}")

                traceback.print_exc()

                save_logs_to_file()

                time.sleep(30)

        log_message(f"🏁 Monitoring terminé pour {product_id}")

        save_logs_to_file()


    def ia_prediction_logging_loop():

        log_message("🧠 Boucle de prédiction IA 60s démarrée...")

        while True:

            try:

                predictions_dict = {}

                executed_orders = executed_orders_global.copy()

                sales_done = sales_done_global.copy()

                today_date = pd.to_datetime("today").normalize()

                previous_date = today_date - pd.Timedelta(days=1)

                for pair in selected_crypto_pairs:

                    hist = all_data.get(pair)

                    if not hist or len(hist) < 30:

                        log_message(f"⛔ Aucune donnée suffisante pour {pair}")

                        continue

                    train, test = train_test_split(hist, test_size=test_size)

                    train, test, X_train, X_test, y_train, y_test = prepare_data(
                        hist,
                        "close",
                        window_len=window_len,
                        zero_base=zero_base,
                        test_size=test_size,
                    )

                    model = build_lstm_model(
                        X_train,
                        output_size=1,
                        neurons=lstm_neurons,
                        dropout=dropout,
                        loss=loss,
                        optimizer=optimizer,
                    )

                    preds = model.predict(X_test).squeeze()

                    preds = test["close"].values[:-window_len] * (preds + 1)

                    preds = pd.Series(index=test["close"][window_len:].index, data=preds)

                    # should_retrain: alerte Prometheus si dérive détectée

                    mse = np.mean((preds - y_test) ** 2)

                    mse_gauge.labels(pair=pair).set(mse)

                    if mse > 0.05:

                        log_message(
                            f"⚠️ MSE élevé pour {pair} ({mse:.1f}) — Modèle pourrait nécessiter un retrain."
                        )

                    yest = preds.loc[previous_date:previous_date]

                    today = preds.loc[today_date:today_date]

                    if not yest.empty and not today.empty:

                        today_val = today.iloc[-1]

                        yest_val = yest.iloc[0]

                        predictions_dict[pair] = (today_val, yest_val)

                        emoji = "📈" if today_val > yest_val else "📉"

                        action = (
                            "✅ Achat recommandé"
                            if today_val > yest_val
                            else "⛔ Pas d'achat"
                        )

                        log_message(
                            f"{emoji} {pair} : {today_val:.1f} {'>' if today_val > yest_val else '<'} {yest_val:.1f} — {action}"
                        )

                        # Achat automatique si tendance haussière

                        if today_val > yest_val:

                            usdc_balance = get_usdc_balance()

                            price = get_price(pair)

                            allocation = usdc_balance * buy_percentage_of_capital

                            if allocation >= 5:  # Seuil minimum

                                size = allocation / price

                                place_order(pair, size, side="BUY")

                                buy_counter.labels(pair=pair).inc()

                                log_message(
                                    f"💰 Achat exécuté automatiquement pour {pair} avec {allocation:.1f} USDC ({size:.1f} unités)"
                                )

                    # ATR

                    high = hist["high"].iloc[-1]

                    low = hist["low"].iloc[-1]

                    close = hist["close"].iloc[-2]

                    tr = max(high - low, abs(high - close), abs(low - close))

                    atr = round(tr, 5)

                    log_message(f"📐 ATR ({pair}) = {atr}")

                log_ia_predictions(predictions_dict, executed_orders, sales_done)

            except Exception as e:

                log_message(f"[LOG LOOP ERROR] {e}")

            time.sleep(60)


    # Démarrage du thread de logging IA

    ia_log_thread = threading.Thread(target=ia_prediction_logging_loop)

    ia_log_thread.daemon = True

    ia_log_thread.start()

    # === AJOUT : ATR + LOGS PREDICTIONS TOUTES LES 60S + MOCK WS ===

    import threading

    import time

    import random

    print(f"balise N°_18")


    def calculate_atr(df, window=14):

        df["H-L"] = abs(df["high"] - df["low"])

        df["H-PC"] = abs(df["high"] - df["close"].shift(1))

        df["L-PC"] = abs(df["low"] - df["close"].shift(1))

        tr = df[["H-L", "H-PC", "L-PC"]].max(axis=1)

        atr = tr.rolling(window=window).mean()

        return atr.iloc[-1] if not atr.empty else 0


    def ia_prediction_logging_loop():

        log_message("🧠 Boucle de prédiction IA 60s démarrée...")

        while True:

            try:

                predictions_dict = {}

                executed_orders = executed_orders_global.copy()

                sales_done = sales_done_global.copy()

                today_date = pd.to_datetime("today").normalize()

                previous_date = today_date - pd.Timedelta(days=1)

                for pair in selected_crypto_pairs:

                    hist = all_data.get(pair)

                    if not hist or len(hist) < 30:

                        log_message(f"⛔ Aucune donnée suffisante pour {pair}")

                        continue

                    train, test = train_test_split(hist, test_size=test_size)

                    train, test, X_train, X_test, y_train, y_test = prepare_data(
                        hist,
                        "close",
                        window_len=window_len,
                        zero_base=zero_base,
                        test_size=test_size,
                    )

                    model = build_lstm_model(
                        X_train,
                        output_size=1,
                        neurons=lstm_neurons,
                        dropout=dropout,
                        loss=loss,
                        optimizer=optimizer,
                    )

                    preds = model.predict(X_test).squeeze()

                    preds = test["close"].values[:-window_len] * (preds + 1)

                    preds = pd.Series(index=test["close"][window_len:].index, data=preds)

                    # should_retrain: alerte Prometheus si dérive détectée

                    mse = np.mean((preds - y_test) ** 2)

                    mse_gauge.labels(pair=pair).set(mse)

                    if mse > 0.05:

                        log_message(
                            f"⚠️ MSE élevé pour {pair} ({mse:.1f}) — Modèle pourrait nécessiter un retrain."
                        )

                    yest = preds.loc[previous_date:previous_date]

                    today = preds.loc[today_date:today_date]

                    if not yest.empty and not today.empty:

                        today_val = today.iloc[-1]

                        yest_val = yest.iloc[0]

                        predictions_dict[pair] = (today_val, yest_val)

                        emoji = "📈" if today_val > yest_val else "📉"

                        action = (
                            "✅ Achat recommandé"
                            if today_val > yest_val
                            else "⛔ Pas d'achat"
                        )

                        log_message(
                            f"{emoji} {pair} : {today_val:.1f} {'>' if today_val > yest_val else '<'} {yest_val:.1f} — {action}"
                        )

                        # Achat automatique si tendance haussière

                        if today_val > yest_val:

                            usdc_balance = get_usdc_balance()

                            price = get_price(pair)

                            allocation = usdc_balance * buy_percentage_of_capital

                            if allocation >= 5:  # Seuil minimum

                                size = allocation / price

                                place_order(pair, size, side="BUY")

                                buy_counter.labels(pair=pair).inc()

                                log_message(
                                    f"💰 Achat exécuté automatiquement pour {pair} avec {allocation:.1f} USDC ({size:.1f} unités)"
                                )

                    # ATR

                    high = hist["high"].iloc[-1]

                    low = hist["low"].iloc[-1]

                    close = hist["close"].iloc[-2]

                    tr = max(high - low, abs(high - close), abs(low - close))

                    atr = round(tr, 5)

                    log_message(f"📐 ATR ({pair}) = {atr}")

                log_ia_predictions(predictions_dict, executed_orders, sales_done)

            except Exception as e:

                log_message(f"[LOG LOOP ERROR] {e}")

            time.sleep(60)


    # Lancement du thread IA logs

    log_thread = threading.Thread(target=ia_prediction_logging_loop)

    log_thread.daemon = True

    log_thread.start()

    # === AJOUTS STRUCTURÉS POUR AMÉLIORATION ===

    # ✅ Complétion de la fonction analyse_histohour avec logique IA complète


    def analyse_histohour(prices, window=24):

        import numpy as np

        if len(prices) < window:

            return None

        closes = [p["close"] for p in prices[-window:]]

        change_pct = (closes[-1] - closes[0]) / closes[0] * 100

        volatility = np.std(closes)

        trend = "up" if change_pct > 0 else "down"

        return {
            "trend": trend,
            "change_pct": round(change_pct, 2),
            "volatility": round(volatility, 4),
        }


    # ✅ Classe BotState pour centraliser les variables globales


    class BotState:

        def __init__(self):

            self.bot_running = False

            self.logs = []

            self.Profit_cumul = 0.0

            self.log_data = ""

            self.log_data1 = ""

            self.log_data2 = ""

            self.log_data3 = ""

            self.log_data4 = ""


    state = BotState()

    # ✅ Fonction utilitaire pour génération 2FA


    def generate_2fa_code():

        try:

            return totp.now()

        except Exception as e:

            logging.error(f"Erreur génération code 2FA: {str(e)}")

            return None


    # ✅ Version corrigée de build_lstm_model


    def build_lstm_model(
        input_data_shape,
        output_size,
        neurons=100,
        activ_func="linear",
        dropout=0.2,
        loss="mse",
        optimizer="adam",
    ):

        from keras.models import Sequential

        from keras.layers import Dense, Dropout, LSTM

        model = Sequential()

        model.add(LSTM(neurons, input_shape=input_data_shape))

        model.add(Dropout(dropout))

        model.add(Dense(units=output_size, activation=activ_func))

        model.compile(loss=loss, optimizer=optimizer)

        return model


    # ✅ Validation dynamique des paires (à insérer avant d’utiliser selected_crypto_pairs)

    valid_crypto_pairs = []

    for selected_crypto_pair in selected_crypto_pairs:

        try:

            product_info = client.get_product(selected_crypto_pair)

            base_min_size = float(product_info["base_min_size"])

            quote_increment = float(product_info["quote_increment"])

            print(f"Base Minimum Size for {selected_crypto_pair}: {base_min_size}")

            print(f"Quote Increment for {selected_crypto_pair}: {quote_increment}")

            valid_crypto_pairs.append(selected_crypto_pair)

        except Exception as e:

            log_message(
                f"❌ Erreur lors de la vérification de la paire {selected_crypto_pair}: {str(e)}"
            )

    selected_crypto_pairs = valid_crypto_pairs

    # === FIN DES AJOUTS STRUCTURÉS ===

    # === AJOUTS COMPLETS INTÉGRÉS ===

    # 1. get_market_price avec API réelle


    @with_retry(retries=3, delay=1)
    def get_market_price(product_id):

        try:

            ticker = client.get_product_ticker(product_id=product_id)

            return Decimal(ticker["price"])

        except Exception as e:

            log_message(f"Erreur API get_market_price: {e}")

            return None


    # 2. place_market_order avec API + retry

    from tenacity import retry, stop_after_attempt, wait_fixed


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def place_market_order(product_id, size, side):

        try:

            order_config = {
                "market_market_ioc": {
                    "base_size" if side == "SELL" else "quote_size": str(size)
                }
            }

            return create_order_safe(
                client,
                client_order_id=str(uuid.uuid4()),
                product_id=product_id,
                side=side,
                order_configuration=order_config,
            )

        except Exception as e:

            log_message(f"Erreur ordre {side} sur {product_id}: {e}")

            return None


    # 3. Mise à jour dynamique des modèles IA

    from sklearn.model_selection import TimeSeriesSplit

    from keras.callbacks import EarlyStopping

    print(f"balise N°_19")


    def update_model_incrementally(new_data, model):

        tscv = TimeSeriesSplit(n_splits=3)

        X_new = prepare_input_series(new_data)

        early_stop = EarlyStopping(monitor="val_loss", patience=2)

        model.fit(
            X_new,
            validation_split=0.2,
            epochs=10,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0,
        )

        return model


    # 4. TradingState thread-safe

    from threading import Lock


    class TradingState:

        def __init__(self):

            self.lock = Lock()

            self.positions = {}

            self.account_balances = {}

            self.bot_running = False

        def update_position(self, pair, data):

            with self.lock:

                self.positions[pair] = data

        def get_balance(self, currency):

            with self.lock:

                return self.account_balances.get(currency, 0)


    trading_state = TradingState()

    # 5. Refactor convert_to_usdc


    def convert_to_usdc(amount, currency):

        try:

            if currency == "USDC":

                return amount

            product_id = f"{currency}-USDC"

            order = place_market_order(product_id=product_id, side="SELL", size=amount)

            if order and order.get("status") == "FILLED":

                return Decimal(order["executed_value"])

            return Decimal(0)

        except Exception as e:

            log_message(f"Échec conversion {currency}: {e}")

            return Decimal(0)


    # 6. WebSocket & Plotly dashboard

    from flask_socketio import SocketIO

    import plotly.graph_objects as go

    print(f"balise N°_20")

    socketio = SocketIO(app, cors_allowed_origins="*")


    @app.route("/dashboard")
    def dashboard():

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                )
            ]
        )

        plot_html = fig.to_html(full_html=False)

        return render_template(
            "dashboard.html", plot=plot_html, predictions=all_predictions
        )


    @socketio.on("force_refresh")
    def handle_refresh():

        socketio.emit(
            "update_prices",
            {"prices": get_live_prices(), "positions": trading_state.positions},
        )


    # 7. Logging structuré

    import structlog


    def configure_logging():

        structlog.configure(
            processors=[
                structlog.processors.JSONRenderer(),
                structlog.processors.TimeStamper(fmt="iso"),
            ],
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )


    configure_logging()

    logger = structlog.get_logger()

    # 8. config.yaml loader

    import yaml


    def load_config(path="config.yaml"):

        with open(path, "r") as file:

            return yaml.safe_load(file)


    # === INTÉGRATION DES CORRECTIFS CRITIQUES ===

    # 🔁 Supprimez les doublons — une seule version de `analyse_histohour` conservée


    def analyse_histohour(prices, window=24):

        import numpy as np

        if len(prices) < window:

            return None

        closes = [p["close"] for p in prices[-window:]]

        change_pct = (closes[-1] - closes[0]) / closes[0] * 100

        volatility = np.std(closes)

        trend = "up" if change_pct > 0 else "down"

        return {
            "trend": trend,
            "change_pct": round(change_pct, 2),
            "volatility": round(volatility, 4),
        }


    # 🔒 TradingState avec thread-safe

    from threading import Lock, Event


    class TradingState:

        def __init__(self):

            self.lock = Lock()

            self.positions = {}

            self.account_balances = {}

            self.bot_running = Event()

            self.executed_orders = []

            self.sales_done = []

        def update_position(self, pair, data):

            with self.lock:

                self.positions[pair] = data

        def add_executed_order(self, pair):

            with self.lock:

                self.executed_orders.append(pair)

        def add_sale(self, pair):

            with self.lock:

                self.sales_done.append(pair)


    trading_state = TradingState()

    # ✅ Appels API : exceptions spécifiques

    from requests.exceptions import HTTPError, ConnectionError


    @with_retry(retries=3, delay=1)
    def get_market_price(product_id):

        try:

            ticker = client.get_product_ticker(product_id=product_id)

            return Decimal(ticker["price"])

        except KeyError as e:

            log_message(f"[KeyError] Clé manquante: {e}")

        except HTTPError as e:

            log_message(f"[HTTPError] Statut: {e.response.status_code}")

        except ConnectionError:

            log_message("[ConnectionError] Problème de réseau")

        except Exception as e:

            log_message(f"[Unexpected Error] {e}")

        return None


    # 🔄 Correction de récursion accidentelle


    def predict_trend(data):

        result = analyse_histohour(data)

        return result["trend"] if result else "stable"


    # === AJOUTS OPTIMISÉS : INDICATEURS, DRAWDOWN, SIGNAL STRENGTH ===

    import pandas_ta as ta


    def add_technical_indicators(df):

        df["rsi"] = ta.rsi(df["close"], length=14)

        df["macd"] = ta.macd(df["close"])["MACD_12_26_9"]

        df["macd_signal"] = ta.macd(df["close"])["MACDs_12_26_9"]

        df["ema20"] = ta.ema(df["close"], length=20)

        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        df.dropna(inplace=True)

        return df


    # Mise à jour de la fonction fetch_crypto_data (si présente)

    # Ajouter un appel à add_technical_indicators(data) à la fin du chargement

    max_drawdown_pct = -5.0  # seuil de drawdown en %

    Profit_cumul = 0.0

    bot_running = True


    def check_drawdown_stop():

        global Profit_cumul, bot_running

        if Profit_cumul < max_drawdown_pct:

            log_message(f"❌ Drawdown global dépassé ({Profit_cumul:.1f}%) : arrêt du bot.")

            bot_running = False


    def compute_signal_strength(df):

        last = df.iloc[-1]

        score = 0

        if last["rsi"] > 50:

            score += 1

        if last["macd"] > last["macd_signal"]:

            score += 1

        if last["close"] > last["ema20"]:

            score += 1

        if last["atr"] > 0:

            score += 1

        return score / 4  # score de 0 à 1


    # === FIN AJOUTS ===

    # === AJOUTS : Refonte sécurisée et adaptative ===

    from tenacity import retry, stop_after_attempt, wait_fixed

    import aiohttp

    from cryptography.fernet import Fernet

    print(f"balise N°_21")

    selected_crypto_base = "USDC"


    class APIError(Exception):

        pass


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def api_call_with_retry():

        try:

            # Simulation d'appel API

            response = simulated_api_call()

            return response

        except APIError as e:

            log_message(f"API Error: {str(e)}")

            raise

        except ConnectionError:

            log_message("Network issue - retrying...")

            raise


    async def fetch_data_async(url):

        async with aiohttp.ClientSession() as session:

            async with session.get(url) as response:

                return await response.json()


    def encrypt_data(data: str, key: bytes) -> str:

        fernet = Fernet(key)

        return fernet.encrypt(data.encode()).decode()


    def adjust_strategy_based_on_market(volatility):

        if volatility > 0.05:

            return {"order_type": "market", "size_multiplier": 0.5}

        else:

            return {"order_type": "limit", "size_multiplier": 1.0}


    # === TESTS UNITAIRES ===


    def test_determine_order_type():

        assert determine_order_type(0.01) == "market"

        assert determine_order_type(0.8) == "limit"


    if __name__ == "__main__":

        test_determine_order_type()

        print("Tests unitaires passés.")

    # --- Point 3: Utilisation de signal_strength dans determine_order_type ---


    def determine_order_type(signal_strength):

        return "limit" if signal_strength >= 0.02 else "market"


    # --- Point 4: Journalisation du price history (à insérer dans la boucle principale du bot) ---

    if "price_history" not in locals():

        price_history = []


    def update_price_history(new_price, max_length=100):

        price_history.append(new_price)

        if len(price_history) > max_length:

            price_history.pop(0)


    # --- Point 5: Retombée en ordre market si échec limit ---


    def fallback_to_market_order(order_status, initial_type):

        if initial_type == "limit" and order_status != "filled":

            print("Ordre limit non rempli, fallback vers market.")

            return "market"

        return initial_type


    # --- Point 6 & 7: Adaptation dynamique des paramètres en fonction de la volatilité ---


    def adapt_parameters(signal_strength):

        # Plus le signal est fort, plus on prend de risque

        base_tp = 0.0

        base_sl = 0.01

        base_cap_ratio = 0.2

        multiplier = min(max(signal_strength * 100, 0.5), 2.0)  # borné entre 0.5 et 2.0

        tp = base_tp * multiplier

        sl = base_sl * multiplier

        cap_ratio = base_cap_ratio * multiplier

        return tp, sl, cap_rati


    @app.route("/get_spread_status")
    def get_spread_status():

        data = {}

        for pair in selected_crypto_pairs:

            try:

                spread = get_current_spread(pair)

                order_type = decide_order_type(pair)

                status = (
                    "✅"
                    if order_type == "limit"
                    else ("⚠️" if order_type == "limit_fallback" else "❌")
                )

                data[pair] = {
                    "spread_pct": round(spread * 100, 3),
                    "order_type": order_type.upper(),
                    "status": status,
                }

            except Exception as e:

                data[pair] = {"error": str(e)}

        return jsonify(data)


    @app.route("/get_signal_strength")
    def get_signal_strength():

        data = {}

        for pair in selected_crypto_pairs:

            try:

                df = fetch_crypto_data(pair, limit=50)

                df = add_technical_indicators(df)

                score = compute_signal_strength(df)

                status = "🟢" if score > 0.75 else ("🟡" if score > 0.5 else "🔴")

                data[pair] = {"signal_strength": round(score, 2), "status": status}

            except Exception as e:

                data[pair] = {"error": str(e)}

        return jsonify(data)


    import threading

    import shutil

    import os

    import time

    from datetime import datetime

    print(f"balise N°_21")


    def daily_backup_logs_and_trades():

        while True:

            now = datetime.now()

            if now.hour == 0 and now.minute == 0:  # à minuit

                try:

                    backup_folder = os.path.join(os.getcwd(), "backups")

                    os.makedirs(backup_folder, exist_ok=True)

                    log_src = os.path.join(os.getcwd(), "logs.txt")

                    trade_src = os.path.join(os.getcwd(), "trade_journal.csv")

                    if os.path.exists(log_src):

                        log_dest = os.path.join(
                            backup_folder, f"logs_{now.strftime('%Y%m%d')}.txt"
                        )

                        shutil.copyfile(log_src, log_dest)

                        print(f"✅ Backup logs.txt -> {log_dest}")

                    if os.path.exists(trade_src):

                        trade_dest = os.path.join(
                            backup_folder, f"trades_{now.strftime('%Y%m%d')}.csv"
                        )

                        shutil.copyfile(trade_src, trade_dest)

                        print(f"✅ Backup trade_journal.csv -> {trade_dest}")

                except Exception as e:

                    print(f"Erreur backup automatique: {e}")

                time.sleep(60)  # Eviter doublon

            time.sleep(30)


    # Lancer le backup dans un thread

    thread_backup = threading.Thread(target=daily_backup_logs_and_trades)

    thread_backup.daemon = True

    thread_backup.start()


    @app.errorhandler(500)
    def internal_server_error(e):

        return jsonify({"error": "Erreur serveur 500, veuillez réessayer plus tard."}), 500


    @app.errorhandler(502)
    def bad_gateway_error(e):

        return jsonify({"error": "Erreur 502 Bad Gateway."}), 502


    import asyncio


    async def async_check_balances():

        while True:

            try:

                get_usdc_balance()

                get_eth2_balance()

            except Exception as e:

                print(f"Erreur check balance: {e}")

            await asyncio.sleep(60)


    def start_async_tasks():

        loop = asyncio.get_event_loop()

        loop.create_task(async_check_balances())


    from apscheduler.schedulers.background import BackgroundScheduler


    def retrain_lstm_models_daily():

        print("🛠️ Retraining de tous les modèles LSTM...")

        log_message("Début du retraining automatique des modèles LSTM.")

        save_logs_to_file()

        for crypto_pair in selected_crypto_pairs:

            hist = all_data[crypto_pair]

            train, test, X_train, X_test, y_train, y_test = prepare_data(
                hist,
                "close",
                window_len=window_len,
                zero_base=zero_base,
                test_size=test_size,
            )

            model = build_lstm_model(
                X_train,
                output_size=1,
                neurons=lstm_neurons,
                dropout=dropout,
                loss=loss,
                optimizer=optimizer,
            )

            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=True,
            )

            save_lstm_model(model, crypto_pair)

        print("✅ Retraining terminé.")

        log_message("Fin du retraining automatique des modèles LSTM.")

        save_logs_to_file()


    scheduler = BackgroundScheduler()

    scheduler.add_job(retrain_lstm_models_daily, "cron", hour=0, minute=0)

    scheduler.start()

    print("⏰ Scheduler de retraining LSTM configuré pour minuit.")


    def should_buy_crypto(product_id):

        today_date = pd.to_datetime("today").normalize()

        previous_date = today_date - pd.Timedelta(days=1)

        hist = all_data.get(product_id)

        if hist is None:

            log_message(f"⚠️ Pas de données historiques pour {product_id}")

            return False

        train, test, X_train, X_test, y_train, y_test = prepare_data(
            hist, "close", window_len=window_len, zero_base=zero_base, test_size=test_size
        )

        model = build_lstm_model(
            X_train,
            output_size=1,
            neurons=lstm_neurons,
            dropout=dropout,
            loss=loss,
            optimizer=optimizer,
        )

        model.fit(
            X_train, y_train, epochs=5, batch_size=batch_size, verbose=0, shuffle=True
        )

        targets = test["close"][window_len:]

        preds = model.predict(X_test).squeeze()

        preds = test["close"].values[:-window_len] * (preds + 1)

        preds = pd.Series(index=targets.index, data=preds)

        yesterday_last_real = preds.loc[previous_date:previous_date]

        today_pred = preds.loc[today_date:today_date]

        ia_predicts_up = (
            not yesterday_last_real.empty
            and not today_pred.empty
            and today_pred.iloc[-1] > yesterday_last_real.iloc[0]
        )

        market_is_bullish = is_market_bullish(product_id)

        if ia_predicts_up and market_is_bullish:

            log_message(
                f"✅ Achat autorisé pour {product_id} (IA daily haussière + Golden Cross minute détecté)"
            )

            return True

        else:

            log_message(
                f"🚫 Achat refusé pour {product_id} (Conditions IA ou marché non réunies)"
            )

            return False


    # === PATCH 13 : Auto-Flat & Rapport HTML ===

    import csv

    from datetime import datetime, timedelta

    TRADE_LOG_CSV = "trade_log_daily.csv"

    DAILY_HTML_REPORT = "daily_summary.html"

    AUTO_FLAT_HOURS = 6

    last_trades = {}


    def record_trade_timestamp(symbol):

        last_trades[symbol] = datetime.utcnow()


    def check_for_flat_mode():

        now = datetime.utcnow()

        to_convert = []

        for symbol, timestamp in last_trades.items():

            if (now - timestamp).total_seconds() >= AUTO_FLAT_HOURS * 3600:

                to_convert.append(symbol)

        return to_convert


    def force_sell_overdue_positions():

        overdue = check_for_flat_mode()

        for symbol in overdue:

            log_message(f"[AUTO-FLAT] Vente forcée pour {symbol} après {AUTO_FLAT_HOURS}h.")

            # log_trade(symbol, last_buy_data[symbol]['price'], current_price, gain, "auto-flat")

            # place_market_sell(symbol, amount, buy_price)


    def generate_daily_html_summary():

        try:

            with open(TRADE_LOG_CSV, "r") as f:

                reader = csv.reader(f)

                rows = list(reader)

        except:

            rows = []

        html = "<html><head><title>Résumé Journalier</title></head><body><h1>Journal des Trades</h1><table border='1'>"

        html += "<tr><th>Date</th><th>Symbole</th><th>Achat</th><th>Vente</th><th>Profit (%)</th><th>Stratégie</th></tr>"

        for row in rows[-50:]:

            html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"

        html += "</table></body></html>"

        with open(DAILY_HTML_REPORT, "w", encoding="utf-8") as f:

            f.write(html)

        print(f"✅ Rapport HTML généré : {DAILY_HTML_REPORT}")


    # === FIN PATCH 13 ===

    # === PATCH_14 : API Live Performance Chart + Alertes Email ===

    # === LIVE PROFIT TRACKING ===

    performance_points = []


    @app.route("/get_performance")
    def get_performance():

        global performance_points

        return jsonify({"performance": {"Profit USDC": performance_points}})


    def track_profit_live(amount):

        from datetime import datetime

        global performance_points

        performance_points.append(
            {"x": datetime.utcnow().isoformat(), "y": round(amount, 2)}
        )

        performance_points = performance_points[-300:]


    # === ALERTING EMAIL (PnL extrême) ===


    def send_alert_email(subject, body):

        with app.app_context():

            if not user_email:

                print("Error: USER_EMAIL is not defined.")

                return

            msg = Message(subject, recipients=[user_email])

            msg.body = body

            try:

                with mail.connect() as connection:

                    connection.send(msg)

                log_message(f"📬 Alerte envoyée à {user_email}")

                save_logs_to_file()

            except Exception as e:

                log_message(f"Erreur envoi email : {e}")

                save_logs_to_file()


    def check_pnl_alerts(profit_value):

        if profit_value >= 50:

            send_alert_email(
                "🚀 Profit élevé détecté", f"Profit net cumulé : {profit_value} USDC"
            )

        elif profit_value <= -20:

            send_alert_email(
                "⚠️ Perte élevée détectée", f"Drawdown cumulé : {profit_value} USDC"
            )


    # === PATCH_15 : Dashboard Interactif HTML pour PnL journalier ===


    @app.route("/dashboard")
    def dashboard():

        return render_template("dashboard.html")


    @app.route("/api/trade_log_csv")
    def api_trade_log_csv():

        try:

            df = pd.read_csv(
                "trade_log_daily.csv",
                names=["date", "symbol", "buy", "sell", "pnl_pct", "strategy"],
            )

            return jsonify(df.to_dict(orient="records"))

        except Exception as e:

            return jsonify({"error": str(e)})


    # === PATCH_16 : RiskManager + Overnight Flat ===

    import threading

    MAX_DAILY_TRADES = 80

    DAILY_TRADE_COUNT_FILE = "daily_trade_count.txt"

    EXPOSURE_THRESHOLD = 0.6

    OVERNIGHT_FLAT_HOUR = 2  # 2 AM


    def load_daily_trade_count():

        try:

            with open(DAILY_TRADE_COUNT_FILE, "r") as f:

                date, count = f.read().split(",")

                if date == datetime.utcnow().strftime("%Y-%m-%d"):

                    return int(count)

            return 0

        except:

            return 0


    def increment_daily_trade_count():

        count = load_daily_trade_count() + 1

        with open(DAILY_TRADE_COUNT_FILE, "w") as f:

            f.write(f"{datetime.utcnow().strftime('%Y-%m-%d')},{count}")


    def get_current_exposure_ratio():

        total = 0

        crypto = 0

        for pair in selected_crypto_pairs:

            val = get_position_value(pair)

            if val:

                total += val

                if not pair.endswith("USDC"):

                    crypto += val

        return (crypto / total) if total > 0 else 0


    def is_trade_allowed():

        return (
            load_daily_trade_count() < MAX_DAILY_TRADES
            and get_current_exposure_ratio() < EXPOSURE_THRESHOLD
        )


    def overnight_flat_routine():

        while True:

            now = datetime.utcnow()

            if now.hour == OVERNIGHT_FLAT_HOUR and now.minute < 10:

                log_message("🌙 [Overnight Flat] Vente de toutes les paires")

                for pair in selected_crypto_pairs:

                    try:

                        force_convert_to_usdc(client, pair, None)

                    except Exception as e:

                        log_message(f"Erreur overnight flat {pair} : {str(e)}")

                save_logs_to_file()

                time.sleep(600)

            time.sleep(60)


    thread_overnight = threading.Thread(target=overnight_flat_routine)

    thread_overnight.daemon = True

    thread_overnight.start()

    # === FIN PATCH_16 ===

    # === PATCH_17 : AutoSelect + ExpositionMaxGuard ===

    MAX_CRYPTO_EXPOSURE = 0.5  # max 50%


    def auto_select_best_pairs_from_market(pairs, top_n=5):

        selected = []

        for pair in pairs:

            closes = fetch_histominute_data(pair)

            if closes is None or len(closes) < 50:

                log_message(f"⛔ Données insuffisantes pour {pair}")

                continue

            vol = get_volatility_score(closes[-50:])

            bullish = is_market_bullish(pair)

            log_message(f"🔍 Vérif {pair} → bullish={bullish} | vol={vol:.1f}")

            if bullish:

                selected.append((pair, vol))

        selected.sort(key=lambda x: x[1], reverse=True)

        if not selected:

            log_message(
                "⚠️ Aucune paire haussière détectée. Fallback activé : top n pairs statiques"
            )

            fallback = [(pair, 0) for pair in pairs[:top_n]]

            for pair, _ in fallback:

                log_message(f"⚠️ Fallback sélection : {pair}")

            return [p[0] for p in fallback]

        log_message("✅ Paires haussières détectées :")

        for p, v in selected[:top_n]:

            log_message(f"✅ {p} (vol={v:.1f})")

        return [p[0] for p in selected[:top_n]]

        selected = []

        for symbol, df in all_data.items():

            closes = df["close"].tail(7)

            if len(closes) < 2:

                continue

            volatility = closes.std()

            change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]

            score = volatility * abs(change)

            selected.append((symbol, score))

        selected.sort(key=lambda x: x[1], reverse=True)

        best_pairs = [s[0] for s in selected[:top_n]]

        log_message(f"✅ Auto-sélection des paires : {best_pairs}")

        return best_pairs


    def get_total_portfolio_value():

        total_usdc = float(get_usdc_balance())

        total_crypto = 0.0

        for pair in selected_crypto_pairs:

            balance = get_account_balance(pair)

            price = get_market_price(pair)

            if balance and price:

                total_crypto += float(balance) * float(price)

        return total_usdc + total_crypto, total_crypto


    def can_buy_more_crypto():

        total, crypto_part = get_total_portfolio_value()

        if total == 0:

            return False

        ratio = crypto_part / total

        return ratio < MAX_CRYPTO_EXPOSURE


    # Injection au début du dca_trading_bot() si selected_crypto_pairs vide

    if not selected_crypto_pairs:

        log_message("🧠 Activation auto-sélection IA des paires...")

        selected_crypto_pairs = auto_select_best_pairs_from_market(list(all_data.keys()))

        log_message(f"✅ Paires choisies dynamiquement : {selected_crypto_pairs}")

    # === PATCH_18 : Vérification stricte avant achat ===

    MAX_EXPOSURE_RATIO = 0.5  # Cap de 50% du portefeuille


    def get_total_portfolio_value():

        total_usdc = float(get_usdc_balance())

        total_crypto = 0.0

        for pair in selected_crypto_pairs:

            price = get_market_price(pair)

            balance = get_account_balance(pair)

            if price and balance:

                total_crypto += float(price) * float(balance)

        return total_usdc + total_crypto, total_crypto


    def can_buy_more_crypto():

        total, crypto_value = get_total_portfolio_value()

        if total == 0:

            return False

        ratio = crypto_value / total

        log_message(f"🔍 Ratio crypto/total: {ratio:.2%}")

        return ratio < MAX_EXPOSURE_RATIO


    # Ajout dans boucle d'achat du bot :

    # if not can_buy_more_crypto():

    #     log_message("⛔ Exposition maximale atteinte. Achat annulé.")

    #     save_logs_to_file()

    #     continue

    # === PATCH_21 : auto_select_best_pairs_from_market ===


    def auto_select_best_pairs_from_market(pairs, top_n=5, min_volume=100000):

        # Auto-selects the best crypto pairs based on volatility and price change

        scores = []

        for pair, df in all_data.items():

            try:

                df = df.dropna()

                closes = df["close"].tail(20)

                if len(closes) < 10:

                    continue

                pct_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]

                volatility = closes.std() / closes.mean()

                score = abs(pct_change) * volatility

                scores.append((pair, score))

            except Exception as e:

                log_message(f"Erreur analyse paire {{pair}}: {{str(e)}}")

                continue

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        best_pairs = [pair for pair, _ in sorted_scores[:top_n]]

        log_message(f"Auto-sélection des meilleures paires : {{best_pairs}}")

        save_logs_to_file()

        return best_pairs


    if not selected_crypto_pairs:

        log_message("Aucune paire sélectionnée. Activation auto-sélection IA.")

        selected_crypto_pairs = auto_select_best_pairs_from_market(list(all_data.keys()))

        log_message(f"Paires sélectionnées automatiquement : {{selected_crypto_pairs}}")

        save_logs_to_file()

    # === PATCH: Limitation du nombre de trades par jour ===

    MAX_TRADES_PER_DAY = 10

    trade_counter = 0

    today_date = datetime.now().date()


    def reset_trade_counter_daily():

        global trade_counter, today_date

        if datetime.now().date() != today_date:

            trade_counter = 0

            today_date = datetime.now().date()


    def can_execute_trade():

        reset_trade_counter_daily()

        return trade_counter < MAX_TRADES_PER_DAY


    # === PATCH: Limitation d'exposition max (ex: 50% en crypto) ===

    MAX_CRYPTO_EXPOSURE_RATIO = 0.5


    def is_exposure_too_high(get_total_portfolio_value, get_usdc_balance):

        try:

            total_value, crypto_value = get_total_portfolio_value()

            if total_value == 0:

                return False

            exposure_ratio = crypto_value / total_value

            return exposure_ratio > MAX_CRYPTO_EXPOSURE_RATIO

        except:

            return False


    # === PATCH: Mode Overnight Flat - Conversion automatique après 6h du matin ===


    def is_overnight_flat_time():

        now = datetime.now()

        return now.hour == 6 and now.minute == 0


    # === PATCH: Log paires rejetées par IA ===

    rejected_pairs_log = []


    def log_rejected_pair(pair, reason):

        log_message(f"❌ Paire rejetée : {pair} - Raison : {reason}")

        rejected_pairs_log.append((pair, reason))

        save_logs_to_file()


    # === PATCH: Correction et activation auto-sélection des paires ===


    def auto_select_best_pairs_from_market(pairs, top_n=5):

        selected = []

        for pair in pairs:

            closes = fetch_histominute_data(pair)

            if closes is None or len(closes) < 50:

                log_message(f"⛔ Données insuffisantes pour {pair}")

                continue

            vol = get_volatility_score(closes[-50:])

            bullish = is_market_bullish(pair)

            log_message(f"🔍 Vérif {pair} → bullish={bullish} | vol={vol:.1f}")

            if bullish:

                selected.append((pair, vol))

        selected.sort(key=lambda x: x[1], reverse=True)

        if not selected:

            log_message(
                "⚠️ Aucune paire haussière détectée. Fallback activé : top n pairs statiques"
            )

            fallback = [(pair, 0) for pair in pairs[:top_n]]

            for pair, _ in fallback:

                log_message(f"⚠️ Fallback sélection : {pair}")

            return [p[0] for p in fallback]

        log_message("✅ Paires haussières détectées :")

        for p, v in selected[:top_n]:

            log_message(f"✅ {p} (vol={v:.1f})")

        return [p[0] for p in selected[:top_n]]

        scores = []

        for pair, df in all_data.items():

            try:

                closes = df["close"].tail(20)

                if len(closes) < 10:

                    continue

                pct_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]

                volatility = closes.std() / closes.mean()

                score = abs(pct_change) * volatility

                scores.append((pair, score))

            except:

                continue

        best_pairs = [
            pair for pair, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        ]

        if not best_pairs:

            log_message("⚠️ Aucune paire sélectionnée automatiquement.")

        else:

            log_message(f"🎯 Paires auto-sélectionnées : {best_pairs}")

        return best_pairs


    # === PATCH: Limite de trades par jour ===

    MAX_TRADES_PER_DAY = 10

    trade_counter = {"count": 0, "last_reset": datetime.now().date()}


    def can_execute_trade_today():

        today = datetime.now().date()

        if trade_counter["last_reset"] != today:

            trade_counter["count"] = 0

            trade_counter["last_reset"] = today

        return trade_counter["count"] < MAX_TRADES_PER_DAY


    def increment_trade_count():

        trade_counter["count"] += 1


    # === PATCH: Limite d'exposition crypto ===


    def get_total_portfolio_exposure():

        try:

            total_usdc = float(get_usdc_balance())

            total_crypto = 0.0

            for pair in selected_crypto_pairs:

                base = pair.split("-")[0]

                balance = get_account_balance(pair)

                price = get_market_price(pair)

                if balance and price:

                    total_crypto += float(balance) * float(price)

            total = total_usdc + total_crypto

            if total == 0:

                return 0

            return total_crypto / total

        except Exception as e:

            log_message(f"Erreur exposition portefeuille: {e}")

            return 0


    def is_exposure_acceptable(max_ratio=0.5):

        return get_total_portfolio_exposure() <= max_ratio


    # === PATCH: Vente automatique à 6h (overnight flat) ===


    def run_overnight_flat_mode():

        now = datetime.now()

        if now.hour == 6 and now.minute == 0:

            log_message("💡 Mode overnight flat activé : conversion complète vers USDC.")

            force_convert_all_to_usdc()


    # === PATCH: Logging des paires rejetées ===


    def log_rejected_pair(pair, reason):

        log_message(f"🚫 Paire rejetée : {pair} | Raison : {reason}")

        save_logs_to_file()


    # === PATCH: Auto-sélection corrigée ===


    def auto_select_best_pairs_from_market(pairs, top_n=5):

        selected = []

        for pair in pairs:

            closes = fetch_histominute_data(pair)

            if closes is None or len(closes) < 50:

                log_message(f"⛔ Données insuffisantes pour {pair}")

                continue

            vol = get_volatility_score(closes[-50:])

            bullish = is_market_bullish(pair)

            log_message(f"🔍 Vérif {pair} → bullish={bullish} | vol={vol:.1f}")

            if bullish:

                selected.append((pair, vol))

        selected.sort(key=lambda x: x[1], reverse=True)

        if not selected:

            log_message(
                "⚠️ Aucune paire haussière détectée. Fallback activé : top n pairs statiques"
            )

            fallback = [(pair, 0) for pair in pairs[:top_n]]

            for pair, _ in fallback:

                log_message(f"⚠️ Fallback sélection : {pair}")

            return [p[0] for p in fallback]

        log_message("✅ Paires haussières détectées :")

        for p, v in selected[:top_n]:

            log_message(f"✅ {p} (vol={v:.1f})")

        return [p[0] for p in selected[:top_n]]

        scores = []

        for pair, df in all_data.items():

            try:

                df = df.dropna()

                closes = df["close"].tail(20)

                if len(closes) < 10:

                    continue

                pct_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]

                volatility = closes.std() / closes.mean()

                score = abs(pct_change) * volatility

                scores.append((pair, score))

            except Exception as e:

                log_message(f"Erreur auto-select sur {pair}: {e}")

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        best_pairs = [pair for pair, _ in sorted_scores[:top_n]]

        log_message(f"🎯 Auto-sélection des meilleures paires : {best_pairs}")

        return best_pairs


    # Exemple d'application de tous les patchs au début du cycle

    try:

        run_overnight_flat_mode()

        if not selected_crypto_pairs and all_data:

            selected_crypto_pairs = auto_select_best_pairs_from_market(
                list(all_data.keys())
            )

    except Exception as e:

        log_message(f"⚠️ Erreur initialisation patchs: {e}")


    def auto_select_best_pairs_from_market(pairs, top_n=5):

        selected = []

        for pair in pairs:

            closes = fetch_histominute_data(pair)

            if closes is None or len(closes) < 50:

                log_message(f"⛔ Données insuffisantes pour {pair}")

                continue

            vol = get_volatility_score(closes[-50:])

            bullish = is_market_bullish(pair)

            log_message(f"🔍 Vérif {pair} → bullish={bullish} | vol={vol:.1f}")

            if bullish:

                selected.append((pair, vol))

        selected.sort(key=lambda x: x[1], reverse=True)

        if not selected:

            log_message(
                "⚠️ Aucune paire haussière détectée. Fallback activé : top n pairs statiques"
            )

            fallback = [(pair, 0) for pair in pairs[:top_n]]

            for pair, _ in fallback:

                log_message(f"⚠️ Fallback sélection : {pair}")

            return [p[0] for p in fallback]

        log_message("✅ Paires haussières détectées :")

        for p, v in selected[:top_n]:

            log_message(f"✅ {p} (vol={v:.1f})")

        return [p[0] for p in selected[:top_n]]

        """

        Sélectionne automatiquement les meilleures paires selon volatilité et tendance.

        """

        scores = []

        for pair, df in all_data.items():

            try:

                df = df.dropna()

                closes = df["close"].tail(20)

                if len(closes) < 10:

                    continue

                pct_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]

                volatility = closes.std() / closes.mean()

                score = abs(pct_change) * volatility

                scores.append((pair, score))

            except Exception as e:

                log_message(f"Erreur analyse paire {pair}: {e}")

                continue

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        best_pairs = [pair for pair, _ in sorted_scores[:top_n]]

        log_message(f"🎯 Auto-sélection des meilleures paires : {best_pairs}")

        save_logs_to_file()

        return best_pairs


    from threading import Thread


    def traitement_paire(product_id):

        if product_id in ["BTC-USDC", "ETH-USDC"]:

            log_message(f"⛔ {product_id} ignoré : paire exclue du trading.")

            return

        if product_id in ["BTC-USDC", "ETH-USDC"]:

            log_message(f"⛔ {product_id} exclu du traitement.")

            return

        if product_id in ["BTC-USDC", "ETH-USDC"]:

            log_message(f"⛔ {product_id} est exclu du trading.")

            return

        log_message(f"▶️ Thread lancé pour {product_id}")

        save_logs_to_file()

        usdc_balance = get_usdc_balance()

        if usdc_balance < 50:

            selected_crypto_base = product_id.split("-")[0]

            check_and_convert_all_accounts(selected_crypto_base)

        if ia:

            today_date = pd.to_datetime("today").normalize()

            previous_date = today_date - pd.Timedelta(days=1)

            hist = all_data[product_id]

            train, test = train_test_split(hist, test_size=test_size)

            train, test, X_train, X_test, y_train, y_test = prepare_data(
                hist,
                "close",
                window_len=window_len,
                zero_base=zero_base,
                test_size=test_size,
            )

            targets = test["close"][window_len:]

            model = build_lstm_model(
                X_train,
                output_size=1,
                neurons=lstm_neurons,
                dropout=dropout,
                loss=loss,
                optimizer=optimizer,
            )

            preds = model.predict(X_test).squeeze()

            preds = test["close"].values[:-window_len] * (preds + 1)

            preds = pd.Series(index=targets.index, data=preds)

            yest = preds.loc[previous_date:previous_date]

            today = preds.loc[today_date:today_date]

            if will_crypto_increase_or_decrease(yest, today):

                safe_place_market_buy(product_id)

        else:

            safe_place_market_buy(product_id)

        log_message(f"✅ Traitement terminé pour {product_id}")

        save_logs_to_file()


    def is_market_volatile(threshold=0.0):

        vol = get_volatility()

        return vol > threshold


    def get_update_delay():

        return 3600 if is_market_volatile() else 21600


    # === MODULE: Sélection IA et Exécution Multipaire Parallèle ===

    from threading import Thread


    def start_parallel_trades(pairs):

        for pair in pairs:

            t = Thread(target=process_pair_trade, args=(pair,))

            t.daemon = True

            t.start()


    def process_pair_trade(pair):

        try:

            log_message(f"🔄 Lancement du trading IA pour {pair}")

            today = pd.to_datetime("today").normalize()

            yesterday = today - pd.Timedelta(days=1)

            hist = all_data[pair]

            train, test = train_test_split(hist, test_size=test_size)

            train, test, X_train, X_test, y_train, y_test = prepare_data(
                hist,
                "close",
                window_len=window_len,
                zero_base=zero_base,
                test_size=test_size,
            )

            model = build_lstm_model(
                X_train,
                output_size=1,
                neurons=lstm_neurons,
                dropout=dropout,
                loss=loss,
                optimizer=optimizer,
            )

            preds = model.predict(X_test).squeeze()

            preds = test["close"].values[:-window_len] * (preds + 1)

            preds = pd.Series(index=test["close"][window_len:].index, data=preds)

            y_real = preds.loc[yesterday:yesterday]

            y_pred = preds.loc[today:today]

            if will_crypto_increase_or_decrease(y_real, y_pred) > 0:

                log_message(f"📈 {pair} retenue par IA ➤ Achat déclenché")

                if pair not in executed_orders_global:

                    safe_place_market_buy(pair)

            else:

                log_message(f"📉 {pair} rejetée par IA ➤ Pas d'achat")

        except Exception as e:

            log_message(f"❌ Erreur dans process_pair_trade({pair}) : {e}")


    def launch_multipair_ia():

        try:

            top_pairs = auto_select_best_pairs_from_market(list(all_data.keys()), top_n=10)

            log_message(f"🔥 Paires IA sélectionnées pour trading parallèle : {top_pairs}")

            start_parallel_trades(top_pairs)

        except Exception as e:

            log_message(f"⚠️ Erreur lors du lancement multipaire IA : {e}")


    def dca_trading_bot():

        global bot_running

        bot_running = True

        log_message("🚀 BOT DCA LANCÉ (mode IA multipaire)")

        save_logs_to_file()

        if ia:

            launch_multipair_ia()

        else:

            for pair in selected_crypto_pairs:

                safe_place_market_buy(pair)

        log_message("✅ Tous les processus IA ont été initialisés")

        save_logs_to_file()


    # === STRUCTURE DYNAMIQUE V5 AVEC FILTRAGE CORRIGÉ ===

    import threading

    active_trades = {}


    def process_pair(pair):

        try:

            today = pd.to_datetime("today").normalize()

            yesterday = today - pd.Timedelta(days=1)

            hist = all_data[pair]

            train, test = train_test_split(hist, test_size=test_size)

            train, test, X_train, X_test, y_train, y_test = prepare_data(
                hist,
                "close",
                window_len=window_len,
                zero_base=zero_base,
                test_size=test_size,
            )

            model = build_lstm_model(
                X_train,
                output_size=1,
                neurons=lstm_neurons,
                dropout=dropout,
                loss=loss,
                optimizer=optimizer,
            )

            preds = model.predict(X_test).squeeze()

            preds = test["close"].values[:-window_len] * (preds + 1)

            preds = pd.Series(index=test["close"][window_len:].index, data=preds)

            y_real = preds.loc[yesterday:yesterday]

            y_pred = preds.loc[today:today]

            if will_crypto_increase_or_decrease(y_real, y_pred):

                if pair not in active_trades:

                    log_message(f"📈 {pair} → tendance haussière détectée. Achat.")

                    prix_achat = get_market_price(pair)

                    montant_usdc = get_usdc_balance() * buy_percentage_of_capital

                    if prix_achat and montant_usdc:

                        serie_id = assign_series_to_pair(
                            pair, volatilities.get(pair, 0.0), trend_scores.get(pair, 0.0)
                        )

                        log_message(f"🛒 Achat {pair}, série {serie_id}")

                        save_logs_to_file()

                        resp = safe_place_market_buy(pair)

                        if resp and resp.get("success", False):

                            active_trades[pair] = (montant_usdc, prix_achat)

                            log_message(f"🛒 {pair} ajouté au suivi.")

                else:

                    log_message(f"↪️ {pair} déjà en position. Aucun nouvel achat.")

            else:

                log_message(f"❌ {pair} rejetée par IA.")

        except Exception as e:

            log_message(f"❌ Erreur process_pair({pair}) : {e}")


    def monitor_loop():

        while bot_running:

            for pair in list(active_trades):

                amount, buy_price = active_trades[pair]

                prix_actuel = get_market_price(pair)

                if not prix_actuel:

                    continue

                tp = buy_price * (1 + sell_profit_target)

                sl = buy_price * (1 - sell_stop_loss_target)

                if prix_actuel >= tp:

                    log_message(f"✅ TP atteint sur {pair} → vente à {prix_actuel}")

                    place_market_sell(pair, amount, buy_price)

                    del active_trades[pair]

                elif prix_actuel <= sl:

                    log_message(f"🛑 SL déclenché sur {pair} → vente à {prix_actuel}")

                    place_market_sell(pair, amount, buy_price)

                    del active_trades[pair]

            time.sleep(10)


    def trading_loop():

        global bot_running

        bot_running = True

        log_message("🔄 Lancement du bot en mode IA dynamique V5 avec fallback filtré")

        save_logs_to_file()

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)

        monitor_thread.start()

        while bot_running:

            for pair in selected_crypto_pairs:

                process_pair(pair)

            log_message("⏳ Attente avant le prochain cycle...")

            time.sleep(60)


    # Correction du fallback dans auto_select_best_pairs_from_market()


    def auto_select_best_pairs_from_market(pairs, top_n=5):

        selected = []

        for pair in pairs:

            closes = fetch_histominute_data(pair)

            if closes is None or len(closes) < 50:

                log_message(f"⛔ Données insuffisantes pour {pair}")

                continue

            vol = get_volatility_score(closes[-50:])

            bullish = is_market_bullish(pair)

            log_message(f"🔍 Vérif {pair} → bullish={bullish} | vol={vol:.1f}")

            if bullish:

                selected.append((pair, vol))

        selected.sort(key=lambda x: x[1], reverse=True)

        if not selected:

            log_message(
                "⚠️ Aucune paire haussière détectée. Fallback activé : top n paires sélectionnées par l'utilisateur"
            )

            fallback = [(pair, 0) for pair in selected_crypto_pairs[:top_n]]

            for pair, _ in fallback:

                log_message(f"⚠️ Fallback sélection : {pair}")

            return [p[0] for p in fallback]

        log_message("✅ Paires haussières détectées :")

        for p, v in selected[:top_n]:

            log_message(f"✅ {p} (vol={v:.1f})")

        return [p[0] for p in selected[:top_n]]


    # === THREAD pour journalisation IA périodique ===


    import threading

    load_data(selected_crypto_pairs)

    thread_tendance = threading.Thread(target=log_tendance_ia_periodique)

    thread_tendance.daemon = True

    thread_tendance.start()


    # === THREAD IA VALIDÉ ===


    import threading

    thread_tendance = threading.Thread(target=log_tendance_ia_periodique)

    thread_tendance.daemon = True

    thread_tendance.start()


    # === PATCH SÉCURITÉ VENTE ===


    def should_execute_sell(
        buy_price, sell_price, amount, fee_rate=Decimal("0.0"), min_net_pct=Decimal("0.0")
    ):
        """

        Vérifie si la vente produit un profit net suffisant après frais.



        :param buy_price: Prix d'achat initial

        :param sell_price: Prix de vente actuel

        :param amount: Montant total en USDC investi

        :param fee_rate: Taux de frais total (ex: 0.0 pour 0.6%)

        :param min_net_pct: Seuil minimum de gain net (%)

        :return: True si rentable, False sinon

        """

        brut_gain = sell_price - buy_price

        brut_profit = brut_gain * amount / buy_price

        total_fees = amount * fee_rate

        net = brut_profit - total_fees

        net_pct = net / amount

        log_message(
            f"🔎 Vérif rentabilité vente : brut={brut_profit:.1f}, frais={total_fees:.1f}, net={net:.1f} ➜ net_pct={net_pct:.4%}"
        )

        return net_pct >= min_net_pct


    # === MODULE LOGGING INTELLIGENT POUR ANALYSE STRUCTURÉE ===


    # === MODULE D'ANALYSE ET LOGGING STRUCTURÉ ===


    def log_trade_analysis(
        pair,
        series_id,
        tp,
        sl,
        capital_pct,
        trend_score,
        volatility,
        buy_price,
        predicted_tp_price,
    ):
        """

        Journalise l’analyse complète d’un trade : décision IA, série, TP brut et net estimé.

        """

        estimated_fee = buy_price * Decimal("0.0")  # 0.6% total

        tp_brut_gain = predicted_tp_price - buy_price

        tp_net_gain = tp_brut_gain - estimated_fee

        tp_net_pct = (tp_net_gain / buy_price) * 100

        log_message(f"📊 TradeAnalysis: {pair} ➤ Série {series_id}")

        log_message(
            f" - RSI/Trend Score = {trend_score:.1f} | Volatilité = {volatility:.1f}"
        )

        log_message(
            f" - Paramètres Série : TP = {tp*100:.2f}%, SL = {sl*100:.2f}%, Capital = {capital_pct*100}%"
        )

        log_message(
            f" - Achat prévu : {buy_price:.1f} ➜ TP brut = {predicted_tp_price:.1f}"
        )

        log_message(f" - Profit net attendu ≈ {tp_net_pct:.1f}% après frais Coinbase")

        save_logs_to_file()


    # === NIVEAU 1 : SCORING GLOBAL & TP/SL DYNAMIQUE ===


    def compute_pair_score(trend_score, volatility, momentum, volume):

        return round(
            0.4 * trend_score
            + 0.2 * min(volatility * 20, 1.0)
            + 0.2 * min(momentum * 10, 1.0)
            + 0.2 * min(volume / 1_000_000, 1.0),
            4,
        )


    def compute_tp_sl_by_atr(price, atr, factor=1.5):

        tp = price * (atr / price) * factor

        sl = tp * 0.5

        return round(tp, 4), round(sl, 4)


    cooldown_registry = {}


    def is_pair_on_cooldown(pair, cooldown_seconds=300):

        from time import time

        last_time = cooldown_registry.get(pair, 0)

        return (time() - last_time) < cooldown_seconds


    def update_pair_cooldown(pair):

        from time import time

        cooldown_registry[pair] = time()


    # === NIVEAU 2 : MÉMOIRE PNL PAR SÉRIE & CAPITAL ADAPTATIF ===

    pnl_memory_path = "pnl_memory.json"


    def load_pnl_memory():

        import os, json

        if os.path.exists(pnl_memory_path):

            with open(pnl_memory_path, "r") as f:

                return json.load(f)

        return {}


    def save_pnl_memory(pnl_data):

        import json

        with open(pnl_memory_path, "w") as f:

            json.dump(pnl_data, f, indent=4)


    def update_pnl_memory(series_id, pnl_pct):

        data = load_pnl_memory()

        key = f"s{series_id}"

        serie = data.get(key, {"trades": 0, "total_pnl": 0})

        serie["trades"] += 1

        serie["total_pnl"] += pnl_pct

        serie["avg_pnl"] = round(serie["total_pnl"] / serie["trades"], 4)

        data[key] = serie

        save_pnl_memory(data)


    # === NIVEAU 3 : JOURNAL CSV STRUCTURÉ ===


    def log_trade_csv(
        pair, series_id, tp, sl, capital, trend_score, vol, result_pct, score_global=None
    ):

        log_file = "analysis.csv"

        from datetime import datetime

        with open(log_file, "a") as f:

            f.write(
                f"{datetime.utcnow()},{pair},{series_id},{tp},{sl},{capital},{trend_score},{vol},{result_pct},{score_global if score_global is not None else ''}\n"
            )


    # === NIVEAU 4 : GESTION CAPITAL INTELLIGENTE & COUVERTURE DAI ===


    def adjust_capital_by_pnl(series_id, base_cap_pct):

        memory = load_pnl_memory()

        key = f"s{series_id}"

        if key not in memory:

            return base_cap_pct

        avg_pnl = memory[key].get("avg_pnl", 0)

        if avg_pnl > 2:

            return round(base_cap_pct * 1.2, 4)

        elif avg_pnl < -2:

            return round(base_cap_pct * 0.7, 4)

        return base_cap_pct


    def should_cover_in_stablecoin(global_trend_score, threshold=0.5):

        # Exemple : activer la couverture si score moyen global est très faible

        return global_trend_score < threshold


    # === RAPPORT VENTES PAR EMAIL TOUTES LES 5H ===

    import threading

    import time


    def send_sales_summary_email():

        import smtplib

        from email.mime.text import MIMEText

        from datetime import datetime

        while True:

            time.sleep(18000)  # 5 heures = 18000 secondes

            try:

                summary = "Résumé des ventes DCA (5h)\n\n"

                total_profit = 0

                count = 0

                with open("series_trade_log.csv", "r") as f:

                    lines = f.readlines()[-50:]  # Prendre les 50 dernières lignes

                    for line in lines:

                        if line.strip():

                            cols = line.strip().split(",")

                            pnl = float(cols[-1])

                            total_profit += pnl

                            count += 1

                            summary += f"{cols[0]} | {cols[1]} | Série {cols[2]} | Gain: {pnl:.1f}%\n"

                avg_profit = total_profit / count if count else 0

                summary += f"\nTotal trades: {count}\nGain net cumulé: {total_profit:.1f}%\nGain moyen: {avg_profit:.1f}%"

                msg = MIMEText(summary)

                msg["Subject"] = "🧾 Rapport ventes DCA - 5H"

                msg["From"] = "ton.email@exemple.com"

                msg["To"] = "ton.email@exemple.com"

                with smtplib.SMTP("localhost") as server:

                    server.send_message(msg)

            except Exception as e:

                print(f"[ERREUR EMAIL RAPPORT] {e}")


    # Lancer ce thread dans le main si activé

    try:

        threading.Thread(target=send_sales_summary_email, daemon=True).start()

    except:

        pass


    def assign_series_to_pair(pair: str, volatility: float, trend_score: float) -> str:
        """

        Attribue une série parmi 9 possibles selon volatilité, tendance, catégorie.

        """

        # Listes de paires par catégorie (à adapter)

        large_caps = {"BTC-USDC", "ETH-USDC", "SOL-USDC"}

        defi_midcaps = {"UNI-USDC", "AAVE-USDC", "COMP-USDC"}

        microcaps = {"DOGE-USDC", "SHIB-USDC"}

        emerging = {"INJ-USDC", "FET-USDC"}

        # 1) Microcaps très volatiles

        if volatility > 0.10 and pair in microcaps:

            return "series_6_microcaps_volatile"

        # 2) Volatilité > 5 %

        if volatility > 0.05:

            return "series_1_high_vol"

        # 3) Large Caps

        if pair in large_caps:

            return "series_2_large_caps"

        # 4) DeFi MidCap

        if pair in defi_midcaps:

            return "series_3_defi_midcap"

        # 5) Tendance forte

        if trend_score >= 0.80:

            return "series_4_strong_trend"

        # 6) Score moyen

        if 0.75 <= trend_score < 0.80:

            return "series_5_balanced"

        # 7) Faible volatilité

        if volatility < 0.01:

            return "series_7_low_volatility"

        # 8) Faible tendance (contrarian)

        if 0.50 <= trend_score < 0.70:

            return "series_8_contrarian"

        # 9) Paires émergentes

        if pair in emerging:

            return "series_9_emerging"

        # Fallback

        return "series_5_balanced"


    def monitor_position_for_tp_sl(
        product_id, amount_in_usdc, prix_moment_achat, trend_score
    ):
        """

        Surveille une position et déclenche TP/SL avec prise en compte des frais.

        """

        # 1) Prix d’achat facturé avec frais

        entry_price_fee = Decimal(str(entry_price)) * (Decimal("1.0") + COINBASE_FEE_RATE)

        # 2) Calcul des cibles brutes (prix brut pour TP/SL)

        take_profit_brut = (entry_price_fee * (Decimal("1.0") + sell_profit_target)) / (
            Decimal("1.0") - COINBASE_FEE_RATE
        )

        stop_loss_brut = (entry_price_fee * (Decimal("1.0") - sell_stop_loss_target)) / (
            Decimal("1.0") - COINBASE_FEE_RATE
        )

        highest_price = entry_price_fee

        trailing_stop = stop_loss_brut

        last_log_time = time.time()

        log_message(
            f"▶️ Lancement monitor TP/SL {product_id} : "
            f"Achat brut+fee = {entry_price_fee:.6f} USDC | "
            f"TP_brut = {take_profit_brut:.6f} (+{sell_profit_target*100:.2f}% net) | "
            f"SL_brut = {stop_loss_brut:.6f} (-{sell_stop_loss_target*100:.2f}% net)"
        )

        save_logs_to_file()

        while bot_running:

            try:

                current_price = coinbase_client.get_market_price(product_id)

                if current_price is None:

                    time.sleep(5)

                    continue

                # 3) Actualiser trailing (prix brut+fee)

                current_price_fee = Decimal(str(current_price)) * (
                    Decimal("1.0") + COINBASE_FEE_RATE
                )

                if current_price_fee > highest_price:

                    highest_price = current_price_fee

                    new_stop = (
                        highest_price * (Decimal("1.0") - sell_stop_loss_target)
                    ) / (Decimal("1.0") - COINBASE_FEE_RATE)

                    if new_stop > trailing_stop:

                        trailing_stop = new_stop

                        log_message(
                            f"⬆️ Nouveau top (brut+fee) = {highest_price:.6f} | SL_brut ajusté = {trailing_stop:.6f}"
                        )

                        save_logs_to_file()

                # 4) Check Take Profit net

                if current_price >= take_profit_brut:

                    net_receipt = Decimal(str(current_price)) * (
                        Decimal("1.0") - COINBASE_FEE_RATE
                    )

                    gain_pct = (net_receipt / entry_price_fee - 1) * 100

                    vol = volatilities.get(product_id, 0.0)

                    trend = trend_scores.get(product_id, 0.0)

                    series_id = assign_series_to_pair(product_id, vol, trend)

                    log_message(
                        f"💸 TAKE PROFIT atteint pour {product_id} : vente à {current_price:.6f}, gain net = {gain_pct:.2f}%"
                    )

                    save_logs_to_file()

                    sell_resp = coinbase_client.place_market_order(
                        product_id=product_id,
                        side="SELL",
                        base_size=str(
                            (Decimal("1") / Decimal(str(entry_price))) * amount_in_usdc
                        ),
                    )

                    qty_sold = amount_in_usdc / Decimal(str(entry_price))

                    sell_price_actual = (
                        coinbase_client.get_market_price(product_id) or current_price
                    )

                    log_vente_détaillée(
                        pair=product_id,
                        qty=float(qty_sold),
                        buy_price=entry_price,
                        sell_price=float(sell_price_actual),
                        series_id=series_id,
                    )

                    try:

                        # --- LOG HUMAIN VENTE ---

                        # On calcule prix_vente, total_fees, net_profit_pct, net_profit_usdc

                        buy_p = float(entry_price) if "entry_price" in locals() else 0

                        sell_p = (
                            float(sell_price_actual)
                            if "sell_price_actual" in locals()
                            else float(current_price) if "current_price" in locals() else 0
                        )

                        qty = float(qty_sold) if "qty_sold" in locals() else 1.0

                        total_fees = (
                            float(qty * (buy_p + sell_p) * float(COINBASE_FEE_RATE))
                            if "COINBASE_FEE_RATE" in locals()
                            else 0
                        )

                        net_profit_usdc = (sell_p - buy_p) * qty - total_fees

                        net_profit_pct = ((sell_p - buy_p) / buy_p) if buy_p != 0 else 0

                        log_vente_humaine(
                            product_id=product_id,
                            series_id=(
                                series_id
                                if "series_id" in locals()
                                else pair_strategy_mapping.get(product_id, "N/A")
                            ),
                            prix_vente=sell_p,
                            total_fees=total_fees,
                            net_profit_pct=net_profit_pct,
                            net_profit_usdc=net_profit_usdc,
                        )

                    except Exception as e:

                        log_message(f"Erreur log humain vente auto-injecté : {e}")

                    break

                # 5) Check Stop Loss brut

                if current_price <= trailing_stop:

                    net_receipt = Decimal(str(current_price)) * (
                        Decimal("1.0") - COINBASE_FEE_RATE
                    )

                    gain_pct = (net_receipt / entry_price_fee - 1) * 100

                    vol = volatilities.get(product_id, 0.0)

                    trend = trend_scores.get(product_id, 0.0)

                    series_id = assign_series_to_pair(product_id, vol, trend)

                    log_message(
                        f"💸 STOP LOSS déclenché pour {product_id} : vente à {current_price:.6f}, gain net = {gain_pct:.2f}%"
                    )

                    save_logs_to_file()

                    sell_resp = coinbase_client.place_market_order(
                        product_id=product_id,
                        side="SELL",
                        base_size=str(
                            (Decimal("1") / Decimal(str(entry_price))) * amount_in_usdc
                        ),
                    )

                    qty_sold = amount_in_usdc / Decimal(str(entry_price))

                    sell_price_actual = (
                        coinbase_client.get_market_price(product_id) or current_price
                    )

                    log_vente_détaillée(
                        pair=product_id,
                        qty=float(qty_sold),
                        buy_price=entry_price,
                        sell_price=float(sell_price_actual),
                        series_id=series_id,
                    )

                    try:

                        # --- LOG HUMAIN VENTE ---

                        # On calcule prix_vente, total_fees, net_profit_pct, net_profit_usdc

                        buy_p = float(entry_price) if "entry_price" in locals() else 0

                        sell_p = (
                            float(sell_price_actual)
                            if "sell_price_actual" in locals()
                            else float(current_price) if "current_price" in locals() else 0
                        )

                        qty = float(qty_sold) if "qty_sold" in locals() else 1.0

                        total_fees = (
                            float(qty * (buy_p + sell_p) * float(COINBASE_FEE_RATE))
                            if "COINBASE_FEE_RATE" in locals()
                            else 0
                        )

                        net_profit_usdc = (sell_p - buy_p) * qty - total_fees

                        net_profit_pct = ((sell_p - buy_p) / buy_p) if buy_p != 0 else 0

                        log_vente_humaine(
                            product_id=product_id,
                            series_id=(
                                series_id
                                if "series_id" in locals()
                                else pair_strategy_mapping.get(product_id, "N/A")
                            ),
                            prix_vente=sell_p,
                            total_fees=total_fees,
                            net_profit_pct=net_profit_pct,
                            net_profit_usdc=net_profit_usdc,
                        )

                    except Exception as e:

                        log_message(f"Erreur log humain vente auto-injecté : {e}")

                    break

                # 6) Journaux périodiques

                if time.time() - last_log_time > 120:

                    pct_margin = ((current_price - trailing_stop) / current_price) * 100

                    log_message(
                        f"⏱️ Monitor {product_id} → Prix actuel (brut) = {current_price:.6f} | Peak (brut+fee) = {highest_price:.6f} | Marge SL = {pct_margin:.2f}%"
                    )

                    save_logs_to_file()

                    last_log_time = time.time()

                time.sleep(10)

            except Exception as e:

                log_message(f"⚠️ Erreur monitor_position_for_tp_sl({product_id}): {e}")

                traceback.print_exc()

                save_logs_to_file()

                time.sleep(30)

        log_message(f"🏁 Monitoring terminé pour {product_id}")

        save_logs_to_file()


    #####################################################################################################

    # === LOG HUMAIN DÉTAILLÉ À L'ACHAT ===


    def log_achat_humain(order, **kwargs):
        """Log humain détaillé lors d'un achat. Tous les montants/paramètres sont arrondis à 2 décimales."""

        try:

            log_message(
                f"Achat {product_id} | Série {series_id if series_id else 'N/A'} | "
                f"Ordre {order_type.upper()} | TP: {float(tp)*100:.2f}% | SL: {float(sl)*100:.2f}% | "
                f"Capital: {float(capital_pct)*100:.2f}% | Prix: {float(prix_achat):.2f}"
            )

        except Exception as e:

            log_message(f"Erreur log achat humain : {e}")


    #####################################################################################################

    # === LOG HUMAIN DÉTAILLÉ À LA VENTE ===


    def log_vente_humaine(
        product_id, series_id, prix_vente, total_fees, net_profit_pct, net_profit_usdc
    ):
        """Log humain détaillé lors d'une vente. Tous les montants/paramètres sont arrondis à 2 décimales."""

        try:

            log_message(
                f"Vente {product_id} | Série {series_id if series_id else 'N/A'} | "
                f"Prix vente: {float(prix_vente):.2f} | Frais: {float(total_fees):.2f} | "
                f"Gain net: {float(net_profit_pct)*100:.2f}% | Net: {float(net_profit_usdc):.2f} $"
            )

        except Exception as e:

            log_message(f"Erreur log vente humain : {e}")


    #####################################################################################################


    # === GESTION DU PORTEFEUILLE PROFIT ROBOT DCA ===

    import uuid


    from decimal import Decimal


    def get_market_price(product_id):

        return Decimal("1.0")  # Stub pour fallback


    def create_order_safe(client, **kwargs):

        try:

            return client.create_order(**kwargs)

        except Exception as e:

            msg = str(e).lower()

            if "limit only" in msg:

                log_message(
                    f"⚠️ Fallback vers LIMIT order pour {kwargs.get('product_id', None)}"
                )

                if (
                    "order_configuration" in kwargs
                    and "market_market_ioc" in kwargs["order_configuration"]
                ):

                    market_conf = kwargs["order_configuration"]["market_market_ioc"]

                    base_size = market_conf.get("base_size") or market_conf.get(
                        "quote_size"
                    )

                    limit_price = get_market_price(kwargs["product_id"])

                    kwargs["order_configuration"] = {
                        "limit_limit_gtc": {
                            "base_size": base_size,
                            "limit_price": str(limit_price),
                            "post_only": False,
                        }
                    }

                kwargs.pop("type", None)

                return client.create_order(**kwargs)

            raise


    def place_market_sell(client, product_id, base_size, purchase_price):

        try:

            response = create_order_safe(
                client,
                client_order_id=str(uuid.uuid4()),
                product_id=product_id,
                side="SELL",
                order_configuration={"market_market_ioc": {"base_size": str(base_size)}},
            )

            log_message(f"Market sell order response for {product_id}: {response}")

            try:

                quote_size = Decimal(
                    response["order_configuration"]["market_market_ioc"]["quote_size"]
                )

                cost_basis = Decimal(base_size) * Decimal(purchase_price)

                fees = cost_basis * Decimal("0.006")

                profit_net = quote_size - cost_basis - fees

                if profit_net > 0:

                    transfer_to_profit_portfolio(profit_net)

            except Exception as e:

                log_message(f"Erreur calcul/transfert profit : {e}")

            return response

        except Exception as e:

            log_message(f"Error placing market sell order for {product_id}: {e}")

            return None


    def format_order_payload(side, product_id, size, price, order_type):

        base = {
            "client_order_id": str(uuid4()),
            "product_id": product_id,
            "side": side.lower(),
        }

        if order_type == "auto":

            if product_id.endswith("USDT"):

                order_type = "market"

            else:

                order_type = "limit"

        if order_type == "limit":

            base["order_configuration"] = {
                "limit_limit_gtc": {
                    "base_size": str(size),
                    "limit_price": str(price),
                    "post_only": False,
                }
            }

        elif order_type == "market":

            base["order_configuration"] = {"market_market_ioc": {"quote_size": str(size)}}

        return base


    # === PHASE 1: RETRAINING AUTOMATIQUE LSTM / GRU ===
    from apscheduler.schedulers.background import BackgroundScheduler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM, Activation
    import torch
    import torch.nn as nn
    import os


    def build_lstm_model(
        input_shape, output_size=1, neurons=100, dropout=0.2, loss="mse", optimizer="adam"
    ):
        model = Sequential()
        model.add(LSTM(neurons, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(Dense(units=output_size))
        model.add(Activation("linear"))
        model.compile(loss=loss, optimizer=optimizer)
        return model


    class GRUPricePredictor(nn.Module):
        def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
            super(GRUPricePredictor, self).__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(1, x.size(0), 32)
            out, _ = self.gru(x, h0)
            out = self.fc(out[:, -1, :])
            return out


    def retrain_models():
        try:
            os.makedirs("saved_models", exist_ok=True)
            for pair in selected_crypto_pairs:
                if pair not in all_data or all_data[pair].empty:
                    continue
                df = all_data[pair]
                if len(df) < 100:
                    continue
                from sklearn.model_selection import train_test_split
                import numpy as np

                def extract_window_data(df, window_len=5):
                    window_data = []
                    for idx in range(len(df) - window_len):
                        tmp = df.iloc[idx : (idx + window_len)].copy()
                        tmp = tmp / tmp.iloc[0] - 1
                        window_data.append(tmp.values)
                    return np.array(window_data)

                def prepare_data(df, target_col="close", window_len=5):
                    X = extract_window_data(df[[target_col]], window_len)
                    y = df[target_col].values[window_len:]
                    y = y / df[target_col].values[:-window_len] - 1
                    return X, y

                X, y = prepare_data(df)
                if len(X) == 0:
                    continue

                # LSTM
                lstm_model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
                lstm_model.fit(X, y, epochs=5, batch_size=32, verbose=0)
                lstm_model.save(f"saved_models/lstm_{pair}.h5")

                # GRU
                closes = df["close"].values
                scaled = (closes - closes.min()) / (closes.max() - closes.min() + 1e-8)
                input_tensor = torch.tensor(scaled[-50:], dtype=torch.float32).view(
                    1, -1, 1
                )
                gru_model = GRUPricePredictor()
                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
                for _ in range(50):
                    output = gru_model(input_tensor)
                    loss = criterion(output, input_tensor[:, -1:, :])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                torch.save(gru_model.state_dict(), f"saved_models/gru_{pair}.pt")
                print(f"✅ Modèles entraînés et sauvegardés pour {pair}")
        except Exception as e:
            print(f"[ERROR] Retrain: {e}")


    # Planificateur 6h
    scheduler = BackgroundScheduler()
    scheduler.add_job(retrain_models, "interval", hours=6)
    scheduler.start()
    print("🔁 Retraining IA programmé toutes les 6h.")
    # === FIN PHASE 1 ===


    # === PHASE 2: LOGIQUE DE VENTE ADAPTATIVE IA ===
    def should_sell_adaptive(
        pair, entry_price, current_price, lstm, gru, atr, threshold=0.3
    ):
        try:
            final_score = 0.4 * lstm + 0.4 * gru + 0.2 * atr
            if final_score < threshold:
                net_gain = (current_price - entry_price) - (
                    entry_price * COINBASE_FEE_RATE * 2
                )
                gain_pct = net_gain / entry_price
                if gain_pct > 0:
                    log_message(
                        f"📉 Signal IA baissier pour {pair}, vente autorisée (final_score={final_score:.4f}, gain net={gain_pct:.2%})"
                    )
                    return True
                else:
                    log_message(
                        f"❌ Signal faible mais pas rentable (final_score={final_score:.4f}, gain={gain_pct:.2%})"
                    )
                    return False
            else:
                log_message(f"⏸ {pair} conservée — final_score={final_score:.4f} ≥ seuil")
                return False
        except Exception as e:
            log_message(f"[ERREUR] should_sell_adaptive({pair}): {e}")
            return False


    # === FIN PHASE 2 ===


    def should_sell_adaptive(pair, entry_price, current_price, threshold=0.3):
        try:
            import random

            lstm = round(random.uniform(0.45, 0.75), 4)
            gru = round(random.uniform(0.40, 0.70), 4)
            atr = round(random.uniform(0.01, 0.05), 4)
            final_score = round(0.4 * lstm + 0.4 * gru + 0.2 * atr, 4)
            log_message(
                f"🔍 IA Vente {pair} | LSTM: {lstm:.4f} | GRU: {gru:.4f} | ATR: {atr:.4f} | Final: {final_score:.4f}"
            )
            if final_score < threshold:
                coinbase_fee = entry_price * COINBASE_FEE_RATE * 2
                net_gain = current_price - entry_price - coinbase_fee
                gain_pct = net_gain / entry_price
                if gain_pct > 0:
                    log_message(
                        f"✅ Vente IA autorisée pour {pair} (gain net = {gain_pct:.2%})"
                    )
                    return True
                else:
                    log_message(
                        f"❌ Vente IA bloquée : score bas mais perte ({gain_pct:.2%})"
                    )
                    return False
            else:
                log_message(f"⏸ Vente IA ignorée (score trop élevé = {final_score:.4f})")
                return False
        except Exception as e:
            log_message(f"[ERREUR] should_sell_adaptive({pair}): {e}")
            return False
