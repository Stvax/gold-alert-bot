"""
Gold A+ Breakout Signal Bot
----------------------------
Standalone script (no TradingView needed) that checks gold (GC=F) 15-minute
price data for a previous-day-high/low breakout with volume, trend, and
momentum confirmation, and sends a Telegram message when the confluence
score crosses the configured threshold.

Designed to be run periodically (e.g. every 15 minutes) by a scheduler
such as GitHub Actions, cron, or Task Scheduler. Each run only checks the
most recently completed candle, so no state needs to be saved between runs.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ============ CONFIG (mirrors the TradingView Pine Script settings) ============
TICKER = "GC=F"            # Gold futures. Alternative: "XAUUSD=X" for a forex-style spot ticker
INTERVAL = "15m"           # Must match how often this script is scheduled to run
VOL_LOOKBACK = 20
VOL_MULTIPLIER = 2.0
MIN_BODY_PCT = 50
EMA_FAST = 9
EMA_SLOW = 50
RSI_LENGTH = 14
RSI_LOW = 40
RSI_HIGH = 65
ATR_LENGTH = 14
ATR_STOP_MULT = 1.5
RR_RATIO = 2.0
MIN_CONFIDENCE = 80

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    if resp.status_code != 200:
        print(f"Telegram send failed: {resp.status_code} {resp.text}")


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, length):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def main():
    # --- Previous day high/low ---
    daily = yf.download(TICKER, period="10d", interval="1d", progress=False)
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    if len(daily) < 2:
        print("Not enough daily data returned - try again shortly.")
        return
    pdh = float(daily["High"].iloc[-2])
    pdl = float(daily["Low"].iloc[-2])

    # --- Intraday data ---
    intraday = yf.download(TICKER, period="5d", interval=INTERVAL, progress=False)
    if isinstance(intraday.columns, pd.MultiIndex):
        intraday.columns = intraday.columns.get_level_values(0)
    if len(intraday) < EMA_SLOW + 5:
        print("Not enough intraday data returned - try again shortly.")
        return

    df = intraday.copy()
    df["ema_fast"] = ema(df["Close"], EMA_FAST)
    df["ema_slow"] = ema(df["Close"], EMA_SLOW)
    df["rsi"] = rsi(df["Close"], RSI_LENGTH)
    df["atr"] = atr(df, ATR_LENGTH)
    df["avg_vol"] = df["Volume"].rolling(VOL_LOOKBACK).mean()
    df["vol_ratio"] = df["Volume"] / df["avg_vol"]
    df["body_pct"] = (df["Close"] - df["Open"]).abs() / (df["High"] - df["Low"]).replace(0, np.nan) * 100

    # Use the last FULLY completed candle, not the one still forming
    last = df.iloc[-2]
    prev = df.iloc[-3]

    long_breakout = prev["Close"] <= pdh and last["Close"] > pdh
    short_breakout = prev["Close"] >= pdl and last["Close"] < pdl

    bull_trend = last["ema_fast"] > last["ema_slow"]
    bear_trend = last["ema_fast"] < last["ema_slow"]
    rsi_ok_long = RSI_LOW < last["rsi"] < 80
    rsi_ok_short = 20 < last["rsi"] < RSI_HIGH

    def score(breakout, trend_ok, rsi_ok):
        pts = 0
        pts += 20 if breakout else 0
        pts += 20 if last["vol_ratio"] >= VOL_MULTIPLIER else 0
        pts += 20 if last["body_pct"] >= MIN_BODY_PCT else 0
        pts += 20 if trend_ok else 0
        pts += 20 if rsi_ok else 0
        return pts

    long_score = score(long_breakout, bull_trend, rsi_ok_long)
    short_score = score(short_breakout, bear_trend, rsi_ok_short)

    close = float(last["Close"])
    atr_val = float(last["atr"])
    candle_dt = last.name
    if candle_dt.tzinfo is None:
        candle_dt = candle_dt.tz_localize("UTC")
    candle_dt_az = candle_dt.tz_convert("America/Phoenix")
    candle_time = candle_dt_az.strftime("%Y-%m-%d %I:%M %p %Z")

    if long_breakout and long_score >= MIN_CONFIDENCE:
        stop = close - atr_val * ATR_STOP_MULT
        target = close + atr_val * ATR_STOP_MULT * RR_RATIO
        msg = (
            f"\U0001F7E2 GOLD LONG breakout\n"
            f"Candle: {candle_time}\n"
            f"Entry: {close:.2f}\nStop: {stop:.2f}\nTarget: {target:.2f}\n"
            f"Confidence: {long_score}%\nRSI: {last['rsi']:.1f}"
        )
        send_telegram(msg)
        print(msg)

    elif short_breakout and short_score >= MIN_CONFIDENCE:
        stop = close + atr_val * ATR_STOP_MULT
        target = close - atr_val * ATR_STOP_MULT * RR_RATIO
        msg = (
            f"\U0001F534 GOLD SHORT breakout\n"
            f"Candle: {candle_time}\n"
            f"Entry: {close:.2f}\nStop: {stop:.2f}\nTarget: {target:.2f}\n"
            f"Confidence: {short_score}%\nRSI: {last['rsi']:.1f}"
        )
        send_telegram(msg)
        print(msg)

    else:
        print(f"No signal this run. Long score: {long_score}%, Short score: {short_score}%, RSI: {last['rsi']:.1f}")


if __name__ == "__main__":
    main()
