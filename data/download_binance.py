# download_binance.py
# Скрипт для загрузки исторических часовых баров с Binance
# API_KEY    = os.getenv("BINANCE_API_KEY") or "ofs9AjAxHy1NWU0oXIgqxvthBiBTamFzL5RHTCh029iTlETvyWEJK2VkCs3x3fp"
# API_SECRET = os.getenv("BINANCE_API_SECRET") or "wmzdq6ba2kmGzcGDhkqcta7KzmRdAudXVqLVO32AgtwgfLXkhferQdKaUNgqfewz"

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_binance.py

Скрипт для загрузки исторических часовых баров с Binance
и сохранения их в корне проекта в data/binance.
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_binance.py

Скрипт для загрузки исторических часовых баров с Binance
Сохраняет CSV в <PROJECT_ROOT>/data/binance независимо от cwd.
"""

import os
import pandas as pd
from binance.client import Client
from binance.enums import KLINE_INTERVAL_1HOUR

# 1) Поддержка fallback: если env vars не заданы, взять из констант
DEFAULT_API_KEY    = "ofs9AjAxHy1NWU0oXIgqxvthBiBTamFzL5RHTCh029iTlETvyWEJK2VkCs3x3fp"
DEFAULT_API_SECRET = "wmzdq6ba2kmGzcGDhkqcta7KzmRdAudXVqLVO32AgtwgfLXkhferQdKaUNgqfewz"

API_KEY    = os.getenv("BINANCE_API_KEY") or DEFAULT_API_KEY
API_SECRET = os.getenv("BINANCE_API_SECRET") or DEFAULT_API_SECRET
if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
    print("⚠️ Warning: using fallback API_KEY/API_SECRET constants.")

# 2) Определяем корень проекта
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))      # .../qc-auto-strategy/data
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))  # .../qc-auto-strategy

# 3) Настройки загрузки
SYMBOLS    = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
INTERVAL   = KLINE_INTERVAL_1HOUR
START_DATE = "1 Jan 2017"
END_DATE   = "1 Jan 2025"

# 4) Папка для результата: <PROJECT_ROOT>/data/binance
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "binance")
os.makedirs(OUT_DIR, exist_ok=True)

# 5) Инициализируем клиент Binance
client = Client(API_KEY, API_SECRET)

for symbol in SYMBOLS:
    print(f"⏳ Downloading {symbol} from {START_DATE} to {END_DATE}…")
    klines = client.get_historical_klines(
        symbol,
        INTERVAL,
        START_DATE,
        END_DATE
    )
    # Формируем DataFrame и приводим типы
    df = pd.DataFrame(
        klines,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]
    )
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df.set_index("open_time", inplace=True)
    df = df[["open","high","low","close","volume"]]

    # Сохраняем CSV в правильную папку
    out_path = os.path.join(OUT_DIR, f"{symbol}_1h.csv")
    df.to_csv(out_path)
    print(f"✅ Saved {symbol}: {len(df)} rows → {out_path}")
