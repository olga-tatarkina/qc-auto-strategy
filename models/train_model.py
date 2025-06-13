# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 22:54:56 2025

@author: user
"""

import numpy as np
# Восстанавливаем устаревшие alias np.float и np.int для совместимости со sklearn
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Определяем корень проекта один раз
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # models/
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))  # корень проекта


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df['return'] = df['close'].pct_change()
    df['target'] = (df['return'].shift(-1) > 0).astype(int)

    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['vol_ma5'] = df['volume'].rolling(5).mean()

    df = df.dropna()
    return df


def load_all_data() -> pd.DataFrame:
    # Папка с данными
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    # Рекурсивный поиск CSV с часовыми данными
    pattern = os.path.join(data_dir, '**', '*_1h.csv')
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No files found under {data_dir} matching '*_1h.csv'")
    dfs = []
    for f in files:
        symbol = os.path.basename(f).split('_')[0]
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df['symbol'] = symbol
        dfs.append(df)
    return pd.concat(dfs)


def main():
    # Загрузка данных
    data = load_all_data()
    data = create_features(data)

    # Фичи и метка
    X = data[['ma5', 'ma10', 'rsi', 'vol_ma5']]
    y = data['target']

    # Разбиение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Обучение
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)

    # Оценка
    print("=== Classification Report ===")
    print(classification_report(y_test, model.predict(X_test)))

    # Сохранение модели
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'binance_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
