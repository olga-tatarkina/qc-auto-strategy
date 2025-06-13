import joblib
import pandas as pd

model = joblib.load('models/binance_model.joblib')
sample = pd.DataFrame({
    'ma5':    [1.0],
    'ma10':   [0.9],
    'rsi':    [50.0],
    'vol_ma5': [1000.0],
})
print(model.predict(sample), model.predict_proba(sample))
