# region imports
from AlgorithmImports import *
import joblib
import pandas as pd
# endregion

class QCAuto(QCAlgorithm):

    def initialize(self):
        """Initialize the algorithm and load the ML model."""
        self.set_start_date(2018, 5, 1)
        self.set_end_date(2018, 5, 5)
        self.set_cash(100000)

        self.symbol = self.add_crypto("BTCUSDT", Resolution.HOUR, Market.BINANCE).symbol

        self.ma5 = self.SMA(self.symbol, 5, Resolution.HOUR)
        self.ma10 = self.SMA(self.symbol, 10, Resolution.HOUR)
        self.rsi = self.RSI(self.symbol, 14, MovingAverageType.WILDERS, Resolution.HOUR)
        self.vol_window = RollingWindow 

        self.set_warm_up(timedelta(days=10))

        self.model = joblib.load("binance_model.joblib")

    def on_data(self, data: Slice):
        if self.is_warming_up or not data.contains_key(self.symbol):
            return

        bar = data[self.symbol]
        self.vol_window.add(bar.volume)

        if len(self.vol_window) < 5:
            return

        df = pd.DataFrame({
            "ma5": [self.ma5.current.value],
            "ma10": [self.ma10.current.value],
            "rsi": [self.rsi.current.value],
            "volume": [list(self.vol_window)[-1]]
        })

        prob = self.model.predict_proba(df)[0][1]
        self.debug(f"Predicted prob: {prob:.2f}")

        if not self.portfolio.invested and prob > 0.7:
            self.set_holdings(self.symbol, 1)
        elif self.portfolio.invested and prob < 0.5:
            self.liquidate(self.symbol)
