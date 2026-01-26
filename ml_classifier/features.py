from dataclasses import dataclass
import pandas as pd
import pandas_ta as ta


@dataclass(frozen=True)
class TestData:
    train: pd.DataFrame
    test: pd.DataFrame
    validation: pd.DataFrame
    feature_cols: list[str]

    @property
    def test_features(self) -> pd.DataFrame:
        return self.train[self.feature_cols]
    
class Features:

    @staticmethod
    def _merge(df_main, ta_obj):
        """Add TA output to df_main, regardless of Series/DataFrame type."""
        if isinstance(ta_obj, pd.DataFrame):
            return pd.concat([df_main, ta_obj], axis=1)
        else:
            return pd.concat([df_main, ta_obj.rename(ta_obj.name)], axis=1)

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # --- Momentum --- #
        for length in [5, 10, 15]:
            df[f"rsi_{length}"] = ta.rsi(df["Close"], length=length)
        # Rate of Change:  This indicator, also (confusingly) known as Momentum, is a pure oscillator that quantifies the percent change.
        df["roc_10"] = ta.roc(df["Close"], length=10)
        # Momentum:  This indicator attempts to quantify speed by using the differences over a bar length.
        df["mom_10"] = ta.mom(df["Close"], length=10)

        # --- Oscillators --- #
        # Stochastic RSI:  This indicator attempts to quantify RSI relative to its High-Low range.
        df = self._merge(df, ta.stochrsi(df["Close"]))
        # Commodity Channel Index: This indicator attempts to identify "overbought" and "oversold" levels relative to a mean.
        df["cci_20"] = ta.cci(df["High"], df["Low"], df["Close"], length=20)
        # William's Percent R: This indicator attempts to identify "overbought" and "oversold" conditions similar to the RSI.
        df["wr_14"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)
        # 'Know Sure Thing': This indicator, by Martin Pring, attempts to capture trends using a smoothed indicator of four different smoothed ROCs.
        df = self._merge(df, ta.kst(df["Close"]))
        # Moving Average Convergence Divergence: This indicator attempts to identify trends.
        df["macd"] = ta.macd(df["Close"])["MACD_12_26_9"]

        # --- Trend --- #
        for length in [5, 10, 20]:
            # Simple Moving Average: This indicator is the the textbook moving average, a rolling sum of values divided by the window period (or length).
            df[f"sma_{length}"] = ta.sma(df["Close"], length=length)
            # Exponential Moving Average: This Moving Average is more responsive than the Simple Moving Average (SMA).
            df[f"ema_{length}"] = ta.ema(df["Close"], length=length)
        # Volume Weighted Moving Average: Computes a weighted average using price and volume.    
        df["vwma_20"] = ta.vwma(df["Close"], df["Volume"], length=20)

        # --- Volatility --- #
        # Bollinger Bands: This indicator, by John Bollinger, attempts to quantify volatility by creating lower and upper bands centered around a moving average.
        df = self._merge(df, ta.bbands(df["Close"], length=20))
        # Average True Range: This indicator attempts to quantify volatility with a focus on gaps or limit moves.
        df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        df = self._merge(df, ta.kc(df["High"], df["Low"], df["Close"], length=20))

        # --- Volume --- #
        # On Balance Volume: This indicator attempts to quantify buying and selling pressure.
        df["obv"] = ta.obv(df["Close"], df["Volume"])
        # Accumulation/Distribution: This indicator attempts to quantify accumulation/distribution from a relative position within it's High-Low range and volume.
        df["ad"] = ta.ad(df["High"], df["Low"], df["Close"], df["Volume"])
        # Elder's Force Index: This indicator attempts to quantify movement magnitude as well as potential reversals and price corrections.
        df["efi"] = ta.efi(df["Close"], df["Volume"])
        # Negative Volume Index: This indicator attempts to identify where smart money is active.
        df["nvi"] = ta.nvi(df["Close"], df["Volume"])
        # Positive Volume Index: This indicator attempts to identify where smart money is active.
        df["pvi"] = ta.pvi(df["Close"], df["Volume"])

        return df

    def training_data(self, df_ta: pd.DataFrame) -> TestData:
        TRAIN_PCT = 0.6
        TEST_PCT = 0.2
        # validation is remaining data
        train_index = int(len(df_ta) * TRAIN_PCT)
        test_index = int(len(df_ta) * (TEST_PCT + TRAIN_PCT))
        train_df, test_df = df_ta.iloc[:train_index], df_ta.iloc[train_index:test_index]
        val_df = df_ta.iloc[test_index:]
        feature_cols = [c for c in df_ta.columns if c not in ["Open","High","Low","Close","Adj Close","Volume"] and not c.startswith("label_")]

        return TestData(train=train_df, test=test_df, validation=val_df, feature_cols=feature_cols)
