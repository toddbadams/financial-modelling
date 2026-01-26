import numpy as np
import pandas as pd


class Labels:

    @staticmethod
    def generate_label(
        data: pd.DataFrame,
        lookahead: int = 5,
        thresh: float = 0.01,
        col: str = "Close"
    ) -> pd.Series:
        """
        Label each row from the mean of the *next* `lookahead` closes:
          2 : future_mean >= current * (1 + thresh)
          1 : future_mean <= current * (1 - thresh)
          0 : otherwise
        """
        future_mean = (
            data[col]
            .shift(-lookahead)
            .rolling(window=lookahead, min_periods=lookahead)
            .mean()
        )
        pct_change = (future_mean - data[col]) / data[col]

        labels = np.select(
            [pct_change >= thresh, pct_change <= -thresh],
            [2, 1],
            default=0
        )

        return pd.Series(labels, index=data.index)

    def compute_all_lables(self, df_ta: pd.DataFrame) -> pd.DataFrame:
        lookaheads = [2, 4, 6, 8, 10]
        thresholds = [0.01, 0.02]

        for la in lookaheads:
            for th in thresholds:
                df_ta[f"label_la{la}_th{th:.3f}"] = self.generate_label(df_ta, lookahead=la, thresh=th)
        
        df_ta.dropna(inplace=True)
        return df_ta
