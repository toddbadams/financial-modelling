import pandas as pd
import yfinance as yf

class DataInjestor:

    @staticmethod
    def get(symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        df = yf.download(symbol, start=start_date, end=end_date)

        # flatten multi-level columns (this is for the recent yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df
