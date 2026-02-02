import typing
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(".cache") / "financial_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class DataInjestor:

    @staticmethod
    def get(
        symbols: str | list[str], start_date: str, end_date: typing.Optional[str] = None
    ) -> pd.DataFrame:
        symbol_key = symbols if isinstance(symbols, str) else "_".join(sorted(symbols))
        cache_name = "_".join(
            filter(
                None,
                (
                    symbol_key,
                    start_date,
                    end_date or "latest",
                ),
            )
        ).replace(":", "-")
        cache_path = CACHE_DIR / f"{cache_name}.parquet"

        if cache_path.exists():
            return pd.read_parquet(cache_path)

        df: pd.DataFrame | None = yf.download(symbols, start=start_date, end=end_date)

        if df is None:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            )

        # flatten multi-level columns (this is for the recent yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.to_parquet(cache_path, index=True)

        return df
