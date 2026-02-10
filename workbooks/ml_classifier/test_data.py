import pandas as pd


from dataclasses import dataclass


@dataclass(frozen=True)
class TestData:
    train: pd.DataFrame
    test: pd.DataFrame
    validation: pd.DataFrame
    feature_cols: list[str]
    label_cols: list[str]

    @property
    def train_features(self) -> pd.DataFrame:
        return self.train[self.feature_cols]

    @property
    def train_lables(self) -> pd.DataFrame:
        return self.train[self.label_cols]

    @property
    def test_features(self) -> pd.DataFrame:
        return self.test[self.feature_cols]

    @property
    def test_lables(self) -> pd.DataFrame:
        return self.test[self.label_cols]

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "TestData":
        TRAIN_PCT = 0.6
        TEST_PCT = 0.2
        train_index = int(len(df) * TRAIN_PCT)
        test_index = int(len(df) * (TEST_PCT + TRAIN_PCT))
        train_df, test_df = df.iloc[:train_index], df.iloc[train_index:test_index]
        val_df = df.iloc[test_index:]
        feature_cols = [
            c
            for c in df.columns
            if c not in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            and not c.startswith("label_")
        ]
        label_cols = [c for c in df.columns if c.startswith("label_")]
        return cls(
            train=train_df,
            test=test_df,
            validation=val_df,
            feature_cols=feature_cols,
            label_cols=label_cols,
        )
