import pandas as pd
from data import DataInjestor

from features import Features
from labels import Labels
from test_data import TestData
from train import BaselineTrain


SYMBOL = "ES=F"
START_DATE = "2010-01-01"
END_DATE = None


df = DataInjestor.get(SYMBOL, START_DATE)
df = Features().add_indicators(df)
df = Labels().compute_all_lables(df)
test_data = TestData.from_df(df)
t = BaselineTrain()
results = t.run(test_data)
print(pd.DataFrame(results).sort_values("accuracy", ascending=False))
t.hyper_param_tunning(test_data, results)
x = 1
