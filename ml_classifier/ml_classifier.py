from data import DataInjestor
from features import Features
from labels import Labels


SYMBOL = "ES=F"
START_DATE = "2010-01-01"
END_DATE = None


df = DataInjestor.get(SYMBOL, START_DATE, END_DATE)
df_ta = Features().add_indicators(df)
df_ta = Labels().compute_all_lables(df_ta)
test_data = Features().training_data(df_ta)

x=1
