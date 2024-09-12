# Auto Regresive Integrated Moving Average
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import adf_check

df = pd.read_csv('data/df.csv', parse_dates=True, index_col='date')
df = pd.DataFrame(df.groupby(df.index.strftime('%Y-%m')).sum()['amount'])
df.columns = ['Value']

# Plot
# plt.plot(df)
# plt.show()

# Trying to make the data stationary
df_testing = pd.DataFrame(np.log(df.Value).diff().diff(12))
adf_check(df_testing.Value.dropna())

