import numpy as np
import matplotlib.pyplot as plt
from utils import adf_check
import pandas as pd

def difference(data, order=1):
    return np.diff(data, n=order)

n = 1000
seasonal_period = 100
offset = 100

random_walk = np.cumsum(np.random.randn(n))
time = np.arange(n)
seasonal_component = 10 * np.sin(2 * np.pi * time / seasonal_period)
seasonal_random_walk = random_walk + seasonal_component + offset


seasonal_random_walk_df = pd.DataFrame(seasonal_random_walk)
# The column name will be 'value'
seasonal_random_walk_df.columns = ['value']

# Plot the seasonal random walk
#plt.plot(seasonal_random_walk_df)
#plt.show()

# Step 1. Prepare ADF test on this plain seasonal random walk
# adf_check(seasonal_random_walk_df.value) # Yes normally this data is non-stationary

# Step 2. Compute the differencing, then do afk test again
seasonal_random_walk_diff = difference(seasonal_random_walk_df.value, order=1)
adf_check(seasonal_random_walk_diff) # Data should be stationary here