import numpy as np
import matplotlib.pyplot as plt

def difference(data, order=1):
    return np.diff(data, n=order)

def predict_arima(train, p, q, d):
    # Step 1: Differencing the series (I)
    differenced = difference(train, d)
    # Step 2: (AR), prediction based on p lagged values
    ar_terms = np.array([differenced[-p:]])
    # Step 3: (MA), using q lagged errors for correction
    errors = np.zeros(q)
    forecast = np.dot(ar_terms, np.ones(p))
    return forecast


# Example data to be processed by ARIMA
data = np.random.normal(50, 5, 100) # 50 is the mean, 5 is the standard deviation, 100 is the number of data points

# Hyperparameters for ARIMA(p,d,q)
p, d, q = 2, 1, 1
prediction = predict_arima(data[:80], p, q, d)
print("Next predicted value: ", prediction)

# This works
# But the outputed value is the valuation, not the value
# Because the data is stationary already

# Plot the data (y is from 0)
plt.plot(data)
# Make the y-axis start from 0
plt.ylim(0, max(data))
plt.show()
