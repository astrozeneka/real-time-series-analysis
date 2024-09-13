
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class StreamAnalysis:
    SLIDING_WINDOW_SIZE = 20
    data = []

    def __init__(self):
        self.data = []
        plt.ion()
        self.fig, self.ax = plt.subplots()

    def append(self, value):
        # The data is a list of [value, is_outlier]
        self.data.append([value, False])
        if len(self.data) < self.SLIDING_WINDOW_SIZE:
            return
        self.analyze()
    
    def analyze(self):

        self.ax.clear()
        d = np.array(self.data)

        # METHOD 1: z-score
        # Check if the latest point is an outlier by using a sliding window of SLIDING_WINDOW_SIZE
        #if len(d) > self.SLIDING_WINDOW_SIZE:
        #    sub_d = d[-self.SLIDING_WINDOW_SIZE:]
        #    mean = np.mean(sub_d[:, 0])
        #    std_dev = np.std(sub_d[:, 0])
        #    z_score = (d[-1][0] - mean) / std_dev
        #    if abs(z_score) > 2:
        #        self.data[-1][1] = True # Flag the point as an outlier

        # METHOD 2: calculating percentile
        #if len(d) > self.SLIDING_WINDOW_SIZE:
        #    FACTOR = 1.2
        #    sub_d = d[-self.SLIDING_WINDOW_SIZE:]
        #    q1 = np.percentile(sub_d[:, 0], 25)
        #    q3 = np.percentile(sub_d[:, 0], 75)
        #    iqr = q3 - q1
        #    lower_bound = q1 - FACTOR * iqr
        #    upper_bound = q3 + FACTOR * iqr
        #    if d[-1][0] < lower_bound or d[-1][0] > upper_bound:
        #        self.data[-1][1] = True

        # METHOD 3: calculating derivative then percentile (selected one)
        #if len(d) > self.SLIDING_WINDOW_SIZE:
        #    sub_d = d[-self.SLIDING_WINDOW_SIZE:]
        #    sub_d = sub_d[:, 0]
        #    sub_d_diff = np.diff(sub_d)
        #    q1 = np.percentile(sub_d_diff, 25)
        #    q3 = np.percentile(sub_d_diff, 75)
        #    iqr = q3 - q1
        #    lower_bound = q1 - 1.5 * iqr
        #    upper_bound = q3 + 1.5 * iqr
        #    if sub_d_diff[-1] < lower_bound or sub_d_diff[-1] > upper_bound:
        #        self.data[-1][1] = True

        # Plot the data
        for i in range(len(d)):
            if d[i][1]:
                # The outlier are scattered in red
                self.ax.scatter(i, d[i][0], c='red', s=3)
            else:
                # The normal data are scattered in blue
                self.ax.scatter(i, d[i][0], c='b', s=1)

        # Simulates the latency (can be removed later)
        plt.pause(0.1)
        

if __name__ == "__main__":
    # Step 1. Generate a random walk for SEASONAL and NOISY data
    n = 1000
    seasonal_period = 100
    offset = 100
    random_walk = np.cumsum(np.random.randn(n))
    time = np.arange(n)
    seasonal_component = 10 * np.sin(2 * np.pi * time / seasonal_period)
    seasonal_random_walk = random_walk + seasonal_component + offset
    
    # Simulating outlisers
    outlier_freq = 0.1
    outlier_indices = np.random.randint(0, n, int(n * outlier_freq))
    outlier_magnitudes = np.random.uniform(-30, 30, int(n * outlier_freq))
    for i in range(len(outlier_indices)):
        seasonal_random_walk[outlier_indices[i]] += outlier_magnitudes[i]

    # Plot scatter plot (dot size is 1)
    #plt.scatter(time, seasonal_random_walk, s=1)
    #plt.show()

    # Step 2. Feed the data to the StreamAnalysis class
    sa = StreamAnalysis()
    for val in seasonal_random_walk:
        sa.append(val)
