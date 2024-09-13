
# Including the needed packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
    StreamAnalysis class is used to analyze a data stream in real-time.
"""
class StreamAnalysis:
    # The window size for calculating the percentile (used for outlier detection)
    PERCENTILE_WINDOW_SIZE = 20

    # The window size for calculating the moving average (used for concept drift detection)
    MOVING_AVERAGE_WINDOW_SIZE = 15

    def __init__(self):
        self.data = []
        # Initialize the plot, plt.ion() is used to make the plot dynamic
        plt.ion()
        self.fig, self.ax = plt.subplots()

    """
        Append a new value to the data stream
    """
    def append(self, value):
        # The data is a list of [value, is_outlier, drift(None, 'up', 'down')]
        self.data.append([value, False, None])
        self.analyze()

    """
        Perform seasonality analysis on the data
        This function will output a plot of the autocorrelation values
        allowing to visually detect the seasonality of the data
    """
    def seasonality_analysis(self, data, index):
        # trying to detect the seasonality between period of 1 and 300
        lags = np.arange(1, 300)
        # Calculate the autocorrelation values
        acv_values = [pd.Series(data).autocorr(lag=lag) for lag in lags]
        # Plot the autocorrelation values for analysis
        fig = plt.figure()
        plt.plot(lags, acv_values)
        plt.title("Seasonality Analysis")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation Value")
        fig.savefig(f'seasonality-analysis-output/seasonality-{index}.png')
        print("    Saved file seasonality-analysis-output/seasonality-{index}.png")

    """
        Analyze the data stream
    """
    def analyze(self):
        d = np.array(self.data) # Convert the data to a numpy array

        # ======== STEP 1. DETECT OUTLIERS ========
        # For detecting outliers, 3 methods were compared (z-score, percentile, derivative + percentile)
        # Due to the nature of the data, the derivative + percentile method is the most suitable

        # Method 1: z-score (not suitable)
        # Check if the latest point is an outlier by using a sliding window of PERCENTILE_WINDOW_SIZE
        # if len(d) > self.PERCENTILE_WINDOW_SIZE:
        #    sub_d = d[-self.PERCENTILE_WINDOW_SIZE:]
        #    mean = np.mean(sub_d[:, 0])
        #    std_dev = np.std(sub_d[:, 0])
        #    z_score = (d[-1][0] - mean) / std_dev
        #    if abs(z_score) > 2:
        #        self.data[-1][1] = True # Flag the point as an outlier

        # Method 2: calculating percentile (not suitable)
        # if len(d) > self.PERCENTILE_WINDOW_SIZE:
        #    FACTOR = 1.2
        #    sub_d = d[-self.PERCENTILE_WINDOW_SIZE:]
        #    q1 = np.percentile(sub_d[:, 0], 25)
        #    q3 = np.percentile(sub_d[:, 0], 75)
        #    iqr = q3 - q1
        #    lower_bound = q1 - FACTOR * iqr
        #    upper_bound = q3 + FACTOR * iqr
        #    if d[-1][0] < lower_bound or d[-1][0] > upper_bound:
        #        self.data[-1][1] = True

        # Method 3: calculating derivative then percentile (selected one)
        if len(d) > self.PERCENTILE_WINDOW_SIZE:
            sub_d = d[-self.PERCENTILE_WINDOW_SIZE:]
            sub_d = sub_d[:, 0]
            sub_d_diff = np.diff(sub_d)
            q1 = np.percentile(sub_d_diff, 25)
            q3 = np.percentile(sub_d_diff, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            if sub_d_diff[-1] < lower_bound or sub_d_diff[-1] > upper_bound:
                self.data[-1][1] = True


        # ======== STEP 2. DETECT CONCEPT DRIFT ========
        if len(d) > self.MOVING_AVERAGE_WINDOW_SIZE:
            # Calculate the moving average all the way to the latest point
            # The outliers are filtered out for better accuracy
            d_filtered = np.array([x[0] for x in d[:-1] if not x[1]])
            # Calculate the derivative
            d_diff = np.diff(d_filtered)
            # Calculate the moving average
            sma = np.convolve(d_diff, np.ones(self.MOVING_AVERAGE_WINDOW_SIZE), 'valid') / self.MOVING_AVERAGE_WINDOW_SIZE

            if len(sma) > 2:
                if sma[-1] > 0 and sma[-2] < 0: # The moving average
                    print("Concept drift detected at t=", len(d))
                    self.data[-1][2] = 'up'
                if sma[-1] < 0 and sma[-2] > 0: # The moving average
                    print("Concept drift detected at t=", len(d))
                    self.data[-1][2] = 'down'


        # ======== STEP 3. ANALYZE SEASONALITY ========
        if len(d) % 128 == 0:
            print("Performing seasonality analysis at t=", len(d))
            self.seasonality_analysis(d[:, 0], len(d))

        # ======== STEP 4. UPDATE THE PLOT ========
        self.ax.clear()
        # Plot only the 200 latest points for smooth visualization
        d_range = range(len(d) - 200, len(d)) if len(d) > 200 else range(len(d))
        for i in d_range:
            if d[i][1]:
                # The outlier are scattered in red
                self.ax.scatter(i, d[i][0], c='red', s=3)
            else:
                # The normal data are scattered in blue
                self.ax.scatter(i, d[i][0], c='b', s=1)
            if d[i][2] == 'up':
                # Plot symbol indicating where the concept drift is detected
                plt.text(i, d[i][0], '⬆️', fontsize=16, color='green')
            if d[i][2] == 'down':
                # Plot symbol indicating where the concept drift is detected
                plt.text(i, d[i][0], '⬇️', fontsize=16, color='red')

            # Set the x-axis limit for better visualization
            self.ax.set_xlim(
                max(0, len(d) - 200),
                max(200, len(d))
            )
            self.ax.set_title("Data Stream Analysis")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Value")
        # Simulate latency of the data stream
        plt.pause(0.1)

import argparse
# parse args
parser = argparse.ArgumentParser(description='Stream Analysis')
parser.add_argument('--std-in-mode', action='store_true', help='Use standard input to feed data')
args = parser.parse_args()

if __name__ == "__main__":

    sa = StreamAnalysis()

    if not args.std_in_mode: # This mode will allow the script to generate random data by itself

        # Generate a random walk for SEASONAL and NOISY data
        n = 1000                # The total number of floating-point values
        seasonal_period = 100   # The simulated seasonality period
        offset = 100            # The starting point of the series
        # Generate series using a random-walk algorithm
        random_walk = np.cumsum(np.random.randn(n))
        time = np.arange(n)
        # The usage of sin function allow to simulate the seasonality
        seasonal_component = 10 * np.sin(2 * np.pi * time / seasonal_period)
        # Combine all components
        seasonal_random_walk = random_walk + seasonal_component + offset

        # The percentage of outliers
        outlier_freq = 0.075
        # Generate random indices
        outlier_indices = np.random.randint(0, n, int(n * outlier_freq))
        # Generate random magnitudes for each
        outlier_magnitudes = np.random.uniform(-20, 20, int(n * outlier_freq))
        # Apply to the series
        for i in range(len(outlier_indices)):
            seasonal_random_walk[outlier_indices[i]] += outlier_magnitudes[i]


        for val in seasonal_random_walk:
            # Append a value to the data stream
            sa.append(val)


    if args.std_in_mode: # This mode will allow the script to read data from standard input
        while True:
            try:
                val = float(input())
                sa.append(val)
            except EOFError:
                break
            except ValueError:
                print("Invalid input, skipping...")
                continue
