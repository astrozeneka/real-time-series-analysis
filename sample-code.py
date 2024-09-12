import matplotlib.pyplot as plt
import time

x_data, y_data = [], []

plt.ion()  # Turn on interactive mode

fig, ax = plt.subplots()

for i in range(100):
    x_data.append(i)
    y_data.append(i * 2)  # Simulating some data
    ax.clear()  # Clear the previous plot
    ax.plot(x_data, y_data)
    plt.draw()  # Update the plot
    plt.pause(0.1)  # Pause for a brief moment (simulate latency)
    time.sleep(0.1)  # Simulated latency

plt.ioff()  # Turn off interactive mode
plt.show()