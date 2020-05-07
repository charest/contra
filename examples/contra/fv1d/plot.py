import matplotlib.pyplot as plt
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1], sep=" ")

plt.subplot(2, 2, 1)
plt.plot(df['x'], df['d'], label='density')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(df['x'], df['v'], label='velocity')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(df['x'], df['p'], label='pressure')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(df['x'], df['e'], label='energy')
plt.legend()

plt.show()
