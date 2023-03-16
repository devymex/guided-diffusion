import sys, pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1])
df = df.drop(columns=['samples', 'step', 'param_norm'])
# df['param_norm'] /= 10000

df = df.rolling(100).sum()
fig, ax = plt.subplots()
df.plot(logy=True, logx=False, ax=ax, grid=True)

for line in ax.get_lines():
    line.set_linewidth(1)

plt.show()
