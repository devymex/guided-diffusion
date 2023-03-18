import sys, os, pandas as pd
import matplotlib.pyplot as plt

input_file = sys.argv[1]

smooth_window_size = 100
line_width = 0.5
plot_size = [15, 10]
saving_dpi = 200
saving_format = '.png'
showing = False
param_norm_divisor = None or 10000

df = pd.read_csv(input_file)
df = df.drop(columns=['samples', 'step'])
if not param_norm_divisor:
    df = df.drop(columns=['param_norm'])
else:
    df['param_norm'] /= param_norm_divisor

df = df.rolling(smooth_window_size).sum()
fig, ax = plt.subplots()
fig.set_figwidth(plot_size[0])
fig.set_figheight(plot_size[1])
df.plot(logy=True, logx=False, ax=ax, grid=True)

for line in ax.get_lines():
    line.set_linewidth(line_width)

output_file = os.path.splitext(input_file)[0] + saving_format
plt.savefig(fname=output_file, dpi=saving_dpi, bbox_inches='tight')
print(f'A figure saved to "{output_file}"')

if showing:
    plt.show()
