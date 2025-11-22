import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import mplfinance as mpf

from io import StringIO
from matplotlib.gridspec import GridSpec


# --- Calculate MACD ---
def compute_macd(df, short=12, long=26, signal=9):
    """Compute MACD Line, Signal Line, and Histogram"""
    df['EMA_short'] = df['close'].ewm(span=short, adjust=False).mean()
    df['EMA_long'] = df['close'].ewm(span=long, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']


def plot_side_by_side_v3(subset_qqq, subset_spy, save_dir, fig_name):
    def calculate_macd(df, slow=26, fast=12, signal=9):
        macd = pd.DataFrame(index=df.index)
        macd['fast_ema'] = df['close'].ewm(span=fast).mean()
        macd['slow_ema'] = df['close'].ewm(span=slow).mean()
        macd['macd'] = macd['fast_ema'] - macd['slow_ema']
        macd['signal'] = macd['macd'].ewm(span=signal).mean()
        macd['hist'] = macd['macd'] - macd['signal']
        return macd

    macd1 = calculate_macd(subset_spy)
    macd2 = calculate_macd(subset_qqq)

    # 2. Create figure and add subplots
    fig = mpf.figure(style='yahoo', figsize=(18, 9))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[3, 1, 1], width_ratios=[1, 1])

    # Create axes for each main plot and its corresponding volume plot
    ax1 = fig.add_subplot(gs[0, 0]) # 2 rows, 2 columns, 1st subplot (main plot 1)
    ax2 = fig.add_subplot(gs[0, 1]) # 2 rows, 2 columns, 2nd subplot (main plot 2)
    ax_macd1 = fig.add_subplot(gs[1, 0])
    ax_macd2 = fig.add_subplot(gs[1, 1])
    volume_ax1 = fig.add_subplot(gs[2, 0], sharex=ax1) # 2 rows, 2 columns, 3rd subplot (volume plot 1)
    volume_ax2 = fig.add_subplot(gs[2, 1], sharex=ax2) # 2 rows, 2 columns, 4th subplot (volume plot 2)

    # 3. Create addplots for each DataFrame, specifying the axis
    def get_addplots(subset, ax):
        addplots = [
            mpf.make_addplot(subset['MA10'], ax=ax, color='grey', width=1.0, linestyle='solid'),
            mpf.make_addplot(subset['MA20'], ax=ax, color='orange', width=1.0, linestyle='solid'),
            mpf.make_addplot(subset['VWAP'], ax=ax, color='purple', width=1.0, linestyle='dashed')
        ]
        return addplots

    addplots1 = get_addplots(subset_spy, ax=ax1)
    addplots2 = get_addplots(subset_qqq, ax=ax2)

    # Helper function to get macd plots
    def get_macd_plots(macd_df, ax_macd):
        colors = ['g' if v >= 0 else 'r' for v in macd_df['hist']]
        macd_plots = [
            mpf.make_addplot(macd_df['macd'], ax=ax_macd, color='fuchsia', ylabel='MACD'),
            mpf.make_addplot(macd_df['signal'], ax=ax_macd, color='b'),
            mpf.make_addplot(macd_df['hist'], ax=ax_macd, type='bar', color=colors)
        ]
        return macd_plots
    macd_plots1 = get_macd_plots(macd1, ax_macd1)
    macd_plots2 = get_macd_plots(macd2, ax_macd2)

    # 4. Define hlines for each plot
    def get_hlines(subset):
        open_range = subset.iloc[:5]
        open_range_high = open_range['high'].max()
        open_range_low = open_range['low'].min()
        hlines = dict(hlines=[open_range_high, open_range_low], colors=['red', 'green'], linestyle='dashed')
        return hlines

    hlines1 = get_hlines(subset_spy)
    hlines2 = get_hlines(subset_qqq)

    # 5. Plot each DataFrame on its own set of axes
    mpf.plot(subset_spy, type='candle', ax=ax1, volume=volume_ax1, addplot=addplots1 + macd_plots1, hlines=hlines1, axtitle=fig_name[:-4] + '_spy')
    mpf.plot(subset_qqq, type='candle', ax=ax2, volume=volume_ax2, addplot=addplots2 + macd_plots2, hlines=hlines2, axtitle=fig_name[:-4] + '_qqq')

    # Adjust layout for better display
    fig.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(save_dir, fig_name), dpi=100) # Use plt.savefig(..) instead if not using external axes mode


df = pd.read_csv(os.path.join("data", "QQQ_1min_firstratedata.csv"), parse_dates=['timestamp'])

# Set timestamp as index
df.set_index('timestamp', inplace=True)

# Simple Moving Averages
df['MA10'] = df['close'].rolling(window=10).mean()
df['MA20'] = df['close'].rolling(window=20).mean()

# Typical price for each row
# todo: double check on vwap compute 
df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
# Cumulative VWAP
df['VWAP'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()

