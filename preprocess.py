import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

import mplfinance as mpf

from sklearn.preprocessing import LabelEncoder, StandardScaler


# Example: features to include in LSTM input
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume',
                'MA10', 'MA20', 'vwap']
START_TIME = pd.to_timedelta("09:31:00")  # starting timestamp each day
SEQ_LEN = 30


# ==========================================
# STEP 1: Load and Clean Data
# ==========================================

def load_label_data(label_path):
    df = pd.read_csv(label_path, parse_dates=['date'])

    # Fill missing values in end_time with start_time
    df["entry"] = df["entry"].fillna(df["end time"])

    # df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # print(df.columns)
    
    # Combine date + start_time or end_time into a unified timestamp
    df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['end time'])
    df = df[['timestamp', 'label', 'entry']].dropna()

    # todo: decide whether need to set index, only mlpfinance lib required this? 
    # df = df.set_index('timestamp')
    return df

def load_feature_data(feature_path):
    df = pd.read_csv(feature_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # todo: decide whether need to set index, only mlpfinance lib required this? 
    # df = df.set_index('timestamp')
    return df

def preprocess_feats(df):
    # Simple Moving Averages
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()

    # Compute typical price
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3

    # For each day, compute cumulative values
    df['vwap'] = df.groupby(df['timestamp'].dt.date).apply(
    lambda g: (g['tp'] * g['volume']).cumsum() / g['volume'].cumsum()
        ).reset_index(level=0, drop=True)

    return df

# ==========================================
# STEP 2: Merge Labels with Features
# ==========================================

def merge_data(features_df, labels_df):
    merged = pd.merge_asof(
        features_df.sort_values('timestamp'),
        labels_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('5min')  # optional time tolerance
    )
    # Fill missing labels with "no"
    # merged['label'] = merged['label'].fillna('no')
    merged = merged.dropna()
    return merged


def create_lstm_sequences(df, feature_cols=FEATURE_COLS):
    """
    Convert each row into an LSTM input sequence from 9:31 to 'end_time'.
    Zero-padding is applied if sequences are shorter than the max length.
    """
    X = []
    y = []
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by day
    for day, day_df in df.groupby(df['timestamp'].dt.date):
        print("day/ day_df: ", day)
        day_df = day_df.sort_values('timestamp')
        print("start processing day_df: ", day_df)

        # extract features & labels
        features = day_df[feature_cols].to_numpy()
        labels = day_df['label_id'].to_numpy()

        # sliding window over the daily data
        for i in range(5, len(day_df)):
            start = i - SEQ_LEN + 1
            if np.isnan(labels[i]): continue


            if start < 0:
                # pad needed
                pad_len = -start
                padded = np.zeros((pad_len, len(feature_cols)))
                seq = np.vstack([padded, features[0:i+1]])
            else:
                seq = features[start:i+1]

            X.append(seq)
            y.append(labels[i])

            # print(X, np.array(X).shape)
            # print("\n\n labels: ", y)
            # exit()

    return np.array(X), np.array(y)


label_path = os.path.join("data", "labels_timestamp_new.csv")
feature_path = os.path.join("data", "QQQ_1min_firstratedata.csv")

features_df = load_feature_data(feature_path)
labels_df = load_label_data(label_path)

print("loaded features df: ", features_df)
print("\n\n loaded labels_df: ", labels_df)

# feature engineering 
features_df = preprocess_feats(features_df)
print("\n\n processed features_df: ", features_df)

merged_df = merge_data(features_df, labels_df)
print("\n\n merged_df: ", merged_df)

# Get counts of each label
label_counts = merged_df["label"].value_counts()
# Get percentage distribution
label_percent = merged_df["label"].value_counts(normalize=True) * 100
print("📊 Label Counts:")
print(label_counts)
print("\n📈 Label Percentages (%):")
print(label_percent.round(2))

# todo: add open range 5-min bound 

# todo: convert labels to values 
label_map = {
    'no': 0,
    'no ': 0,
    'up': 1,
    'aup': 1,
    '5min-rb-up': 1,
    'rb-up': 1,
    'vup': 1,
    'down': 2,
    'adown': 2,
    '5min-rb-down': 2,
    'rb-down': 2,
    'consolid': 0
}
merged_df['label_id'] = merged_df['label'].map(label_map)
label_counts = merged_df["label_id"].value_counts()
label_percent = merged_df["label_id"].value_counts(normalize=True) * 100

print("\n\nmerged Label Counts:")
print(label_counts)
print("\n📈 Label Percentages (%):")
print(label_percent.round(2))

# Sving 
merged_df.to_csv("processed_stock_data.csv", index=False)


print("\n\n Creating sequences ...... ")
X, y = create_lstm_sequences(merged_df)
np.savez("stock_sequences.npz", X=X, y=y)
