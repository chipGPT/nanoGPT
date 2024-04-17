import numpy as np
import h5py as h5
import re
import sys
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

BASENAME = "TL_CS"

def get_data_from_csv(csv_file):
    """
    Read data from CSV file and return as numpy array.
    """
    df = pd.read_csv(csv_file)
    return df.values

def process_data(data):
    """
    Process data to intermediate format.
    """
    def _get_embedding(date_str):
        date = datetime.strptime(date_str, '%Y-%m-%d')
        day = (date - datetime(date.year, 1, 1)).days
        embedding = (np.sin((day/365)*2*np.pi), np.cos((day/365)*2*np.pi))
        return embedding

    processed_data = []
    for i in range(len(data)-1):
        embedding = _get_embedding(data[i, 0])
        next_embedding = _get_embedding(data[i+1, 0])

        cs = data[i, 1:].astype(float)

        days_sin = np.ones_like(cs[:1])*embedding[0]
        days_cos = np.ones_like(cs[:1])*embedding[1]
        next_days_sin = np.ones_like(cs[:1])*next_embedding[0]
        next_days_cos = np.ones_like(cs[:1])*next_embedding[1]

        processed_data.append(np.concatenate([(cs/1000)*2-1, days_sin, days_cos, next_days_sin, next_days_cos]))

    return np.array(processed_data)

if __name__ == "__main__":
    grid = int(sys.argv[1])
    tile = f"TL{grid:02d}"

    csv_file = f"./data/{tile}_data.csv"
    data = get_data_from_csv(csv_file)

    train_data = data[data[:, 0] < '2022-12-31']
    test_data = data[data[:, 0] >= '2022-12-31']

    print(f"{tile}: processing {len(data)} frames")

    train_processed = process_data(train_data)
    test_processed = process_data(test_data)

    np.save(f"./data/{tile}_train.npy", train_processed.astype(np.float16))
    np.save(f"./data/{tile}_test.npy", test_processed.astype(np.float16))
