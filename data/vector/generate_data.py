import numpy as np
import pandas as pd
from pathlib import Path

def generate_csv_data(tile):
    """
    Generate data using sine, cosine, and triangle waves and save as CSV file.
    """
    num_samples = 1000
    num_features = 10

    dates = pd.date_range(start='2022-01-01', end='2023-12-31', periods=num_samples)
    data = np.zeros((num_samples, num_features))

    for i in range(num_features):
        if i % 3 == 0:
            data[:, i] = np.sin(np.linspace(0, 2*np.pi, num_samples))
        elif i % 3 == 1:
            data[:, i] = np.cos(np.linspace(0, 2*np.pi, num_samples))
        else:
            data[:, i] = np.abs(np.linspace(-1, 1, num_samples) % 1 - 0.5) * 2

    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(num_features)])
    df.insert(0, "date", dates.strftime('%Y-%m-%d'))

    csv_file = f"./data/{tile}_data.csv"
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    tiles = [f"TL{grid:02d}" for grid in range(1, 6)]

    Path("./data").mkdir(exist_ok=True)

    for tile in tiles:
        generate_csv_data(tile)
