import sys
import pandas as pd

def visualize_data(csv_file, column):
    """
    Visualize the first 100 entries of the selected column.
    """
    df = pd.read_csv(csv_file)

    if column not in df.columns:
        print(f"Column '{column}' not found in the CSV file.")
        return

    data = df[column].values[:100]

    min_val = data.min()
    max_val = data.max()
    range_val = max_val - min_val

    for val in data:
        normalized_val = (val - min_val) / range_val
        bar_length = int(normalized_val * 50)
        bar = '█' * bar_length
        print(f"{val:.4f} {bar}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python visualize_data.py <csv_file> <column>")
        sys.exit(1)

    csv_file = sys.argv[1]
    column = sys.argv[2]

    visualize_data(csv_file, column)
