import pandas as pd
import argparse
import json

def tsv_to_json_auto_columns(input_file, output_file, delimiter='\t'):
    """
    Converts a TSV file to a JSON file, automatically detecting column names.

    Args:
        input_file (str): Path to the input TSV file.
        output_file (str): Path to the output JSON file.
        delimiter (str, optional): Delimiter used in the TSV file. Defaults to '\t' (tab).
    """
    try:
        # Read the TSV file using pandas, automatically detecting header
        df = pd.read_csv(input_file, sep=delimiter, header='infer')

        # Convert DataFrame to a list of dictionaries (JSON format)
        data = df.to_dict(orient='records')

        # Write to JSON file with pretty printing
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Successfully converted '{input_file}' to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except pd.errors.ParserError:
        print(f"Error: Invalid TSV format in '{input_file}'. Check the delimiter and file structure.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a TSV file to JSON with automatic column name detection.")
    parser.add_argument("input_file", help="Path to the input TSV file.")
    parser.add_argument("output_file", help="Path to the output JSON file.")
    parser.add_argument("--delimiter", default='\t', help="Delimiter used in the TSV (default: tab)")

    args = parser.parse_args()

    tsv_to_json_auto_columns(args.input_file, args.output_file, args.delimiter)
