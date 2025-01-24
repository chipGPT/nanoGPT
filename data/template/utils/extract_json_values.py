import json
import argparse

def extract_values_by_key(json_file, key, output_file):
    """
    Extracts all values associated with a specific key from a JSON file
    and writes them to an output text file, each value on a new line.

    Args:
        json_file: Path to the input JSON file.
        key: The key to search for in the JSON data.
        output_file: Path to the output text file.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
            data = json.load(f_in)

            def extract_values(data, key, f_out):
                if isinstance(data, dict):
                    for k, v in data.items():
                        if k == key:
                            f_out.write(str(v) + '\n')
                        else:
                            extract_values(v, key, f_out)
                elif isinstance(data, list):
                    for item in data:
                        extract_values(item, key, f_out)

            extract_values(data, key, f_out)

    except FileNotFoundError:
        print(f"Error: Input file '{json_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract values of a specific key from a JSON file to a text file.")
    parser.add_argument("json_file", help="Path to the input JSON file.")
    parser.add_argument("key", help="The key whose values you want to extract.")
    parser.add_argument("output_file", help="Path to the output text file.")

    args = parser.parse_args()

    extract_values_by_key(args.json_file, args.key, args.output_file)
    print(f"Values for key '{args.key}' extracted to '{args.output_file}'")

if __name__ == "__main__":
    main()
