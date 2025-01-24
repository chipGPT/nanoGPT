import argparse
import collections

def analyze_text(char_file, text_file, num_lines, save_lines):
    """
    Analyzes a text file based on a given set of characters.

    Args:
        char_file (str): Path to the file containing the list of characters.
        text_file (str): Path to the text file to be analyzed.
        num_lines (int): Number of lines to display (for option 2).
        save_lines (str): Path to save the lines with characters (optional).
    """

    try:
        with open(char_file, 'r', encoding='utf-8') as f:
            chars = set(f.read().split())
    except FileNotFoundError:
        print(f"Error: Character file not found at {char_file}")
        return

    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except FileNotFoundError:
        print(f"Error: Text file not found at {text_file}")
        return

    # 1. CLI Histogram of character occurrences
    char_counts = collections.Counter(c for c in text_content if c in chars)

    if char_counts:
        print("\nCharacter Frequency (CLI Histogram):")
        max_count = max(char_counts.values())
        for char, count in sorted(char_counts.items()):
            bar_length = int(40 * count / max_count)  # Scale bar to 40 characters
            print(f"{char}: {'█' * bar_length} ({count})")
    else:
        print("No characters from the character file were found in the text file.")

    # 2. List of lines with characters
    lines_with_chars = []
    with open(text_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if any(c in line for c in chars):
                lines_with_chars.append(f"Line {line_num}: {line.strip()}")

    if lines_with_chars:
        print("\nLines containing specified characters:")
        for line in lines_with_chars[:num_lines]:
            print(line)

        if save_lines:
            try:
                with open(save_lines, 'w', encoding='utf-8') as outfile:
                    for line in lines_with_chars:
                        outfile.write(line + '\n')
                print(f"Lines saved to {save_lines}")
            except Exception as e:
                print(f"Error saving lines to file: {e}")
    else:
        print("No lines in the text file contain the specified characters.")

def main():
    parser = argparse.ArgumentParser(description="Analyze a text file based on a set of characters.")
    parser.add_argument("char_file", help="Path to the file containing the list of characters.")
    parser.add_argument("text_file", help="Path to the text file to analyze.")
    args = parser.parse_args()

    while True:
        print("\nChoose an option:")
        print("1. Show CLI histogram of character occurrences")
        print("2. List lines containing characters")
        print("3. Exit")

        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            analyze_text(args.char_file, args.text_file, 0, None)
        elif choice == '2':
            num_lines = input("Enter the number of lines to display (default 10): ")
            num_lines = int(num_lines) if num_lines.isdigit() else 10

            save_option = input("Save lines to a file? (y/n): ").lower()
            save_path = None
            if save_option == 'y':
                save_path = input("Enter the path to save the file: ")

            analyze_text(args.char_file, args.text_file, num_lines, save_path)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
