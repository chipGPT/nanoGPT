import subprocess
from dragonmapper import hanzi
import jieba
import argparse
import re
import json


def transcribe_chinese(sentence):
    """Transcribe a Chinese sentence into its phonemes using dragonmapper."""
    try:
        result = hanzi.to_ipa(sentence)
        return "".join(result)
    except Exception as e:
        return f"Error in transcribing Chinese: {str(e)}"


def handle_mixed_language(word):
    """Handle a word with potential Chinese, Language, or number content."""
    if word.isdigit():  # Detect numbers but just pass through for now (different in each language)
        return word
    elif any(hanzi.is_simplified(char) for char in word):  # Detect Chinese
        return transcribe_chinese(word)
    else:  # Non-Chinese Word
        return "[[[[[" + word + "]]]]]"


def transcribe_multilingual(data, output_file, json_inplace_update=False, json_input_field="sentence",
                            json_output_field="sentence_ipa"):
    """
    Transcribe multilingual sentences (English and Chinese, with numbers) and save to a file.

    Args:
        data: The input data (list of dictionaries if JSON, list of strings if plain text).
        output_file: Path to the output file.
        json_inplace_update: If True, process JSON input and add IPA to the same JSON.
        json_input_field: The field in the JSON data to transcribe (default: "sentence").
        json_output_field: The field to write the IPA transcription to (default: "sentence_ipa").
    """
    if json_inplace_update:
        # In-place update for JSON data
        for item in data:
            if json_input_field in item:
                sentence = item[json_input_field]
                result = []

                # Split sentence using jieba
                seg_list = jieba.cut(sentence, cut_all=False)
                seg_sentence = "".join(seg_list)

                # Split sentence but keep punctuation
                words = re.findall(r'\w+|[^\w\s]', seg_sentence, re.UNICODE)
                for word in words:
                    if re.match(r'\w+', word):  # Only process words
                        result.append(handle_mixed_language(word))
                    else:
                        result.append(word)  # Preserve punctuation

                transcription_result = " ".join(result)
                item[json_output_field] = transcription_result

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"In-place JSON transcription saved to {output_file}")

    else:
        # Standard transcription (either JSON or plain text to plain text output)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                result = []
                if isinstance(item, dict):
                    sentence = item.get(json_input_field, "")
                else:
                    sentence = item

                # Split sentence using jieba
                seg_list = jieba.cut(sentence, cut_all=False)
                seg_sentence = "".join(seg_list)

                # Split sentence but keep punctuation
                words = re.findall(r'\w+|[^\w\s]', seg_sentence, re.UNICODE)
                for word in words:
                    if re.match(r'\w+', word):  # Only process words
                        result.append(handle_mixed_language(word))
                    else:
                        result.append(word)  # Preserve punctuation

                transcription_result = " ".join(result)
                f.write(transcription_result + "\n")
                print(transcription_result)  # Print to console


def main():
    parser = argparse.ArgumentParser(description='Transcribe multilingual sentences into IPA phonemes.')
    parser.add_argument('input_file', type=str,
                        help='Path to the input file containing sentences in json format.')
    parser.add_argument('output_file', type=str, help='Path to the output file for IPA transcription.')
    parser.add_argument('--input_type', type=str, choices=['json', 'text'], default='json',
                        help='Type of input file: "json" or "text" (default: json)')
    parser.add_argument("-j", "--json_inplace_update", action="store_true",
                        help="Process JSON input and add IPA to the same JSON entries")
    parser.add_argument("--json_input_field", default="sentence",
                        help="JSON field to read from (default: sentence)")
    parser.add_argument("--json_output_field", default="sentence_ipa",
                        help="JSON field to write IPA to (default: sentence_ipa)")

    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            if args.input_type == 'json':
                data = json.load(f)
            else:
                data = f.readlines()

        transcribe_multilingual(data, args.output_file, args.json_inplace_update,
                                args.json_input_field, args.json_output_field)

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{args.input_file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    main()
