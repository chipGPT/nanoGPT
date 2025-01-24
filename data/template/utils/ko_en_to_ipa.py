import subprocess
from konlpy.tag import Okt
import argparse
import re
import json

def transcribe_korean(sentence):
    """Transcribe a Korean sentence into its phonemes using KoNLPy (Okt)."""
    okt = Okt()
    tokens = okt.morphs(sentence)
    tokenized_sentence = ' '.join(tokens)

    try:
        result = subprocess.run(
            ["espeak-ng", "-q", "-v", "ko", "--ipa", tokenized_sentence],
            capture_output=True,
            text=True
        )

        # Remove unwanted characters
        transcription = result.stdout.strip().replace("ㆍ"," ")
        # Check for failed transcription markers
        if "(en)" in transcription or "(ko)" in transcription:
            return "[[[[[" + sentence + "]]]]]"# Return original sentence on failure
        return transcription
    except Exception as e:
        return f"Error in transcribing Korean: {str(e)}"

def handle_mixed_language(word):
    """Handle a word with potential Korean, Language, or number content."""
    if word.isdigit():  # Detect numbers but just pass through for now (different in each language)
        return word
    elif any('가' <= char <= '힣' for char in word):  # Detect Korean
        return transcribe_korean(word)
    else:  # Non-Korean Word
        return "[[[[[" + word + "]]]]]"

def transcribe_multilingual(sentences, input_json_key=None, output_json_key='ipa'):
    """
    Transcribe multilingual sentences and update JSON data directly.

    Args:
        sentences: JSON string or a loaded JSON object.
        input_json_key: Key to extract sentences from in a JSON.
        output_json_key: Key to store IPA transcription in the JSON (default: 'ipa').

    Returns:
        The modified JSON string with IPA transcriptions added.
    """
    try:
        data = json.loads(sentences) if isinstance(sentences, str) else sentences
        if not isinstance(data, list):
            raise ValueError("JSON data should be a list of objects.")

        for item in data:
            if input_json_key in item:
                sentence = item[input_json_key]
                result = []
                words = re.findall(r'\w+|[^\w\s]', sentence, re.UNICODE)
                for word in words:
                    if re.match(r'\w+', word):
                        result.append(handle_mixed_language(word))
                    else:
                        result.append(word)
                transcription_result = " ".join(result)
                item[output_json_key] = transcription_result  # Update directly
                print(transcription_result)
            else:
                print(f"Warning: Key '{input_json_key}' not found in item: {item}")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: {e}")
        return None

    return json.dumps(data, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Transcribe multilingual sentences into IPA phonemes and update JSON data.')
    parser.add_argument('input_file', type=str, help='Path to the input JSON file.')
    parser.add_argument('--input_json_key', type=str, required=True, help='The key of the Korean text to convert to IPA in the JSON file.')
    parser.add_argument('--output_json_key', type=str, default='ipa', help='The key to store the IPA transcription in the JSON file (default: "ipa").')

    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_content = f.read()

        # Transcribe and get the updated JSON data
        updated_json_data = transcribe_multilingual(
            input_content,
            args.input_json_key,
            args.output_json_key
        )

        # Overwrite the original file with the updated JSON
        if updated_json_data:
            with open(args.input_file, 'w', encoding='utf-8') as f:
                f.write(updated_json_data)
            print(f"Successfully updated JSON data in '{args.input_file}'")

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")

if __name__ == '__main__':
    main()
