import unittest
import os
import sys  # Import sys to exit with error codes
from tokenizers import (
    NumericRangeTokenizer,
    SentencePieceTokenizer,
    TiktokenTokenizer,
    CustomTokenizer,
    CharTokenizer,
    CustomCharTokenizerWithByteFallback,
    CSVIntegerTokenizer,  # Import the CSVIntegerTokenizer
)
from argparse import Namespace
from rich.console import Console
from rich.theme import Theme
from rich.table import Table

console = Console(theme=Theme({
    "pass": "bold green",
    "fail": "bold red",
    "test_name": "bold yellow",
    "separator": "grey50",
    "input": "bold cyan",
    "output": "bold magenta",
    "info": "bold blue"
}))


class RichTestResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.test_results = []

    def addSuccess(self, test):
        self.test_results.append((test, 'PASS'))
        console.print("[bold green]Test Passed.[/bold green]")
        super().addSuccess(test)

    def addFailure(self, test, err):
        self.test_results.append((test, 'FAIL'))
        console.print("[bold red]Test Failed.[/bold red]")
        super().addFailure(test, err)

    def addError(self, test, err):
        self.test_results.append((test, 'FAIL'))
        console.print("[bold red]Test Error.[/bold red]")
        super().addError(test, err)

    def startTest(self, test):
        console.print('-' * 80, style='separator')
        console.print(f"Running test: [bold]{test._testMethodName}[/bold]", style='test_name')
        super().startTest(test)

    def stopTest(self, test):
        super().stopTest(test)


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTokenizers)
    result = RichTestResult()
    suite.run(result)
    # Print final table
    console.print('=' * 80, style='separator')
    console.print("[bold]Test Results:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test")
    table.add_column("Result", justify="center")
    for test, status in result.test_results:
        test_name = test._testMethodName
        if status == 'PASS':
            style = "pass"
        else:
            style = "fail"
        table.add_row(test_name, f"[{style}]{status}[/{style}]")
    console.print(table)
    # Exit with error code if any test failed
    if not result.wasSuccessful():
        sys.exit(1)  # Exit with status code 1 if tests failed
    else:
        sys.exit(0)  # Exit with status code 0 if all tests passed


class TestTokenizers(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_text = "Hello\nworld\nThis is a test."
        self.numeric_data = "123\n456\n789"
        self.tokens_file = "tokens.txt"

        # Create a tokens file for custom tokenizers
        with open(self.tokens_file, 'w') as f:
            f.write("Hello\nworld\nThis is a test.\n")

    def tearDown(self):
        # Clean up tokens file
        if os.path.exists(self.tokens_file):
            os.remove(self.tokens_file)
        # Remove temporary files created by SentencePiece
        for fname in ["spm_input.txt", "trained_spm_model"]:
            for ext in ["", ".model", ".vocab"]:
                full_name = f"{fname}{ext}"
                if os.path.exists(full_name):
                    os.remove(full_name)
        if os.path.exists("meta.pkl"):
            os.remove("meta.pkl")
        if os.path.exists("remaining.txt"):
            os.remove("remaining.txt")
        # Clean up custom_chars_file if exists
        if hasattr(self, 'custom_chars_file') and os.path.exists(self.custom_chars_file):
            os.remove(self.custom_chars_file)
        # Clean up CSV input file if exists
        if hasattr(self, 'csv_input_file') and os.path.exists(self.csv_input_file):
            os.remove(self.csv_input_file)
        # Clean up 'out' directory if it exists
        if os.path.exists("out"):
            import shutil
            shutil.rmtree("out")

    def test_numeric_range_tokenizer(self):
        args = Namespace(min_token=100, max_token=1000)
        tokenizer = NumericRangeTokenizer(args)
        ids = tokenizer.tokenize(self.numeric_data)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.numeric_data.strip(), style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        self.assertEqual(self.numeric_data.strip(), detokenized)

    def test_sentencepiece_tokenizer(self):
        args = Namespace(
            vocab_size=30,
            spm_model_file=None,
            spm_vocab_file=None,
            skip_tokenization=False
        )
        # Simulate training data
        with open("spm_input.txt", "w") as f:
            f.write(self.sample_text)
        tokenizer = SentencePieceTokenizer(args, input_files="spm_input.txt")
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.sample_text, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        self.assertEqual(self.sample_text, detokenized)

    def test_tiktoken_tokenizer(self):
        args = Namespace(tiktoken_encoding='gpt2')
        tokenizer = TiktokenTokenizer(args)
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.sample_text, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        self.assertEqual(self.sample_text, detokenized)

    def test_custom_tokenizer(self):
        args = Namespace(tokens_file=self.tokens_file)
        tokenizer = CustomTokenizer(args)
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.sample_text, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        tokens_to_check = ["Hello", "world", "This", "is", "a", "test"]
        for token in tokens_to_check:
            self.assertIn(token, detokenized)

    def test_char_tokenizer(self):
        args = Namespace(reuse_chars=False)
        tokenizer = CharTokenizer(args, self.sample_text, None)
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.sample_text, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        self.assertEqual(self.sample_text, detokenized)

    def test_custom_char_tokenizer_with_byte_fallback(self):
        self.custom_chars_file = "custom_chars.txt"
        args = Namespace(custom_chars_file=self.custom_chars_file)
        # Create a custom characters file for testing
        with open(args.custom_chars_file, 'w', encoding='utf-8') as f:
            f.write('a\nb\nc\n')

        tokenizer = CustomCharTokenizerWithByteFallback(args)
        test_string = 'abc😊'

        ids = tokenizer.tokenize(test_string)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(test_string, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        self.assertEqual(test_string, detokenized)

    def test_csv_integer_tokenizer(self):
        # Sample CSV content
        sample_csv = "2020,10,31\n2021,11,1\n2022,12,15"
        args = Namespace(
            method='csv_integer',
            field_prefixes=['y', 'm', 'd'],
            field_min_values=[2020, 1, 1],
            field_max_values=[2022, 12, 31],
            train_input=None,
            val_input=None,
            train_output='train.bin',
            val_output='val.bin',
        )
        tokenizer = CSVIntegerTokenizer(args)
        ids = tokenizer.tokenize(sample_csv)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(sample_csv, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        self.assertEqual(sample_csv, detokenized)

if __name__ == '__main__':
    run_tests()

