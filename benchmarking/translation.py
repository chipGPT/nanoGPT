# translation.py
import os
import time
import torch
import numpy as np
from sacrebleu import corpus_bleu
from datasets import load_dataset

def load_translation_dataset(dataset_name, source_lang, target_lang, split="test"):
    """
    Load a translation dataset using Hugging Face's datasets library.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., "wmt24", "wmt23").
        source_lang (str): Source language code (e.g., "en").
        target_lang (str): Target language code (e.g., "de").
        split (str): Split of the dataset to use (default: "test").

    Returns:
        Dataset: A Hugging Face Dataset object containing the translation examples.
    """
    dataset = load_dataset(dataset_name, f"{source_lang}-{target_lang}", split=split)
    return dataset

def translate_batch(model, encode, decode, source_texts, device, max_length=100):
    """
    Translate a batch of source texts using the given model.

    Args:
        model: The translation model.
        encode: Function to encode text into input IDs.
        decode: Function to decode output IDs into text.
        source_texts (list of str): List of source texts to translate.
        device (str): The device to run inference on.
        max_length (int): Maximum token length for translations (default: 100).

    Returns:
        list of str: List of translated texts.
    """
    model.eval()
    inputs = [encode(text) for text in source_texts]
    input_ids = torch.tensor(inputs, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=max_length, num_beams=5)
    return [decode(output.tolist()) for output in outputs]

def evaluate_translation(model, encode, decode, dataset, source_lang, target_lang, device):
    """
    Evaluate a translation model using BLEU score.

    Args:
        model: The translation model.
        encode: Function to encode text into input IDs.
        decode: Function to decode output IDs into text.
        dataset: The translation dataset.
        source_lang (str): Source language code.
        target_lang (str): Target language code.
        device (str): The device to run inference on.

    Returns:
        float: BLEU score of the model's translations.
    """
    source_texts = [example[source_lang] for example in dataset]
    reference_texts = [[example[target_lang]] for example in dataset]

    translations = []
    batch_size = 16

    for i in range(0, len(source_texts), batch_size):
        batch = source_texts[i:i + batch_size]
        translated_batch = translate_batch(model, encode, decode, batch, device)
        translations.extend(translated_batch)

    bleu = corpus_bleu(translations, reference_texts)
    return bleu.score

def benchmark_translation(model, encode, decode, dataset_name, source_lang, target_lang, device):
    """
    Benchmark translation performance on a specific dataset.

    Args:
        model: The translation model.
        encode: Function to encode text into input IDs.
        decode: Function to decode output IDs into text.
        dataset_name (str): Name of the translation dataset (e.g., "wmt24").
        source_lang (str): Source language code.
        target_lang (str): Target language code.
        device (str): The device to run inference on.

    Returns:
        float: BLEU score of the model's translations.
    """
    print(f"Loading dataset {dataset_name} ({source_lang} -> {target_lang})...")
    dataset = load_translation_dataset(dataset_name, source_lang, target_lang)

    print("Evaluating translations...")
    start_time = time.time()
    bleu_score = evaluate_translation(model, encode, decode, dataset, source_lang, target_lang, device)
    end_time = time.time()

    print(f"BLEU Score: {bleu_score:.2f}")
    print(f"Evaluation Time: {end_time - start_time:.2f} seconds")

    return bleu_score
