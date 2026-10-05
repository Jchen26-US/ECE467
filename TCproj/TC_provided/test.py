#!/usr/bin/env python3

import json
import math
import os
import re
import sys
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text):
    """Use the same tokenizer as the training program."""
    return TOKEN_RE.findall(text.lower())


def resolve_path(list_file, document_path):
    """
    Try the path exactly as written first.
    If it does not exist, interpret it relative to the directory
    containing the test-list file.
    """
    if os.path.exists(document_path):
        return document_path

    base_dir = os.path.dirname(os.path.abspath(list_file))
    alternate = os.path.join(base_dir, document_path)

    if os.path.exists(alternate):
        return alternate

    return document_path


def classify(tokens, model):
    alpha = float(model["alpha"])
    categories = model["categories"]
    vocabulary = set(model["vocabulary"])
    vocabulary_size = int(model["vocabulary_size"])
    category_token_totals = model["category_token_totals"]
    word_counts = model["word_counts"]
    log_priors = model["log_priors"]

    token_counts = Counter(token for token in tokens if token in vocabulary)

    best_category = None
    best_score = float("-inf")

    for category in categories:
        score = float(log_priors[category])

        denominator = (
            int(category_token_totals[category])
            + alpha * vocabulary_size
        )

        category_counts = word_counts[category]

        for token, count in token_counts.items():
            numerator = category_counts.get(token, 0) + alpha
            score += count * math.log(numerator / denominator)

        if score > best_score:
            best_score = score
            best_category = category

    return best_category


def test_naive_bayes(model_file, test_list_file, output_file):
    with open(model_file, "r", encoding="utf-8") as infile:
        model = json.load(infile)

    if model.get("model_type") != "multinomial_naive_bayes":
        raise ValueError("The supplied model is not a supported Naive Bayes model.")

    predictions = []

    with open(test_list_file, "r", encoding="utf-8", errors="ignore") as test_list:
        for line_number, line in enumerate(test_list, start=1):
            document_path = line.strip()
            if not document_path:
                continue

            actual_path = resolve_path(test_list_file, document_path)

            with open(actual_path, "r", encoding="utf-8", errors="ignore") as document:
                tokens = tokenize(document.read())

            predicted_category = classify(tokens, model)
            predictions.append((document_path, predicted_category))

    with open(output_file, "w", encoding="utf-8") as outfile:
        for document_path, category in predictions:
            outfile.write(f"{document_path} {category}\n")

    print(f"Classified {len(predictions)} documents.")
    print(f"Saved predictions to: {output_file}")


def main():
    if len(sys.argv) == 4:
        model_file = sys.argv[1]
        test_list_file = sys.argv[2]
        output_file = sys.argv[3]
    elif len(sys.argv) == 1:
        model_file = input("Model file: ").strip()
        test_list_file = input("Test list file: ").strip()
        output_file = input("Output predictions file: ").strip()
    else:
        print(
            f"Usage: {sys.argv[0]} MODEL_FILE TEST_LIST OUTPUT_FILE",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        test_naive_bayes(model_file, test_list_file, output_file)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
