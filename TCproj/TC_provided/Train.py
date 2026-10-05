#!/usr/bin/env python3

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

"""
STOP_WORDS = { #list of ignored words
    "the", "a", "an", "and", "or", "but",
    "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "is", "are", "was",
    "were", "be", "been", "being",
    "this", "that", "it", "as"
}
"""


def tokenize(text):
    return TOKEN_RE.findall(text.lower())
    #return [word for word in tokens if word not in STOP_WORDS]


def resolve_path(list_file, document_path):
    """
    Try the path exactly as written first.
    If it does not exist, interpret it relative to the directory
    containing the training-list file.
    """
    if os.path.exists(document_path):
        return document_path

    base_dir = os.path.dirname(os.path.abspath(list_file))
    alternate = os.path.join(base_dir, document_path)

    if os.path.exists(alternate):
        return alternate

    return document_path


def train_naive_bayes(training_file, model_file, alpha=1.0):
    # Number of documents in each category.
    category_doc_counts = Counter()

    # Total number of word tokens observed in each category.
    category_token_totals = Counter()

    # word_counts[category][word] = number of occurrences.
    word_counts = defaultdict(Counter)

    vocabulary = set()
    total_documents = 0

    with open(training_file, "r", encoding="utf-8", errors="ignore") as labels:
        for line_number, line in enumerate(labels, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                document_path, category = line.rsplit(maxsplit=1)
            except ValueError:
                raise ValueError(
                    f"Invalid line {line_number} in {training_file}: {line!r}"
                )

            actual_path = resolve_path(training_file, document_path)

            with open(actual_path, "r", encoding="utf-8", errors="ignore") as document:
                tokens = tokenize(document.read())

            category_doc_counts[category] += 1
            total_documents += 1

            counts = Counter(tokens)
            word_counts[category].update(counts)
            category_token_totals[category] += sum(counts.values())
            vocabulary.update(counts.keys())

    if total_documents == 0:
        raise ValueError("Training file contains no usable training documents.")

    categories = sorted(category_doc_counts.keys())
    vocabulary_size = len(vocabulary)

    # Save log priors so the testing program does less repeated work.
    log_priors = {
        category: math.log(category_doc_counts[category] / total_documents)
        for category in categories
    }

    model = {
        "model_type": "multinomial_naive_bayes",
        "alpha": alpha,
        "categories": categories,
        "total_documents": total_documents,
        "category_doc_counts": dict(category_doc_counts),
        "category_token_totals": dict(category_token_totals),
        "vocabulary_size": vocabulary_size,
        "vocabulary": sorted(vocabulary),
        "word_counts": {
            category: dict(word_counts[category])
            for category in categories
        },
        "log_priors": log_priors,
    }

    with open(model_file, "w", encoding="utf-8") as out:
        json.dump(model, out)

    print(f"Trained on {total_documents} documents.")
    print(f"Categories: {', '.join(categories)}")
    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Saved model to: {model_file}")


def main():
    if len(sys.argv) == 4:
        training_file = sys.argv[1]
        model_file = sys.argv[2]
        alpha = float(sys.argv[3])
    elif len(sys.argv) == 3:
        training_file = sys.argv[1]
        model_file = sys.argv[2]
        alpha = 0.1
    elif len(sys.argv) == 1:
        training_file = input("Training labels file: ").strip()
        model_file = input("Output model file: ").strip()
        alpha_text = input("Laplace smoothing alpha [1.0]: ").strip()
        alpha = float(alpha_text) if alpha_text else 1.0
    else:
        print(
            f"Usage: {sys.argv[0]} TRAINING_LABELS MODEL_FILE [ALPHA]",
            file=sys.stderr,
        )
        sys.exit(1)

    if alpha <= 0:
        print("Alpha must be greater than 0.", file=sys.stderr)
        sys.exit(1)

    try:
        train_naive_bayes(training_file, model_file, alpha)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
