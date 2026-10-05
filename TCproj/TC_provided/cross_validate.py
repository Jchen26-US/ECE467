#!/usr/bin/env python3

import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def resolve_path(labels_file, document_path):
    if os.path.exists(document_path):
        return document_path

    base_dir = os.path.dirname(os.path.abspath(labels_file))
    alternate = os.path.join(base_dir, document_path)

    if os.path.exists(alternate):
        return alternate

    return document_path


def read_labeled_documents(labels_file):
    """
    Read:
        path/to/document LABEL
    from a labels file.
    """
    data = []

    with open(labels_file, "r", encoding="utf-8", errors="ignore") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                path, label = line.rsplit(maxsplit=1)
            except ValueError:
                raise ValueError(
                    f"Invalid line {line_number} in {labels_file}: {line!r}"
                )

            data.append((path, label))

    if not data:
        raise ValueError("No labeled documents were found.")

    return data


def make_stratified_folds(data, k=5, seed=42):
    """
    Group examples by label, shuffle each label separately,
    then distribute examples round-robin among the folds.

    This keeps each fold's category proportions approximately balanced.
    """
    by_label = defaultdict(list)

    for item in data:
        by_label[item[1]].append(item)

    rng = random.Random(seed)

    folds = [[] for _ in range(k)]

    for label, items in by_label.items():
        rng.shuffle(items)

        for i, item in enumerate(items):
            folds[i % k].append(item)

    # Shuffle within each fold so examples are not grouped by label.
    for fold in folds:
        rng.shuffle(fold)

    return folds


def train_naive_bayes(training_data, labels_file, alpha=1.0):
    """
    Train Multinomial Naive Bayes manually.
    """
    category_doc_counts = Counter()
    category_token_totals = Counter()
    word_counts = defaultdict(Counter)

    vocabulary = set()
    total_documents = 0

    for document_path, category in training_data:
        actual_path = resolve_path(labels_file, document_path)

        with open(actual_path, "r", encoding="utf-8", errors="ignore") as document:
            tokens = tokenize(document.read())

        category_doc_counts[category] += 1
        total_documents += 1

        counts = Counter(tokens)

        word_counts[category].update(counts)
        category_token_totals[category] += sum(counts.values())
        vocabulary.update(counts.keys())

    categories = sorted(category_doc_counts.keys())
    vocabulary_size = len(vocabulary)

    log_priors = {
        category: math.log(category_doc_counts[category] / total_documents)
        for category in categories
    }

    return {
        "alpha": alpha,
        "categories": categories,
        "category_token_totals": category_token_totals,
        "word_counts": word_counts,
        "vocabulary": vocabulary,
        "vocabulary_size": vocabulary_size,
        "log_priors": log_priors,
    }


def classify(tokens, model):
    alpha = model["alpha"]
    categories = model["categories"]
    category_token_totals = model["category_token_totals"]
    word_counts = model["word_counts"]
    vocabulary = model["vocabulary"]
    vocabulary_size = model["vocabulary_size"]
    log_priors = model["log_priors"]

    token_counts = Counter(
        token for token in tokens
        if token in vocabulary
    )

    best_category = None
    best_score = float("-inf")

    for category in categories:
        score = log_priors[category]

        denominator = (
            category_token_totals[category]
            + alpha * vocabulary_size
        )

        for token, count in token_counts.items():
            numerator = word_counts[category].get(token, 0) + alpha

            score += count * math.log(
                numerator / denominator
            )

        if score > best_score:
            best_score = score
            best_category = category

    return best_category


def evaluate(model, validation_data, labels_file):
    correct = 0
    total = 0

    # confusion[actual][predicted]
    confusion = defaultdict(Counter)

    for document_path, actual_label in validation_data:
        actual_path = resolve_path(labels_file, document_path)

        with open(actual_path, "r", encoding="utf-8", errors="ignore") as document:
            tokens = tokenize(document.read())

        predicted_label = classify(tokens, model)

        confusion[actual_label][predicted_label] += 1

        if predicted_label == actual_label:
            correct += 1

        total += 1

    accuracy = correct / total if total else 0.0

    return accuracy, correct, total, confusion


def print_confusion_matrix(confusion, categories):
    print("Confusion matrix (rows = actual, columns = predicted):")

    print(f"{'':10}", end="")
    for category in categories:
        print(f"{category:10}", end="")
    print()

    for actual in categories:
        print(f"{actual:10}", end="")

        for predicted in categories:
            print(f"{confusion[actual][predicted]:10}", end="")

        print()


def cross_validate(labels_file, k=5, alpha=0.1, seed=42):
    data = read_labeled_documents(labels_file)

    if k < 2:
        raise ValueError("k must be at least 2.")

    if k > len(data):
        raise ValueError("k cannot be larger than the number of documents.")

    folds = make_stratified_folds(data, k, seed)

    fold_accuracies = []
    total_correct = 0
    total_tested = 0

    all_confusion = defaultdict(Counter)

    categories = sorted({label for _, label in data})

    print(f"Documents: {len(data)}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Folds: {k}")
    print(f"Alpha: {alpha}")
    print()

    for fold_index in range(k):
        validation_data = folds[fold_index]

        training_data = []

        for i in range(k):
            if i != fold_index:
                training_data.extend(folds[i])

        model = train_naive_bayes(
            training_data,
            labels_file,
            alpha
        )

        accuracy, correct, total, confusion = evaluate(
            model,
            validation_data,
            labels_file
        )

        fold_accuracies.append(accuracy)
        total_correct += correct
        total_tested += total

        for actual in confusion:
            all_confusion[actual].update(confusion[actual])

        print(
            f"Fold {fold_index + 1}: "
            f"{correct}/{total} correct "
            f"({accuracy * 100:.2f}%)"
        )

    print()
    print("Results")
    print("-------")

    average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    overall_accuracy = total_correct / total_tested

    print(f"Average fold accuracy: {average_accuracy * 100:.2f}%")
    print(f"Overall accuracy:      {overall_accuracy * 100:.2f}%")
    print()

    print_confusion_matrix(all_confusion, categories)


def main():
    if len(sys.argv) >= 2:
        labels_file = sys.argv[1]
    else:
        labels_file = input("Training labels file: ").strip()

    if len(sys.argv) >= 3:
        k = int(sys.argv[2])
    else:
        k = 5

    if len(sys.argv) >= 4:
        alpha = float(sys.argv[3])
    else:
        alpha = 1.0

    if len(sys.argv) >= 5:
        seed = int(sys.argv[4])
    else:
        seed = 42

    try:
        cross_validate(
            labels_file,
            k=k,
            alpha=alpha,
            seed=seed
        )

    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
