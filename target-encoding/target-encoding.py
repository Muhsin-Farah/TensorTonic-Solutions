import numpy as np


def target_encoding(categories: list, targets: list) -> list:
    """Returns each category replaced by its mean target as a Python list."""
    cat = np.asarray(categories)
    targ = np.asarray(targets, dtype=float)

    # Find unique categories and their index mappings
    unique_cats, inverse = np.unique(cat, return_inverse=True)

    # Sum targets and count occurrences per unique category using bincount
    category_sums = np.bincount(inverse, weights=targ)
    category_counts = np.bincount(inverse)

    # Calculate mean target for each unique category
    category_means = category_sums / category_counts

    # Replace each original category with its corresponding mean target
    encoded = category_means[inverse]

    return encoded.tolist()