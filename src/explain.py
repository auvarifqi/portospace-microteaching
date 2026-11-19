"""Lexical explanations from a fitted TF-IDF + LogisticRegression pipeline."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.pipeline import Pipeline


def top_global_terms(
    model: Pipeline,
    k: int = 15,
    class_index: int = 1,
) -> List[Tuple[str, float]]:
    """Return top-k terms by coefficient magnitude for the positive class."""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        return []
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coefs = classifier.coef_[class_index if classifier.coef_.shape[0] > 1 else 0]
    order = np.argsort(coefs)[::-1][:k]
    return [(str(feature_names[i]), float(coefs[i])) for i in order]


def top_terms_for_text(
    model: Pipeline,
    text: str,
    k: int = 8,
    class_index: int = 1,
) -> List[Tuple[str, float]]:
    """
    Per-message evidence: tfidf_value × coefficient for present features.
    These are model evidence scores, not human causal reasons.
    """
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        return []

    X = vectorizer.transform([text])
    coefs = classifier.coef_[class_index if classifier.coef_.shape[0] > 1 else 0]
    row = X.tocoo()
    contributions = []
    feature_names = vectorizer.get_feature_names_out()
    for col, value in zip(row.col, row.data):
        contributions.append((str(feature_names[col]), float(value * coefs[col])))
    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions[:k]


def format_term_list(terms: Sequence[Tuple[str, float]]) -> List[str]:
    return [name for name, _ in terms]
