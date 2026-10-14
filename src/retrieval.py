"""Deterministic note retrieval (TF-IDF vectors + cosine similarity).

Why TF-IDF and not a neural embedding / vector DB: there are only 10 short notes, the
sandbox this was built in had no route to a model hub, and TF-IDF is free, offline and
bit-for-bit repeatable (no model download, no GPU, no sampling). The class exposes one
method, `rank(query)`, so a sentence-embedding backend can be swapped in without touching
the rest of the pipeline. Retrieval only PROPOSES candidates; src/notes.py decides.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class NoteRetriever:
    def __init__(self, notes_df: pd.DataFrame):
        self.ids = list(notes_df["note_id"])
        docs = (notes_df["applies_to"].str.replace("-", " ") + " " + notes_df["note"]).tolist()
        self._vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vec.fit_transform(docs)

    def rank(self, query: str) -> List[Tuple[str, float]]:
        """All notes, best first. Ties broken by note_id so order is fully deterministic."""
        q = self._vec.transform([query])
        sims = (self._matrix @ q.T).toarray().ravel()
        order = sorted(range(len(self.ids)), key=lambda i: (-round(float(sims[i]), 9), self.ids[i]))
        return [(self.ids[i], float(sims[i])) for i in order]


def build_query(route: str, week_start) -> str:
    origin, destination = route.split("-", 1)
    return f"{origin} {destination} {route.replace('-', ' ')} freight cost increase week of {week_start}"
