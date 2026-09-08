"""Lightweight in-memory retrieval over the support knowledge base (RAG).

A full vector database is overkill for eight short articles, so this module
implements a small TF-IDF retriever with pure Python: documents are tokenised
once at import, queries are scored by IDF-weighted term overlap, and the best
matches are returned. The interface (`search`) is deliberately the same shape a
FAISS/Chroma retriever would expose, so swapping in embeddings later is a
one-file change.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common words carry no retrieval signal and would blur the ranking.
_STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from how i if in is it my me of on or
    our so that the their there they this to was what when where which who why will with
    you your""".split()
)

# Matches scoring below this fraction of the best hit are dropped as noise.
# Genuine secondary matches land around 35-60% of the top score; incidental
# word overlap lands below 15%, so a quarter separates them cleanly.
_RELATIVE_SCORE_CUTOFF = 0.25


def _singularize(token: str) -> str:
    """Crude plural stripping so "plans" and "plan" match.

    Not a real stemmer, but applied to documents and queries alike it costs
    four lines and fixes the most common miss in a keyword index.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on word characters, drop stopwords and singularize."""
    return [
        _singularize(token)
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    ]


@dataclass(frozen=True)
class Document:
    """One knowledge-base article."""

    id: str
    title: str
    content: str
    tags: tuple[str, ...]

    def as_context(self) -> str:
        """Render the article the way it is shown to the model."""
        return f"[{self.id}] {self.title}\n{self.content}"


@dataclass(frozen=True)
class SearchResult:
    """A retrieved article plus its relevance score."""

    document: Document
    score: float


class KnowledgeBase:
    """A tiny TF-IDF search index over support documentation."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self._term_frequencies: list[Counter[str]] = []
        self._idf: dict[str, float] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Compute per-document term counts and corpus-wide IDF weights."""
        document_frequency: Counter[str] = Counter()
        for document in self._documents:
            # Title and tags repeat so they weigh more than body prose.
            tokens = (
                _tokenize(document.title) * 3
                + [t for tag in document.tags for t in _tokenize(tag)] * 3
                + _tokenize(document.content)
            )
            counts = Counter(tokens)
            self._term_frequencies.append(counts)
            document_frequency.update(counts.keys())

        total = max(len(self._documents), 1)
        self._idf = {
            term: math.log((1 + total) / (1 + freq)) + 1.0
            for term, freq in document_frequency.items()
        }

    @property
    def documents(self) -> list[Document]:
        """All indexed documents."""
        return list(self._documents)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Return up to `top_k` articles ranked by IDF-weighted term overlap."""
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        results: list[SearchResult] = []
        for document, counts in zip(self._documents, self._term_frequencies, strict=True):
            length = sum(counts.values()) or 1
            score = sum(
                (counts[term] / length) * self._idf.get(term, 0.0)
                for term in query_terms
                if term in counts
            )
            # Normalise by query length so short and long questions compare.
            score = score / math.sqrt(len(query_terms))
            if score > 0:
                results.append(SearchResult(document=document, score=round(score, 6)))

        results.sort(key=lambda result: result.score, reverse=True)
        best = results[0].score if results else 0.0
        # Drop weak tail matches: they only add noise to the model's context
        # and would be reported as `sources` the answer never actually used.
        cutoff = max(_RELATIVE_SCORE_CUTOFF * best, 1e-6)
        return [result for result in results[:top_k] if result.score >= cutoff]


def _load_documents(path: Path) -> list[Document]:
    """Read and validate the knowledge-base JSON file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not load knowledge base from %s", path)
        return []

    documents: list[Document] = []
    for entry in raw:
        try:
            documents.append(
                Document(
                    id=entry["id"],
                    title=entry["title"],
                    content=entry["content"],
                    tags=tuple(entry.get("tags", [])),
                )
            )
        except (KeyError, TypeError):
            logger.warning("Skipping malformed knowledge-base entry: %r", entry)
    return documents


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    """Return the process-wide knowledge base, built on first use."""
    documents = _load_documents(DATA_DIR / "support_docs.json")
    logger.info("Knowledge base loaded with %d documents", len(documents))
    return KnowledgeBase(documents)
