"""Lightweight BM25-based RAG index for PDF context retrieval.

Stores per-namespace index files and .md extracts in attachments/.
Zero external dependencies — uses only Python stdlib (math, re, json, collections).

NAMESPACE DESIGN
----------------
The same PDF can be extracted multiple times with different "lenses" — one per
pipeline stage that needs to read the research corpus:

    CP3 (research_design)  → namespace "cp3_methodology"
        Extracts: method, hypotheses, variables, instruments, validity, ethics,
        timeline — whatever is needed to PROPOSE a research design.

    CP4 (data_collection)  → namespace "cp4_collection"   [future]
        Extracts: data sources, collection protocols, sampling, instruments,
        storage formats — whatever is needed to PLAN data collection.

    CP5 (analysis)         → namespace "cp5_analysis"     [future]
        Extracts: statistical methods, models, software, validation approaches —
        whatever is needed to PLAN the analysis.

Each namespace produces:
  - A separate .md extract per PDF: ``{stem}__{namespace}.md``
  - A separate BM25 index file:     ``.rag_index__{namespace}.json``

This keeps views fully isolated — a CP3 methodology search never returns
a CP4 data-collection extract, and vice-versa.

BM25 NOTES
----------
BM25 (Best Match 25) is the ranking function used by Elasticsearch and Lucene.
Superior to TF-IDF because it:
  - Saturates term frequency (diminishing returns for repeated terms).
  - Normalises for document length (long docs don't dominate by size alone).
  - Handles unseen query terms gracefully (zero score, no crash).

Typical throughput: < 5 ms per query for 30 docs × 500 words each.

Usage:
    rag = RagIndex(attachments_dir, namespace="cp3_methodology")
    rag.add("paper_A", "quasi-experimental study hypothesis H1 …")
    results = rag.search("hypothesis variables measurement scale", top_k=5)
    for doc_id, excerpt, score in results:
        print(doc_id, score)
    # Cache .md path for a given PDF stem:
    md_path = rag.md_path_for("paper_A")   # → attachments/paper_A__cp3_methodology.md
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# BM25 hyper-parameters (Robertson & Zaragoza 2009 recommended defaults)
_BM25_K1 = 1.5   # term-frequency saturation — 1.2–2.0
_BM25_B = 0.75   # length normalisation — 0.0 (off) to 1.0 (full)

# Maximum characters stored per document in the index JSON
# (~5000 chars ≈ ~1250 tokens — enough for BM25, avoids huge files)
_MAX_STORED_CHARS = 5000

# Separator between PDF stem and namespace in filenames
_NS_SEP = "__"


def _tokenise(text: str) -> list[str]:
    """Lower-case, strip punctuation, split on whitespace.

    Returns:
        List of lowercase word tokens (length >= 2).
    """
    text = text.lower()
    text = re.sub(r"[^a-záàâãéèêíïóôõöúûüçñ\s]", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if len(t) > 1]


class RagIndex:
    """Persistent, namespace-isolated BM25 index over PDF .md extracts.

    One ``RagIndex`` instance corresponds to exactly one (attachments_dir,
    namespace) pair.  Documents added here are never visible to an index
    created with a different namespace.

    Files on disk:
        attachments/{stem}__{namespace}.md   — extracted text for each PDF
        attachments/.rag_index__{namespace}.json — BM25 index (auto-updated)

    Args:
        attachments_dir: Path to the attachments/ directory.
        namespace: Identifier for this extraction context, e.g.
            ``"cp3_methodology"``.  Must be a valid filename component
            (no slashes, no spaces).
    """

    def __init__(self, attachments_dir: Path, namespace: str) -> None:
        if not namespace or _NS_SEP in namespace:
            raise ValueError(
                f"namespace must be non-empty and must not contain '{_NS_SEP}': {namespace!r}"
            )
        self.attachments_dir = attachments_dir
        self.namespace = namespace
        self.index_path = attachments_dir / f".rag_index{_NS_SEP}{namespace}.json"
        # doc_id → {"text": str, "tokens": list[str]}
        self._store: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def md_path_for(self, pdf_stem: str) -> Path:
        """Return the .md path for a given PDF stem under this namespace.

        Args:
            pdf_stem: Filename stem of the PDF (without extension).

        Returns:
            ``attachments/{pdf_stem}__{namespace}.md``
        """
        return self.attachments_dir / f"{pdf_stem}{_NS_SEP}{self.namespace}.md"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, doc_id: str, text: str) -> None:
        """Add or replace a document in this namespace's index.

        Args:
            doc_id: Unique identifier — typically the PDF stem.
            text: Full document text (the .md extract content).
        """
        tokens = _tokenise(text)
        self._store[doc_id] = {
            "text": text[:_MAX_STORED_CHARS],
            "tokens": tokens,
        }
        self._save()
        logger.debug(
            "RagIndex[%s]: indexed '%s' (%d tokens)", self.namespace, doc_id, len(tokens)
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[str, str, float]]:
        """Return top-k documents matching *query* using BM25 scoring.

        Only documents in this namespace are considered.

        Args:
            query: Free-text query string.
            top_k: Maximum number of results to return.

        Returns:
            List of ``(doc_id, text_excerpt, bm25_score)`` sorted by score
            descending.  ``text_excerpt`` is up to 1000 chars.
            Returns an empty list when the index is empty or no terms match.
        """
        if not self._store:
            return []
        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        n_docs = len(self._store)
        avg_dl = sum(len(v["tokens"]) for v in self._store.values()) / n_docs

        # Document-frequency per term
        df: Counter[str] = Counter()
        for doc in self._store.values():
            for t in set(doc["tokens"]):
                df[t] += 1

        scores: dict[str, float] = {}
        for doc_id, doc in self._store.items():
            dl = len(doc["tokens"])
            tf_map = Counter(doc["tokens"])
            score = 0.0
            for qt in query_tokens:
                if qt not in df:
                    continue
                idf = math.log((n_docs - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0)
                tf = tf_map.get(qt, 0)
                tf_norm = (
                    tf * (_BM25_K1 + 1)
                    / (tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avg_dl))
                )
                score += idf * tf_norm
            scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in ranked[:top_k]:
            if score <= 0:
                break
            results.append((doc_id, self._store[doc_id]["text"][:1000], score))
        return results

    def contains(self, doc_id: str) -> bool:
        """Return True if *doc_id* is already indexed in this namespace."""
        return doc_id in self._store

    def size(self) -> int:
        """Return number of indexed documents in this namespace."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            self.index_path.write_text(
                json.dumps(self._store, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(
                "RagIndex[%s]: could not save to %s: %s", self.namespace, self.index_path, exc
            )

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            self._store = json.loads(self.index_path.read_text(encoding="utf-8"))
            logger.debug(
                "RagIndex[%s]: loaded %d docs from %s",
                self.namespace, len(self._store), self.index_path,
            )
        except Exception as exc:
            logger.warning(
                "RagIndex[%s]: could not load %s: %s — starting fresh",
                self.namespace, self.index_path, exc,
            )
            self._store = {}
