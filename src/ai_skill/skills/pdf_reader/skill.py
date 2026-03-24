"""PDF reader skill implementation.

Primary backend: PyMuPDF (fitz) — superior multi-column layout handling,
table detection, and formula extraction. License: AGPL-3.0.

Fallback backend: pypdf — pure Python, BSD license, weaker layout handling.
Used automatically when PyMuPDF is not available (e.g. closed-source deployments).

Parameters accepted in SkillInput.parameters:
    url (str): URL of the PDF to download. Mutually exclusive with file_path.
    file_path (str): Local file system path to the PDF.
    max_pages (int): Maximum pages to extract. Default: 0 (all pages).
    extract_metadata (bool): Include PDF metadata in result. Default: True.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import httpx

from ai_skill.core.url_safety import UrlSafetyGuard
from ai_skill.skills.base import BaseSkill, SkillInput, SkillMeta, SkillOutput

logger = logging.getLogger(__name__)

_MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024  # 50 MB safety limit


def _try_import_pymupdf() -> Any | None:
    """Attempt to import PyMuPDF. Returns the module or None."""
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24 exposes 'pymupdf' namespace
        return fitz
    except ImportError:
        pass
    try:
        import fitz  # older PyMuPDF versions
        return fitz
    except ImportError:
        return None


class PdfReaderSkill(BaseSkill):
    """Extract text and metadata from PDF documents.

    Supports both remote URLs (downloaded via httpx) and local file paths.
    PyMuPDF preserves reading order in multi-column academic papers and
    correctly handles rotated text, embedded fonts, and non-Latin scripts.

    URL inputs are validated by Google Safe Browsing before download.
    """

    SKILL_META = SkillMeta(
        name="pdf_reader",
        description=(
            "Extract text, metadata, and structure from PDF documents. "
            "Accepts a URL or local file path. Uses PyMuPDF for high-quality "
            "extraction of multi-column academic papers."
        ),
        version="1.0.0",
        author="ai_skill",
        license="MIT",
        tags=["pdf", "extraction", "text", "academic", "documents", "pymupdf"],
        dependencies=[],
    )

    def __init__(self) -> None:
        """Initialise with URL safety guard and PDF backend detection."""
        self._safety_guard = UrlSafetyGuard()
        self._fitz = _try_import_pymupdf()
        if self._fitz is None:
            logger.warning(
                "PyMuPDF not available. Falling back to pypdf. "
                "Multi-column layout accuracy may be reduced."
            )

    def run(self, input: SkillInput) -> SkillOutput:
        """Extract text from a PDF document.

        Args:
            input: SkillInput with parameters: url or file_path (one required),
                max_pages, extract_metadata.

        Returns:
            SkillOutput with result containing:
                text (str): Extracted text content.
                pages (int): Number of pages processed.
                metadata (dict): PDF metadata (if extract_metadata is True).
                backend (str): "pymupdf" or "pypdf".
        """
        params = input.parameters
        url: str | None = params.get("url")
        file_path: str | None = params.get("file_path")

        if not url and not file_path:
            return self._error_output(
                "Either 'url' or 'file_path' parameter is required."
            )

        max_pages: int = int(params.get("max_pages", 0))
        extract_metadata: bool = bool(params.get("extract_metadata", True))

        pdf_path: Path | None = None
        tmp_file: tempfile.NamedTemporaryFile | None = None  # type: ignore[type-arg]

        try:
            if url:
                if not self._safety_guard.is_safe(url):
                    return self._error_output(
                        f"URL blocked by Safe Browsing: {url}"
                    )
                try:
                    pdf_path, tmp_file = self._download_pdf(url)
                except Exception as dl_exc:
                    # Binary download failed — try text extraction via API fallbacks
                    logger.warning(
                        "pdf_reader: binary download failed for %s (%s). "
                        "Trying Firecrawl / Exa / Tavily text extraction.",
                        url, dl_exc,
                    )
                    text = self._fetch_text_via_apis(url)
                    if text:
                        return SkillOutput(
                            skill_name="pdf_reader",
                            result={
                                "text": text,
                                "pages": 0,
                                "total_pages": 0,
                                "metadata": {},
                                "backend": "api_text_fallback",
                            },
                            confidence=0.6,
                            sources=[url],
                            error=None,
                            cached=False,
                        )
                    return self._error_output(
                        f"PDF download failed and all text-extraction fallbacks "
                        f"exhausted for {url}. Original error: {dl_exc}"
                    )
            else:
                pdf_path = Path(file_path)  # type: ignore[arg-type]
                if not pdf_path.exists():
                    return self._error_output(
                        f"File not found: {file_path}"
                    )

            if self._fitz is not None:
                result = self._extract_with_pymupdf(
                    pdf_path, max_pages, extract_metadata
                )
                backend = "pymupdf"
            else:
                result = self._extract_with_pypdf(
                    pdf_path, max_pages, extract_metadata
                )
                backend = "pypdf"

            result["backend"] = backend
            source_ref = url or str(file_path)

            return SkillOutput(
                skill_name="pdf_reader",
                result=result,
                confidence=0.9 if result.get("text") else 0.3,
                sources=[source_ref],
                error=None,
                cached=False,
            )

        except Exception as exc:
            logger.error("PDF extraction failed: %s", exc)
            return self._error_output(str(exc))
        finally:
            if tmp_file is not None:
                try:
                    tmp_file.close()
                except Exception:
                    pass

    def _download_pdf(
        self, url: str
    ) -> tuple[Path, "tempfile.NamedTemporaryFile[bytes]"]:
        """Download a PDF from a URL to a temporary file.

        Args:
            url: The URL to download.

        Returns:
            Tuple of (Path to temp file, the NamedTemporaryFile object to close later).

        Raises:
            ValueError: If the download exceeds _MAX_DOWNLOAD_SIZE or the
                server returns non-PDF content.
            httpx.HTTPError: On network errors.
        """
        _UA = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        downloaded = 0

        with httpx.stream(
            "GET", url,
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": _UA, "Accept": "application/pdf,*/*"},
        ) as response:
            response.raise_for_status()
            ct = response.headers.get("content-type", "").lower()
            # Bail early if the server returns HTML instead of PDF
            if "html" in ct and "pdf" not in ct:
                tmp.close()
                raise ValueError(
                    f"Server returned HTML instead of PDF "
                    f"(Content-Type: {ct}). URL may be a landing page."
                )
            for chunk in response.iter_bytes(chunk_size=65536):
                downloaded += len(chunk)
                if downloaded > _MAX_DOWNLOAD_SIZE:
                    tmp.close()
                    raise ValueError(
                        f"PDF download exceeds {_MAX_DOWNLOAD_SIZE // (1024*1024)} MB limit."
                    )
                tmp.write(chunk)

        tmp.flush()
        return Path(tmp.name), tmp

    @staticmethod
    def _fetch_text_via_apis(url: str) -> str:
        """Attempt to extract PDF text via Firecrawl, Exa, or Tavily.

        Used as a fallback when the binary PDF download fails (bot-blocking,
        auth redirect, etc.). Returns plain text or empty string.
        """
        import os

        # ── Firecrawl ─────────────────────────────────────────────
        fc_key = os.environ.get("FIRECRAWL_API_KEY", "")
        if fc_key:
            try:
                from firecrawl import FirecrawlApp  # type: ignore[import-untyped]
                app = FirecrawlApp(api_key=fc_key)
                scrape_fn = getattr(app, "scrape", None) or getattr(app, "scrape_url")
                result = scrape_fn(url, formats=["markdown"])
                md = (getattr(result, "markdown", None) or
                      (result.get("markdown", "") if isinstance(result, dict) else ""))
                if md:
                    logger.debug("pdf_reader: Firecrawl text fallback OK for %s (%d chars)", url, len(md))
                    return md
            except Exception as exc:
                logger.debug("pdf_reader: Firecrawl fallback failed for %s: %s", url, exc)

        # ── Exa get_contents ──────────────────────────────────────
        exa_key = os.environ.get("EXA_API_KEY", "")
        if exa_key:
            try:
                from exa_py import Exa  # type: ignore[import-untyped]
                exa = Exa(api_key=exa_key)
                res = exa.get_contents([url], text={"max_characters": 10000})
                items = getattr(res, "results", [])
                if items:
                    text = getattr(items[0], "text", "") or ""
                    if text:
                        logger.debug("pdf_reader: Exa text fallback OK for %s (%d chars)", url, len(text))
                        return text
            except Exception as exc:
                logger.debug("pdf_reader: Exa fallback failed for %s: %s", url, exc)

        # ── Tavily extract ────────────────────────────────────────
        tv_key = os.environ.get("TAVILY_API_KEY", "")
        if tv_key:
            try:
                from tavily import TavilyClient  # type: ignore[import-untyped]
                client = TavilyClient(api_key=tv_key)
                resp = client.extract(urls=[url])
                results = resp.get("results", [])
                if results:
                    text = results[0].get("raw_content", "") or ""
                    if text:
                        logger.debug("pdf_reader: Tavily text fallback OK for %s (%d chars)", url, len(text))
                        return text
            except Exception as exc:
                logger.debug("pdf_reader: Tavily fallback failed for %s: %s", url, exc)

        return ""

    def _extract_with_pymupdf(
        self,
        path: Path,
        max_pages: int,
        extract_metadata: bool,
    ) -> dict[str, Any]:
        """Extract text and metadata using PyMuPDF.

        Args:
            path: Path to the PDF file.
            max_pages: Maximum pages to process (0 = all).
            extract_metadata: Whether to include PDF metadata.

        Returns:
            Dict with keys: text, pages, metadata.
        """
        fitz = self._fitz
        doc = fitz.open(str(path))
        num_pages = len(doc)
        pages_to_process = num_pages if max_pages == 0 else min(max_pages, num_pages)

        text_parts: list[str] = []
        for page_num in range(pages_to_process):
            page = doc[page_num]
            # sort=True preserves reading order in multi-column layouts
            text_parts.append(page.get_text(sort=True))

        metadata: dict[str, Any] = {}
        if extract_metadata:
            raw_meta = doc.metadata or {}
            metadata = {k: v for k, v in raw_meta.items() if v}

        doc.close()

        return {
            "text": "\n\n".join(text_parts),
            "pages": pages_to_process,
            "total_pages": num_pages,
            "metadata": metadata,
        }

    def _extract_with_pypdf(
        self,
        path: Path,
        max_pages: int,
        extract_metadata: bool,
    ) -> dict[str, Any]:
        """Extract text and metadata using pypdf (fallback).

        Args:
            path: Path to the PDF file.
            max_pages: Maximum pages to process (0 = all).
            extract_metadata: Whether to include PDF metadata.

        Returns:
            Dict with keys: text, pages, metadata.
        """
        try:
            import pypdf
        except ImportError as exc:
            raise ImportError(
                "pypdf fallback requires: uv add pypdf"
            ) from exc

        text_parts: list[str] = []
        metadata: dict[str, Any] = {}

        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            num_pages = len(reader.pages)
            pages_to_process = num_pages if max_pages == 0 else min(max_pages, num_pages)

            for i in range(pages_to_process):
                page_text = reader.pages[i].extract_text() or ""
                text_parts.append(page_text)

            if extract_metadata and reader.metadata:
                for key, value in reader.metadata.items():
                    clean_key = key.lstrip("/")
                    metadata[clean_key] = str(value)

        return {
            "text": "\n\n".join(text_parts),
            "pages": pages_to_process,
            "total_pages": num_pages,
            "metadata": metadata,
        }


SKILL_CLASS = PdfReaderSkill
