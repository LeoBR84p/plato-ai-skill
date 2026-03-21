"""Google Drive reference skill.

Allows the agent to consult a designated Google Drive folder as a local
reference library, supplementing online search. The folder may contain:
  - PDF files  (extracted via PyMuPDF / pypdf, same as pdf_reader skill)
  - Word files (.docx, extracted via python-docx)
  - Plain text (.txt, .md)

Authentication uses a Google Cloud service account whose credentials file
path is provided via GOOGLE_APPLICATION_CREDENTIALS (standard ADC convention)
or GOOGLE_SERVICE_ACCOUNT_FILE. The service account must be granted at minimum
"Viewer" access to the Drive folder (share the folder with the service account
email listed in GOOGLE_API_SERVICE_ACCOUNT).

Parameters accepted in SkillInput.parameters:
    query (str): Text to search within document content. Required.
    folder_id (str): Google Drive folder ID. Defaults to GOOGLE_DRIVE_FOLDER_ID
        env var.
    max_files (int): Maximum files to scan per call. Default: 20.
    max_chars_per_file (int): Character limit per document extract. Default: 8000.
    file_types (list[str]): Filter by MIME types. Default: pdf, docx, txt, md.

Environment variables:
    GOOGLE_DRIVE_FOLDER_ID          Folder to treat as reference library.
    GOOGLE_APPLICATION_CREDENTIALS  Path to service account JSON key file.
    GOOGLE_SERVICE_ACCOUNT_FILE     Alternative to GOOGLE_APPLICATION_CREDENTIALS.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from ai_skill.core.url_safety import UrlSafetyGuard
from ai_skill.skills.base import BaseSkill, SkillInput, SkillMeta, SkillOutput

logger = logging.getLogger(__name__)

# Supported MIME types and their labels
_MIME_PDF = "application/pdf"
_MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_MIME_DOC = "application/msword"
_MIME_TXT = "text/plain"
_MIME_MD = "text/markdown"
_MIME_GDOC = "application/vnd.google-apps.document"  # Google Docs → export as txt

_SUPPORTED_MIMES: set[str] = {_MIME_PDF, _MIME_DOCX, _MIME_DOC, _MIME_TXT, _MIME_MD, _MIME_GDOC}

# Google Docs are exported as plain text
_EXPORT_MIME: dict[str, str] = {
    _MIME_GDOC: "text/plain",
}


def _service_account_file() -> str | None:
    """Return the path to the service account JSON file, if configured.

    Returns:
        Path string, or None if not set.
    """
    return (
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
    )


class GoogleDriveSkill(BaseSkill):
    """Search and extract text from a Google Drive reference folder.

    The skill lists all supported files in the configured Drive folder,
    downloads each one, extracts its text, and returns the excerpts that
    contain the search query terms (case-insensitive substring match).

    This lets the agent consult the user's personal document library
    (papers, books, reports) as offline reference, complementing live web
    searches.

    Requires:
        - GOOGLE_DRIVE_FOLDER_ID env var (or parameter)
        - GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_SERVICE_ACCOUNT_FILE
          pointing to a service account JSON that has Viewer access to the folder.
    """

    SKILL_META = SkillMeta(
        name="google_drive",
        description=(
            "Search and extract text from PDF, Word, and text files stored in "
            "a designated Google Drive folder. Returns relevant excerpts matching "
            "the query. Use as offline reference library to supplement web search."
        ),
        version="1.0.0",
        author="ai_skill",
        license="MIT",
        tags=["google-drive", "reference", "pdf", "docx", "local-library", "offline"],
        dependencies=[],
    )

    def __init__(self) -> None:
        """Initialise the skill; Google API client is created lazily."""
        self._service: Any | None = None
        self._safety_guard = UrlSafetyGuard()

    def _get_service(self) -> Any:
        """Lazily build the Google Drive API service client.

        Returns:
            An authenticated ``googleapiclient.discovery.Resource`` for Drive v3.

        Raises:
            RuntimeError: If credentials are not configured.
        """
        if self._service is not None:
            return self._service

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise RuntimeError(
                "Google Drive dependencies not installed. "
                "Run: uv add google-api-python-client google-auth"
            ) from exc

        sa_file = _service_account_file()
        if not sa_file:
            raise RuntimeError(
                "Google Drive service account not configured. "
                "Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_SERVICE_ACCOUNT_FILE "
                "to the path of the service account JSON key file."
            )

        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        creds = service_account.Credentials.from_service_account_file(sa_file, scopes=scopes)
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def run(self, input: SkillInput) -> SkillOutput:
        """Search the Drive reference folder for documents matching the query.

        Args:
            input: SkillInput with parameters: query (required), folder_id,
                max_files, max_chars_per_file, file_types.

        Returns:
            SkillOutput with result containing:
                matches (list[dict]): Each item has keys: file_name, file_id,
                    mime_type, excerpt, char_count.
                total_files_scanned (int): Number of files examined.
                folder_id (str): The Drive folder that was searched.
        """
        params = input.parameters
        query: str | None = params.get("query")
        if not query or not query.strip():
            return self._error_output("Parameter 'query' is required and must not be empty.")

        folder_id: str = params.get("folder_id") or os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "")
        if not folder_id:
            return self._error_output(
                "Google Drive folder ID not configured. "
                "Pass 'folder_id' parameter or set GOOGLE_DRIVE_FOLDER_ID."
            )

        max_files: int = int(params.get("max_files", 20))
        max_chars: int = int(params.get("max_chars_per_file", 8000))

        try:
            service = self._get_service()
        except RuntimeError as exc:
            return self._error_output(str(exc))

        # List files in the folder
        try:
            files = self._list_files(service, folder_id, max_files)
        except Exception as exc:
            logger.error("Drive API list error: %s", exc)
            return self._error_output(f"Failed to list Drive folder: {exc}")

        if not files:
            return SkillOutput(
                skill_name="google_drive",
                result={
                    "matches": [],
                    "total_files_scanned": 0,
                    "folder_id": folder_id,
                    "message": "No supported files found in the Drive folder.",
                },
                confidence=0.3,
                sources=[f"https://drive.google.com/drive/folders/{folder_id}"],
                error=None,
                cached=False,
            )

        matches: list[dict[str, Any]] = []
        for file_meta in files:
            try:
                text = self._extract_text(service, file_meta, max_chars)
            except Exception as exc:
                logger.warning("Could not extract text from '%s': %s", file_meta["name"], exc)
                continue

            excerpt = _find_excerpt(text, query, context_chars=400)
            if excerpt:
                matches.append({
                    "file_name": file_meta["name"],
                    "file_id": file_meta["id"],
                    "mime_type": file_meta["mimeType"],
                    "excerpt": excerpt,
                    "char_count": len(text),
                })

        confidence = min(0.9, 0.5 + 0.1 * len(matches)) if matches else 0.3

        return SkillOutput(
            skill_name="google_drive",
            result={
                "matches": matches,
                "total_files_scanned": len(files),
                "folder_id": folder_id,
            },
            confidence=confidence,
            sources=[f"https://drive.google.com/drive/folders/{folder_id}"],
            error=None,
            cached=False,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _list_files(
        self,
        service: Any,
        folder_id: str,
        max_files: int,
    ) -> list[dict[str, Any]]:
        """Return file metadata for supported documents in the Drive folder.

        Args:
            service: Authenticated Drive v3 service resource.
            folder_id: The Drive folder ID to list.
            max_files: Maximum number of files to return.

        Returns:
            List of file metadata dicts with keys: id, name, mimeType.
        """
        mime_filter = " or ".join(
            f"mimeType='{m}'" for m in sorted(_SUPPORTED_MIMES)
        )
        q = f"'{folder_id}' in parents and ({mime_filter}) and trashed=false"

        response = (
            service.files()
            .list(
                q=q,
                pageSize=min(max_files, 100),
                fields="files(id, name, mimeType)",
                orderBy="modifiedTime desc",
            )
            .execute()
        )
        return response.get("files", [])[:max_files]

    def _extract_text(
        self,
        service: Any,
        file_meta: dict[str, Any],
        max_chars: int,
    ) -> str:
        """Download and extract plain text from a Drive file.

        Args:
            service: Authenticated Drive v3 service resource.
            file_meta: File metadata dict with keys: id, name, mimeType.
            max_chars: Maximum characters to return.

        Returns:
            Extracted text (may be truncated to max_chars).
        """
        mime = file_meta["mimeType"]
        file_id = file_meta["id"]

        if mime in _EXPORT_MIME:
            # Google Docs: use export endpoint
            export_mime = _EXPORT_MIME[mime]
            content = (
                service.files().export(fileId=file_id, mimeType=export_mime).execute()
            )
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="replace")[:max_chars]
            return str(content)[:max_chars]

        # Binary files: download and extract
        request = service.files().get_media(fileId=file_id)
        buf = io.BytesIO()

        from googleapiclient.http import MediaIoBaseDownload
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        buf.seek(0)
        data = buf.read()

        if mime == _MIME_PDF:
            return _extract_pdf_bytes(data)[:max_chars]
        if mime in (_MIME_DOCX, _MIME_DOC):
            return _extract_docx_bytes(data)[:max_chars]
        # text/plain, text/markdown
        return data.decode("utf-8", errors="replace")[:max_chars]


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def _extract_pdf_bytes(data: bytes) -> str:
    """Extract text from raw PDF bytes using PyMuPDF (or pypdf fallback).

    Args:
        data: Raw PDF file bytes.

    Returns:
        Extracted text string.
    """
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # type: ignore[no-redef]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        parts = [page.get_text(sort=True) for page in doc]
        doc.close()
        Path(tmp_path).unlink(missing_ok=True)
        return "\n\n".join(parts)

    except ImportError:
        pass

    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    except ImportError as exc:
        raise RuntimeError("No PDF library available (install pymupdf or pypdf)") from exc


def _extract_docx_bytes(data: bytes) -> str:
    """Extract text from raw .docx bytes using python-docx.

    Args:
        data: Raw .docx file bytes.

    Returns:
        Extracted text string (paragraphs joined by newlines).
    """
    try:
        import docx
    except ImportError as exc:
        raise RuntimeError(
            "python-docx not installed. Run: uv add python-docx"
        ) from exc

    doc = docx.Document(io.BytesIO(data))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def _find_excerpt(text: str, query: str, context_chars: int = 400) -> str | None:
    """Return a snippet from text that contains the query, or None if not found.

    Performs case-insensitive search. Returns up to context_chars characters
    centred around the first match.

    Args:
        text: The full document text.
        query: The search query string.
        context_chars: Characters of context to include around the match.

    Returns:
        A text excerpt, or None if query is not found.
    """
    # Simple multi-term: any individual word from the query must appear
    terms = [t for t in re.split(r"\s+", query.strip().lower()) if len(t) > 2]
    text_lower = text.lower()

    best_pos: int | None = None
    for term in terms:
        pos = text_lower.find(term)
        if pos != -1:
            best_pos = pos
            break

    if best_pos is None:
        return None

    half = context_chars // 2
    start = max(0, best_pos - half)
    end = min(len(text), best_pos + half)
    snippet = text[start:end].strip()

    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"

    return snippet


SKILL_CLASS = GoogleDriveSkill
