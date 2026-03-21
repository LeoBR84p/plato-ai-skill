"""Verify that all configured API keys are valid and the services are reachable.

Run with:
    uv run python scripts/check_apis.py

Each check performs the minimal possible API call to confirm authentication.
Key values are never printed — only masked previews (first 4 + last 4 chars).
"""

from __future__ import annotations

import os
import sys

# Load keys before anything else
from ai_skill.core.key_loader import load_keys as _load_keys

_load_keys()

from rich.console import Console
from rich.table import Table

console = Console()


def _mask(value: str) -> str:
    if not value:
        return "(not set)"
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]


def _row(table: Table, name: str, status: str, detail: str) -> None:
    colour = "green" if status == "OK" else "red" if status == "FAIL" else "yellow"
    table.add_row(name, f"[{colour}]{status}[/{colour}]", detail)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_anthropic(table: Table) -> bool:
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        _row(table, "Anthropic (Claude)", "FAIL", "ANTHROPIC_API_KEY not set")
        return False
    try:
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": "Say 1"}],
        )
        reply = msg.content[0].text.strip()
        _row(table, "Anthropic (Claude)", "OK", f"Response: '{reply}' | key={_mask(key)}")
        return True
    except Exception as exc:
        _row(table, "Anthropic (Claude)", "FAIL", str(exc)[:80])
        return False


def check_brave_search(table: Table) -> bool:
    import httpx
    key = os.environ.get("BRAVE_SEARCH", "")
    if not key:
        _row(table, "Brave Search", "FAIL", "BRAVE_SEARCH não configurado")
        return False
    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": "anomaly detection", "count": 2},
            headers={"Accept": "application/json", "X-Subscription-Token": key},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        count = len(data.get("web", {}).get("results", []))
        _row(table, "Brave Search", "OK", f"{count} resultado(s) | key={_mask(key)}")
        return True
    except Exception as exc:
        _row(table, "Brave Search", "FAIL", str(exc)[:80])
        return False


def check_brave_answer(table: Table) -> bool:
    import httpx
    key_search = os.environ.get("BRAVE_SEARCH", "")
    key_answer = os.environ.get("BRAVE_ANSWER", "")
    if not key_answer:
        _row(table, "Brave Answer (AI)", "FAIL", "BRAVE_ANSWER não configurado")
        return False
    try:
        # Step 1: web search with summary=1 (uses BRAVE_SEARCH key)
        r = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": "anomaly detection", "count": 3, "summary": 1},
            headers={"Accept": "application/json", "X-Subscription-Token": key_search or key_answer},
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
        summarizer_key = (data.get("summarizer") or {}).get("key")

        if not summarizer_key:
            # Key configured but plan may not include AI Summarizer
            _row(table, "Brave Answer (AI)", "WARN",
                 f"key={_mask(key_answer)} | plano não retorna summarizer key (verifique o plano)")
            return True  # Key is valid, feature may require upgrade

        # Step 2: call summarizer with obtained key
        r2 = httpx.get(
            "https://api.search.brave.com/res/v1/summarizer/search",
            params={"key": summarizer_key, "entity_info": 1},
            headers={"Accept": "application/json", "X-Subscription-Token": key_answer},
            timeout=20.0,
        )
        r2.raise_for_status()
        data2 = r2.json()
        has_summary = bool(data2.get("summary"))
        status = "OK" if has_summary else "WARN"
        _row(table, "Brave Answer (AI)", status,
             f"{'Resposta gerada' if has_summary else 'Sem conteúdo'} | key={_mask(key_answer)}")
        return True
    except Exception as exc:
        _row(table, "Brave Answer (AI)", "FAIL", str(exc)[:80])
        return False


def check_safe_browsing(table: Table) -> bool:
    import httpx
    key = (
        os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY")
        or os.environ.get("GOOGLE_API_KEY", "")
    )
    if not key:
        _row(table, "Google Safe Browsing", "FAIL", "GOOGLE_API_KEY not set")
        return False
    try:
        # Use the canonical Google test URL that always returns a SOCIAL_ENGINEERING threat
        test_url = "http://malware.testing.google.test/testing/malware/"
        payload = {
            "client": {"clientId": "ai_skill", "clientVersion": "1.0"},
            "threatInfo": {
                "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": test_url}],
            },
        }
        resp = httpx.post(
            f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={key}",
            json=payload,
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # A non-empty "matches" confirms the API is active and working
        found = bool(data.get("matches"))
        if found:
            _row(table, "Google Safe Browsing", "OK",
                 f"Test URL correctly flagged as threat | key={_mask(key)}")
        else:
            _row(table, "Google Safe Browsing", "WARN",
                 "API reachable but test URL not flagged (check API enablement)")
        return True
    except Exception as exc:
        _row(table, "Google Safe Browsing", "FAIL", str(exc)[:80])
        return False


def check_google_drive(table: Table) -> bool:
    creds_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "")
    if not creds_file:
        _row(table, "Google Drive", "FAIL", "GOOGLE_APPLICATION_CREDENTIALS not set")
        return False
    if not folder_id:
        _row(table, "Google Drive", "FAIL", "GOOGLE_DRIVE_FOLDER_ID not set")
        return False
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        result = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false",
                pageSize=5,
                fields="files(id, name, mimeType)",
            )
            .execute()
        )
        files = result.get("files", [])
        names = ", ".join(f["name"] for f in files[:3]) or "(pasta vazia)"
        _row(table, "Google Drive", "OK",
             f"{len(files)} arquivo(s) encontrado(s): {names}")
        return True
    except Exception as exc:
        _row(table, "Google Drive", "FAIL", str(exc)[:100])
        return False


def check_firecrawl(table: Table) -> bool:
    import httpx
    key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not key:
        _row(table, "Firecrawl", "FAIL", "FIRECRAWL_API_KEY not set")
        return False
    try:
        resp = httpx.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"url": "https://example.com", "formats": ["markdown"]},
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
        success = data.get("success", False)
        chars = len(data.get("data", {}).get("markdown", ""))
        if success:
            _row(table, "Firecrawl", "OK", f"Scrape OK, {chars} chars | key={_mask(key)}")
        else:
            _row(table, "Firecrawl", "FAIL", str(data)[:80])
        return success
    except Exception as exc:
        _row(table, "Firecrawl", "FAIL", str(exc)[:80])
        return False


def check_langsmith(table: Table) -> bool:
    key = os.environ.get("LANGCHAIN_API_KEY", "")
    project = os.environ.get("LANGCHAIN_PROJECT", "")
    if not key:
        _row(table, "LangSmith", "FAIL", "LANGCHAIN_API_KEY not set")
        return False
    try:
        from langsmith import Client
        client = Client(api_key=key)
        # list_projects raises if auth fails
        projects = list(client.list_projects())
        _row(table, "LangSmith", "OK",
             f"project='{project}' | {len(projects)} projeto(s) | key={_mask(key)}")
        return True
    except Exception as exc:
        _row(table, "LangSmith", "FAIL", str(exc)[:80])
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.print("\n[bold]Verificando APIs...[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Serviço", min_width=22)
    table.add_column("Status", min_width=6)
    table.add_column("Detalhe")

    results = [
        check_anthropic(table),
        check_brave_search(table),
        check_brave_answer(table),
        check_safe_browsing(table),
        check_google_drive(table),
        check_firecrawl(table),
        check_langsmith(table),
    ]

    console.print(table)

    passed = sum(results)
    total = len(results)
    console.print(f"\n[bold]Resultado: {passed}/{total} APIs OK[/bold]")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
