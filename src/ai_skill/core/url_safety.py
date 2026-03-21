"""URL safety pre-filter using Google Safe Browsing API v4.

Applied automatically before any external URL access. URLs flagged as
dangerous are discarded and logged. The pipeline continues without them.

References:
    https://developers.google.com/safe-browsing/v4/lookup-api
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_SAFE_BROWSING_ENDPOINT = (
    "https://safebrowsing.googleapis.com/v4/threatMatches:find"
)

_THREAT_TYPES = [
    "MALWARE",
    "SOCIAL_ENGINEERING",
    "UNWANTED_SOFTWARE",
    "POTENTIALLY_HARMFUL_APPLICATION",
]

_PLATFORM_TYPES = ["ANY_PLATFORM"]
_THREAT_ENTRY_TYPES = ["URL"]

# Google-provided test URL that always triggers MALWARE — safe for unit tests.
SAFE_BROWSING_TEST_URL = "http://malware.testing.google.test/testing/malware/"


class UrlSafetyGuard:
    """Pre-filter that checks URLs against Google Safe Browsing API v4.

    When GOOGLE_API_KEY is not set, the guard operates in passthrough mode
    (all URLs are considered safe) and emits a warning.

    Example:
        >>> guard = UrlSafetyGuard()
        >>> safe = guard.filter(["https://arxiv.org", "http://malware.example"])
        >>> # Returns only safe URLs
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the guard.

        Args:
            api_key: Google API key for Safe Browsing. Defaults to
                GOOGLE_SAFE_BROWSING_API_KEY env var, falling back to
                GOOGLE_API_KEY. If neither is set, guard operates in
                passthrough mode.
        """
        self._api_key = (
            api_key
            or os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not self._api_key:
            logger.warning(
                "GOOGLE_SAFE_BROWSING_API_KEY not set. UrlSafetyGuard running in "
                "passthrough mode — URLs will not be checked for safety."
            )

    @property
    def is_active(self) -> bool:
        """Return True when the guard has an API key and is actively checking."""
        return bool(self._api_key)

    def filter(self, urls: list[str]) -> list[str]:
        """Return only URLs that are not flagged by Safe Browsing.

        Args:
            urls: List of URLs to check.

        Returns:
            Subset of urls that passed the safety check.
            If the API is unreachable or the key is missing, all urls are returned
            (fail-open policy) and a warning is logged.
        """
        if not urls:
            return []
        if not self._api_key:
            return urls

        try:
            threats = self._check_urls(urls)
        except Exception as exc:
            logger.warning(
                "Safe Browsing check failed (%s). Proceeding with all URLs "
                "(fail-open). Configure GOOGLE_API_KEY to enable protection.",
                exc,
            )
            return urls

        safe_urls: list[str] = []
        for url in urls:
            if url in threats:
                logger.warning(
                    "URL blocked by Safe Browsing [%s]: %s",
                    ", ".join(threats[url]),
                    url,
                )
            else:
                safe_urls.append(url)

        blocked = len(urls) - len(safe_urls)
        if blocked:
            logger.info("Safe Browsing blocked %d of %d URLs.", blocked, len(urls))

        return safe_urls

    def is_safe(self, url: str) -> bool:
        """Return True if the URL is not flagged by Safe Browsing.

        Args:
            url: A single URL to check.

        Returns:
            True if the URL is considered safe (or if guard is in passthrough mode).
        """
        return url in self.filter([url])

    def _check_urls(self, urls: list[str]) -> dict[str, list[str]]:
        """Call the Safe Browsing API and return a threat map.

        Args:
            urls: URLs to send to the API.

        Returns:
            Dict mapping flagged URL → list of threat type strings.
        """
        payload: dict[str, Any] = {
            "client": {
                "clientId": "ai-skill",
                "clientVersion": "0.1.0",
            },
            "threatInfo": {
                "threatTypes": _THREAT_TYPES,
                "platformTypes": _PLATFORM_TYPES,
                "threatEntryTypes": _THREAT_ENTRY_TYPES,
                "threatEntries": [{"url": u} for u in urls],
            },
        }

        response = httpx.post(
            _SAFE_BROWSING_ENDPOINT,
            params={"key": self._api_key},
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        threats: dict[str, list[str]] = {}
        for match in data.get("matches", []):
            url: str = match["threat"]["url"]
            threat_type: str = match["threatType"]
            threats.setdefault(url, []).append(threat_type)

        return threats
