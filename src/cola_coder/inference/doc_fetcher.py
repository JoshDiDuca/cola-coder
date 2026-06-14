"""Web documentation fetcher with local caching.

Fetches documentation pages for supported frameworks (React, Next.js, Zod,
TypeORM), converts HTML to clean markdown, and caches results locally.
The cached markdown is returned wrapped in <|doc|> format tokens so it can
be inserted directly into model prompts.

Usage:
    from cola_coder.inference.doc_fetcher import DocFetcher

    fetcher = DocFetcher()
    doc = fetcher.fetch("react", "useState")
    if doc:
        prompt = doc + "\\n\\n" + user_code
"""

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

FEATURE_ENABLED = True

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known documentation URL patterns
# ---------------------------------------------------------------------------

DOC_URLS: dict[str, dict] = {
    "react": {
        "base": "https://react.dev/reference/react",
        "pages": [
            "useState",
            "useEffect",
            "useContext",
            "useReducer",
            "useCallback",
            "useMemo",
            "useRef",
            "useId",
            "useSyncExternalStore",
        ],
    },
    "nextjs": {
        "base": "https://nextjs.org/docs",
        "pages": [
            "app/api-reference",
            "app/building-your-application/routing",
        ],
    },
    "zod": {
        "base": "https://zod.dev",
        "pages": [""],  # Single-page docs
    },
    "typeorm": {
        "base": "https://typeorm.io",
        "pages": [
            "entities",
            "relations",
            "repository-api",
            "query-builder",
        ],
    },
}

# Seconds to wait between HTTP requests to avoid hammering servers
_REQUEST_DELAY = 0.5


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# DocFetcher
# ---------------------------------------------------------------------------


class DocFetcher:
    """Fetch and cache framework documentation, returning <|doc|>-formatted strings.

    Workflow for each fetch() call:
      1. Check local disk cache — return immediately if valid.
      2. Fetch from web via _fetch_url().
      3. Strip HTML to clean markdown via _html_to_markdown().
      4. Write to cache.
      5. Return as a <|doc|>-formatted string.
    """

    def __init__(self, cache_dir: str = "data/doc_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fetch(
        self,
        framework: str,
        topic: str,
        version: str | None = None,
    ) -> str | None:
        """Fetch documentation for a specific framework topic.

        Args:
            framework: Framework key — one of "react", "nextjs", "zod", "typeorm".
            topic:     Topic / page slug, e.g. "useState".
            version:   Optional version string for the <|doc|> header tag.

        Returns:
            A <|doc|>-formatted string, or None on failure.
        """
        if not FEATURE_ENABLED:
            return None

        fw = framework.lower()
        version = version or "latest"

        cache_path = self._get_cache_path(fw, topic, version)

        # ---- Cache hit ----
        if cache_path.exists() and self._is_cache_valid(cache_path):
            try:
                markdown = cache_path.read_text(encoding="utf-8")
                return self._wrap_doc(fw, topic, version, markdown)
            except OSError as exc:
                logger.warning("Could not read cache %s: %s", cache_path, exc)

        # ---- Fetch from web ----
        url = self._resolve_url(fw, topic)
        if url is None:
            logger.warning("No URL pattern for framework=%r topic=%r", fw, topic)
            return None

        logger.debug("Fetching %s", url)
        html = self._fetch_url(url)
        if html is None:
            return None

        markdown = self._html_to_markdown(html)
        if not markdown.strip():
            logger.warning("Empty markdown extracted from %s", url)
            return None

        # Defense-in-depth (OWASP LLM01): fetched docs are UNTRUSTED retrieved
        # content prepended to model prompts. Flag indirect prompt-injection
        # directives / hidden control characters so a poisoned doc is visible in
        # logs rather than silently smuggled into context. Non-blocking by design
        # (docs may legitimately discuss these phrases); the signal is the value.
        from ..security.injection_patterns import scan_injection
        hits = scan_injection(markdown)
        if hits:
            logger.warning(
                "Possible prompt injection in fetched doc %s: %s",
                url, ", ".join(hits),
            )

        # ---- Cache result ----
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(markdown, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write cache %s: %s", cache_path, exc)

        return self._wrap_doc(fw, topic, version, markdown)

    def get_relevant_docs(
        self,
        framework: str,
        version: str,
        code_context: str,
    ) -> str:
        """Given code context, find and return the most relevant documentation.

        Scans the code_context for known topic names (hooks, entity keywords,
        etc.) and fetches documentation for those topics.

        Args:
            framework:    Framework key.
            version:      Version string for the <|doc|> header.
            code_context: Source code to scan for relevant identifiers.

        Returns:
            Concatenated <|doc|>-formatted strings for all found topics,
            or an empty string if nothing is found.
        """
        if not FEATURE_ENABLED:
            return ""

        fw = framework.lower()
        if fw not in DOC_URLS:
            return ""

        pages = DOC_URLS[fw].get("pages", [])
        if not pages:
            return ""

        # Find which pages are referenced in the code
        relevant: list[str] = []
        for page in pages:
            if not page:
                continue
            # Match the last path segment as an identifier in the code
            slug = page.split("/")[-1]
            if slug and re.search(r"\b" + re.escape(slug) + r"\b", code_context):
                relevant.append(page)

        # Fallback: return the first page if nothing matched
        if not relevant and pages:
            fallback = pages[0]
            if fallback:
                relevant = [fallback]

        parts: list[str] = []
        for page in relevant[:3]:  # Cap at 3 to avoid blowing the context budget
            doc = self.fetch(fw, page, version)
            if doc:
                parts.append(doc)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _fetch_url(self, url: str) -> str | None:
        """Fetch URL content with error handling and rate limiting.

        Returns raw HTML string or None on any failure.
        """
        try:
            import urllib.request
            import urllib.error
        except ImportError:
            logger.error("urllib not available — cannot fetch docs.")
            return None

        # Rate limiting: enforce a minimum gap between requests
        now = time.monotonic()
        wait = _REQUEST_DELAY - (now - self._last_request)
        if wait > 0:
            time.sleep(wait)

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; cola-coder/1.0; "
                        "+https://github.com/cola-coder)"
                    )
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                self._last_request = time.monotonic()
                charset = "utf-8"
                content_type = resp.headers.get("Content-Type", "")
                if "charset=" in content_type:
                    charset = content_type.split("charset=")[-1].split(";")[0].strip()
                return resp.read().decode(charset, errors="replace")

        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch %s: %s", url, exc)
            self._last_request = time.monotonic()
            return None

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to clean markdown, preserving code blocks.

        Extracts the <main> or <article> element (skipping nav/footer/sidebar),
        converts headers, lists, links, and code blocks to markdown syntax, and
        strips excess whitespace.

        Falls back to stripping all tags if BeautifulSoup is unavailable.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning(
                "beautifulsoup4 not installed — returning plain text. "
                "Install with: pip install beautifulsoup4"
            )
            return self._strip_tags_fallback(html)

        soup = BeautifulSoup(html, "html.parser")

        # Remove navigation chrome
        for tag in soup.find_all(
            ["nav", "footer", "aside", "header", "script", "style", "noscript"]
        ):
            tag.decompose()
        for tag in soup.find_all(attrs={"role": ["navigation", "banner", "complementary"]}):
            tag.decompose()
        for tag in soup.find_all(class_=re.compile(r"sidebar|nav|menu|toc|breadcrumb", re.I)):
            tag.decompose()

        # Prefer the main content region
        root = soup.find("main") or soup.find("article") or soup.find("body") or soup

        lines: list[str] = []
        self._render_node(root, lines)

        text = "\n".join(lines)
        # Collapse runs of 3+ blank lines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _render_node(self, node, lines: list[str]) -> None:
        """Recursively render an HTML node tree into markdown lines."""
        try:
            from bs4 import NavigableString, Tag
        except ImportError:
            return

        if isinstance(node, NavigableString):
            text = str(node)
            if text.strip():
                lines.append(text.rstrip())
            return

        if not isinstance(node, Tag):
            return

        tag = node.name.lower() if node.name else ""

        # Headings
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(f"\n{'#' * level} {text}\n")
            return

        # Code blocks — preserve verbatim
        if tag == "pre":
            code_tag = node.find("code")
            code_text = (code_tag or node).get_text()
            # Try to detect language from class attribute
            lang = ""
            classes = (code_tag or node).get("class", [])
            for cls in classes:
                m = re.match(r"language-(\w+)", cls)
                if m:
                    lang = m.group(1)
                    break
            lines.append(f"\n```{lang}")
            lines.append(code_text.rstrip())
            lines.append("```\n")
            return

        # Inline code
        if tag == "code":
            text = node.get_text()
            lines.append(f"`{text}`")
            return

        # Paragraphs
        if tag == "p":
            parts: list[str] = []
            for child in node.children:
                if isinstance(child, NavigableString):
                    parts.append(str(child))
                elif isinstance(child, Tag):
                    inner_lines: list[str] = []
                    self._render_node(child, inner_lines)
                    parts.append("".join(inner_lines))
            text = "".join(parts).strip()
            if text:
                lines.append(f"\n{text}\n")
            return

        # Lists
        if tag in ("ul", "ol"):
            lines.append("")
            for i, item in enumerate(node.find_all("li", recursive=False)):
                prefix = f"{i + 1}." if tag == "ol" else "-"
                text = item.get_text(" ", strip=True)
                lines.append(f"{prefix} {text}")
            lines.append("")
            return

        # Links — keep the label, discard the URL
        if tag == "a":
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(text)
            return

        # Strong / em — pass through without special formatting
        if tag in ("strong", "b", "em", "i", "span"):
            for child in node.children:
                self._render_node(child, lines)
            return

        # Horizontal rule
        if tag == "hr":
            lines.append("\n---\n")
            return

        # Default: recurse into children
        for child in node.children:
            self._render_node(child, lines)

    def _strip_tags_fallback(self, html: str) -> str:
        """Fallback: strip all HTML tags and return plain text."""
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _resolve_url(self, framework: str, topic: str) -> str | None:
        """Construct a documentation URL from the DOC_URLS registry."""
        if framework not in DOC_URLS:
            return None
        spec = DOC_URLS[framework]
        base = spec["base"].rstrip("/")
        if not topic:
            return base
        return f"{base}/{topic}"

    def _get_cache_path(self, framework: str, topic: str, version: str) -> Path:
        """Return the cache file path for a specific doc page."""
        safe_topic = re.sub(r"[^\w\-.]", "_", topic) if topic else "index"
        safe_version = re.sub(r"[^\w.\-]", "_", version)
        filename = f"{safe_topic}.md"
        return self.cache_dir / framework / safe_version / filename

    def _is_cache_valid(self, cache_path: Path, ttl_days: int = 7) -> bool:
        """Return True if the cached file exists and is within the TTL."""
        if not cache_path.exists():
            return False
        try:
            mtime = cache_path.stat().st_mtime
            age_days = (
                datetime.now(timezone.utc).timestamp() - mtime
            ) / 86400
            return age_days < ttl_days
        except OSError:
            return False

    def _wrap_doc(
        self,
        framework: str,
        topic: str,
        version: str,
        markdown: str,
    ) -> str:
        """Wrap markdown content in <|doc|> context tokens."""
        label = f"{framework}@{version}"
        slug = topic.split("/")[-1] if topic else framework
        header = f"<|doc|>{label} - {slug}<|/doc|>"
        # Trim to a reasonable length to avoid blowing the prompt budget
        max_chars = 8_000
        body = markdown[:max_chars]
        if len(markdown) > max_chars:
            body += "\n... (truncated)"
        return f"{header}\n{body.strip()}\n<|eos|>"
