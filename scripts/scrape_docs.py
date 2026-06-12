"""Documentation scraper for Cola-Coder training data.

Fetches official framework documentation pages, strips navigation/footers/
sidebars, converts the main content to clean Markdown, and saves individual
.md files under data/docs/{framework}/{version}/.  A manifest.json is written
alongside the scraped files.

Supported frameworks: react, nextjs, zod, typeorm
(extend FRAMEWORK_CONFIG below to add more).

Usage:
    python scripts/scrape_docs.py --framework react --version 18
    python scripts/scrape_docs.py --framework nextjs --version 14
    python scripts/scrape_docs.py --all
    python scripts/scrape_docs.py --list
    python scripts/scrape_docs.py --framework zod --version 3 --output-dir /tmp/docs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

# ---------------------------------------------------------------------------
# Project path bootstrap (same pattern as other scripts)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from cola_coder.cli import cli
except ImportError:
    # Minimal fallback so the script can still warn the user
    class _FallbackCLI:  # type: ignore[no-redef]
        def header(self, *a: object, **kw: object) -> None:
            print(" ".join(str(x) for x in a))

        def info(self, k: object, v: object) -> None:
            print(f"  {k}: {v}")

        def success(self, m: object) -> None:
            print(f"[OK] {m}")

        def warn(self, m: object) -> None:
            print(f"[WARN] {m}")

        def error(self, m: object, hint: str = "") -> None:
            print(f"[ERR] {m}" + (f"\n  {hint}" if hint else ""))

        def fatal(self, m: object, hint: str = "") -> None:
            self.error(m, hint)
            sys.exit(1)

        def step(self, cur: int, tot: int, msg: object) -> None:
            print(f"\nStep {cur}/{tot} · {msg}")

        def dim(self, m: object) -> None:
            print(f"  {m}")

        def done(self, m: object, extras: dict | None = None) -> None:
            print(f"\n[DONE] {m}")
            for k, v in (extras or {}).items():
                print(f"  {k}: {v}")

    cli = _FallbackCLI()  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Optional dependency check
# ---------------------------------------------------------------------------

try:
    import requests
    from bs4 import BeautifulSoup, Tag
except ImportError:
    cli.fatal(
        "Missing dependencies: requests and beautifulsoup4",
        hint="pip install requests beautifulsoup4",
    )

# ---------------------------------------------------------------------------
# Framework configuration
# ---------------------------------------------------------------------------

#: Each entry maps a framework key to a list of seed URLs that the scraper
#: will visit.  Add more entries here to extend supported frameworks.
FRAMEWORK_CONFIG: dict[str, dict] = {
    "react": {
        "display_name": "React",
        "default_version": "18",
        "seed_urls": [
            "https://react.dev/reference/react",
            "https://react.dev/reference/react-dom",
            "https://react.dev/reference/react-dom/components",
            "https://react.dev/learn",
        ],
        "base_domain": "react.dev",
        "version_tag_template": "react@{version}.2.0",
    },
    "nextjs": {
        "display_name": "Next.js",
        "default_version": "14",
        "seed_urls": [
            "https://nextjs.org/docs",
            "https://nextjs.org/docs/app",
            "https://nextjs.org/docs/app/api-reference",
            "https://nextjs.org/docs/pages/api-reference",
        ],
        "base_domain": "nextjs.org",
        "version_tag_template": "nextjs@{version}.0.0",
    },
    "zod": {
        "display_name": "Zod",
        "default_version": "3",
        "seed_urls": [
            "https://zod.dev",
        ],
        "base_domain": "zod.dev",
        "version_tag_template": "zod@{version}.0.0",
    },
    "typeorm": {
        "display_name": "TypeORM",
        "default_version": "0.3",
        "seed_urls": [
            "https://typeorm.io",
            "https://typeorm.io/#/entities",
            "https://typeorm.io/#/repository-api",
            "https://typeorm.io/#/select-query-builder",
        ],
        "base_domain": "typeorm.io",
        "version_tag_template": "typeorm@{version}.0",
    },
}

# ---------------------------------------------------------------------------
# HTML → Markdown helpers
# ---------------------------------------------------------------------------

#: CSS selectors for elements to strip before extracting main content.
_NOISE_SELECTORS: list[str] = [
    "nav",
    "header",
    "footer",
    "aside",
    "[role='navigation']",
    "[role='banner']",
    "[role='contentinfo']",
    ".sidebar",
    ".nav",
    ".toc",
    ".table-of-contents",
    "#sidebar",
    "#nav",
    "#toc",
    ".cookie-banner",
    ".announcement",
    "script",
    "style",
    "noscript",
    "iframe",
]

#: CSS selectors tried in order to locate the main content region.
_MAIN_SELECTORS: list[str] = [
    "main",
    "article",
    "[role='main']",
    ".docs-content",
    ".content",
    "#content",
    ".documentation",
    ".markdown-body",
    ".prose",
]


def _strip_noise(soup: BeautifulSoup) -> None:
    """Remove navigation, sidebars, and other boilerplate in-place."""
    for selector in _NOISE_SELECTORS:
        for el in soup.select(selector):
            el.decompose()


def _find_main(soup: BeautifulSoup) -> Tag | None:
    """Return the best candidate element for the main page content."""
    for selector in _MAIN_SELECTORS:
        el = soup.select_one(selector)
        if el:
            return el  # type: ignore[return-value]
    return soup.body  # type: ignore[return-value]


def _element_to_markdown(el: Tag, base_url: str = "") -> str:  # noqa: C901
    """Convert a BeautifulSoup element tree to Markdown.

    Handles headings (h1–h6), paragraphs, code blocks, inline code, links,
    lists (ul/ol), and horizontal rules.  Non-semantic wrapper elements are
    recursed into transparently.

    Args:
        el:       The root element to convert.
        base_url: Used to resolve relative hrefs in anchor tags.

    Returns:
        A Markdown string with normalised whitespace.
    """
    lines: list[str] = []

    def _walk(node: Tag, depth: int = 0) -> None:  # noqa: C901
        name = getattr(node, "name", None)
        if name is None:
            # NavigableString — just append the text
            text = str(node).strip()
            if text:
                lines.append(text)
            return

        # ── Block-level headings ────────────────────────────────────────
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(f"\n{'#' * level} {text}\n")
            return

        # ── Paragraphs ──────────────────────────────────────────────────
        if name == "p":
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(f"\n{text}\n")
            return

        # ── Fenced code blocks ──────────────────────────────────────────
        if name == "pre":
            code_el = node.find("code")
            # The highlight-language class lives on EITHER <pre> or <code>
            # depending on the highlighter (Prism puts it on <code>, but
            # highlight.js/MDX/Docusaurus often put it on <pre>). Check both,
            # and accept the `language-x` and shorthand `lang-x` forms — else
            # the example ships with an UNTAGGED fence and the model can't tell
            # what language it is (DATA-039).
            classes = " ".join(node.get("class", []) or [])  # type: ignore[arg-type]
            if code_el:
                classes += " " + " ".join(code_el.get("class", []) or [])  # type: ignore[arg-type]
            m = re.search(r"(?:language|lang)-(\w+)", classes)
            lang = m.group(1) if m else ""
            code_text = (code_el or node).get_text()
            lines.append(f"\n```{lang}\n{code_text}\n```\n")
            return

        # ── Inline code ─────────────────────────────────────────────────
        if name == "code":
            # Standalone <code> outside a <pre> — keep inline
            text = node.get_text()
            if text:
                lines.append(f"`{text}`")
            return

        # ── Lists (ul / ol) ─────────────────────────────────────────────
        if name in ("ul", "ol"):
            ordered = name == "ol"
            lines.append("")
            counter = 1
            for child in node.children:
                if getattr(child, "name", None) != "li":
                    continue
                prefix = f"{counter}. " if ordered else "- "
                # A <pre> code block inside a list item (common in step-by-step
                # guides) must NOT be flattened into the bullet text — that
                # collapses its indentation/newlines into one space-joined line
                # of broken code (DATA-040). Detach the code blocks first so the
                # bullet gets only the description, then render each block as a
                # proper fence (with DATA-039 language detection via _walk).
                pres = child.find_all("pre")  # type: ignore[union-attr]
                for p in pres:
                    p.extract()
                desc = child.get_text(" ", strip=True)  # type: ignore[union-attr]
                emitted = False
                if desc:
                    lines.append(f"{prefix}{desc}")
                    emitted = True
                for p in pres:
                    _walk(p)
                    emitted = True
                if emitted and ordered:
                    counter += 1
            lines.append("")
            return

        # ── Horizontal rule ─────────────────────────────────────────────
        if name == "hr":
            lines.append("\n---\n")
            return

        # ── Anchor tags — preserve link text, optionally absolute href ──
        if name == "a":
            text = node.get_text(" ", strip=True)
            href = node.get("href", "")
            if href and base_url:
                href = urljoin(base_url, str(href))
            if text:
                if href:
                    lines.append(f"[{text}]({href})")
                else:
                    lines.append(text)
            return

        # ── Block-level containers — recurse ────────────────────────────
        if name in (
            "div", "section", "article", "main", "header", "footer",
            "aside", "span", "strong", "em", "b", "i", "td", "th",
            "tr", "tbody", "thead", "table",
        ):
            for child in node.children:
                _walk(child, depth + 1)  # type: ignore[arg-type]
            return

        # ── Everything else — just recurse ──────────────────────────────
        for child in node.children:
            _walk(child, depth + 1)  # type: ignore[arg-type]

    _walk(el)

    # Collapse consecutive blank lines and normalise whitespace
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_markdown(html: str, base_url: str = "") -> str:
    """Full pipeline: parse HTML → strip noise → extract main → convert.

    Args:
        html:     Raw HTML string.
        base_url: Page URL used to resolve relative links.

    Returns:
        Clean Markdown string.
    """
    soup = BeautifulSoup(html, "html.parser")
    _strip_noise(soup)
    main_el = _find_main(soup)
    if main_el is None:
        return ""
    return _element_to_markdown(main_el, base_url=base_url)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; cola-coder-docs-scraper/1.0; "
        "+https://github.com/cola-coder)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch(url: str, session: requests.Session, timeout: int = 30) -> str | None:
    """Fetch a URL and return its HTML, or None on failure.

    Args:
        url:     Full URL to fetch.
        session: Requests session to use (handles connection pooling).
        timeout: Request timeout in seconds.

    Returns:
        HTML string, or None if the request failed.
    """
    try:
        response = session.get(url, timeout=timeout, headers=_DEFAULT_HEADERS)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 429:
            cli.warn(f"Rate limited (429) on {url} — waiting 30 s")
            time.sleep(30)
            try:
                response = session.get(url, timeout=timeout, headers=_DEFAULT_HEADERS)
                response.raise_for_status()
                return response.text
            except requests.RequestException as retry_exc:
                cli.warn(f"Retry failed for {url}: {retry_exc}")
                return None
        cli.warn(f"HTTP {status} fetching {url}: {exc}")
        return None
    except requests.exceptions.ConnectionError as exc:
        cli.warn(f"Connection error for {url}: {exc}")
        return None
    except requests.exceptions.Timeout:
        cli.warn(f"Timeout fetching {url}")
        return None
    except requests.RequestException as exc:
        cli.warn(f"Request failed for {url}: {exc}")
        return None


def _collect_internal_links(
    html: str,
    base_url: str,
    base_domain: str,
    *,
    max_links: int = 100,
) -> list[str]:
    """Extract internal links from an HTML page.

    Only returns links whose hostname matches base_domain and whose path
    starts with the same prefix as the base_url path.

    Args:
        html:        Raw page HTML.
        base_url:    The URL this page was fetched from.
        base_domain: Hostname to restrict links to.
        max_links:   Upper bound on returned links.

    Returns:
        Deduplicated list of absolute URLs.
    """
    soup = BeautifulSoup(html, "html.parser")
    base_path = urlparse(base_url).path.rstrip("/")
    seen: set[str] = set()
    links: list[str] = []

    for tag in soup.find_all("a", href=True):
        href = str(tag["href"]).split("#")[0].strip()  # drop fragment
        if not href:
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)

        # Must be same domain and path must be under base
        if parsed.netloc != base_domain:
            continue
        if not parsed.path.startswith(base_path):
            continue
        # Normalise trailing slash
        url_clean = absolute.rstrip("/")
        if url_clean not in seen:
            seen.add(url_clean)
            links.append(url_clean)
            if len(links) >= max_links:
                break

    return links


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def _url_to_slug(url: str) -> str:
    """Convert a URL path to a filesystem-safe slug.

    Example: https://react.dev/reference/react/useState → useState
    """
    path = urlparse(url).path.strip("/")
    # Replace slashes and non-word chars with underscores
    slug = re.sub(r"[^\w\-]", "_", path)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "index"


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

def scrape_framework(
    framework: str,
    version: str,
    output_dir: Path,
    *,
    delay: float = 1.0,
    max_pages: int = 200,
) -> list[dict]:
    """Scrape documentation for a single framework and save to disk.

    Args:
        framework:  Framework key from FRAMEWORK_CONFIG.
        version:    Version string (e.g. "18", "14", "3").
        output_dir: Root output directory (files go in output_dir/framework/version/).
        delay:      Seconds to sleep between requests (be polite).
        max_pages:  Maximum number of pages to scrape per framework.

    Returns:
        List of manifest entry dicts (one per saved file).
    """
    if framework not in FRAMEWORK_CONFIG:
        cli.error(
            f"Unknown framework: {framework!r}",
            hint=f"Supported: {', '.join(FRAMEWORK_CONFIG)}",
        )
        return []

    config = FRAMEWORK_CONFIG[framework]
    display_name = config["display_name"]
    base_domain: str = config["base_domain"]
    seed_urls: list[str] = config["seed_urls"]
    version_tag = config["version_tag_template"].format(version=version)

    save_dir = output_dir / framework / version
    save_dir.mkdir(parents=True, exist_ok=True)

    cli.step(1, 3, f"Scraping {display_name} {version}")
    cli.info("Output dir", str(save_dir))
    cli.info("Seed URLs", len(seed_urls))

    session = requests.Session()
    session.headers.update(_DEFAULT_HEADERS)

    visited: set[str] = set()
    queue: list[str] = list(seed_urls)
    manifest_entries: list[dict] = []

    page_count = 0

    while queue and page_count < max_pages:
        url = queue.pop(0)
        url_clean = url.rstrip("/")
        if url_clean in visited:
            continue
        visited.add(url_clean)

        cli.dim(f"  [{page_count + 1}] {url}")

        html = _fetch(url, session)
        if html is None:
            time.sleep(delay)
            continue

        # Discover more links from this page (only from seed pages to avoid
        # crawling the entire site — just their sub-trees)
        if page_count < len(seed_urls):
            sub_links = _collect_internal_links(
                html, url, base_domain, max_links=50,
            )
            for link in sub_links:
                if link.rstrip("/") not in visited and link not in queue:
                    queue.append(link)

        # Convert to Markdown
        markdown = html_to_markdown(html, base_url=url)
        if not markdown.strip():
            cli.warn(f"Empty content at {url} — skipping")
            time.sleep(delay)
            continue

        # Prepend version tag header
        header = f"// Framework: {version_tag}\n// Source: {url}\n\n"
        full_content = header + markdown

        # Save file
        slug = _url_to_slug(url)
        file_path = save_dir / f"{slug}.md"
        # Handle slug collisions by appending a counter
        if file_path.exists():
            counter = 2
            while file_path.exists():
                file_path = save_dir / f"{slug}_{counter}.md"
                counter += 1

        file_path.write_text(full_content, encoding="utf-8")

        manifest_entries.append({
            "framework": framework,
            "version": version,
            "version_tag": version_tag,
            "source_url": url,
            "file": str(file_path.relative_to(output_dir)),
            "chars": len(full_content),
        })

        page_count += 1
        time.sleep(delay)

    cli.step(2, 3, "Writing manifest")
    scrape_date = datetime.now(tz=timezone.utc).isoformat()
    manifest = {
        "framework": framework,
        "display_name": display_name,
        "version": version,
        "version_tag": version_tag,
        "scrape_date": scrape_date,
        "pages_scraped": len(manifest_entries),
        "source_urls": [e["source_url"] for e in manifest_entries],
        "files": manifest_entries,
    }
    manifest_path = save_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cli.info("Manifest", str(manifest_path))

    cli.step(3, 3, "Done")
    cli.success(
        f"{display_name} {version}: {len(manifest_entries)} pages → {save_dir}"
    )

    return manifest_entries


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape framework documentation for Cola-Coder training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/scrape_docs.py --framework react --version 18
  python scripts/scrape_docs.py --framework nextjs --version 14
  python scripts/scrape_docs.py --all
  python scripts/scrape_docs.py --list
  python scripts/scrape_docs.py --framework zod --version 3 --output-dir /tmp/docs
""",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--framework",
        metavar="NAME",
        help="Framework to scrape (e.g. react, nextjs, zod, typeorm).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Scrape all supported frameworks at their default versions.",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List supported frameworks and exit.",
    )

    parser.add_argument(
        "--version",
        metavar="VER",
        default=None,
        help=(
            "Version string to use (e.g. 18, 14, 3). "
            "Defaults to each framework's default_version."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default="data/docs",
        help="Root output directory (default: data/docs/).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Seconds to sleep between requests (default: 1.0).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=200,
        metavar="N",
        help="Maximum pages to scrape per framework (default: 200).",
    )

    return parser


def main() -> None:
    """Entry point for the documentation scraper."""
    parser = _build_parser()
    args = parser.parse_args()

    cli.header("Cola-Coder", "Docs Scraper")

    # ── --list ───────────────────────────────────────────────────────────
    if args.list:
        cli.info("Supported frameworks", "")
        for key, cfg in FRAMEWORK_CONFIG.items():
            cli.info(
                f"  {key}",
                f"{cfg['display_name']}  (default version: {cfg['default_version']})",
            )
        return

    output_dir = Path(args.output_dir)

    # ── --all ────────────────────────────────────────────────────────────
    if args.all:
        total = len(FRAMEWORK_CONFIG)
        for idx, (fw_key, fw_cfg) in enumerate(FRAMEWORK_CONFIG.items(), start=1):
            cli.info(f"\n[{idx}/{total}] Scraping", fw_cfg["display_name"])
            ver = fw_cfg["default_version"]
            try:
                scrape_framework(
                    fw_key,
                    ver,
                    output_dir,
                    delay=args.delay,
                    max_pages=args.max_pages,
                )
            except KeyboardInterrupt:
                cli.warn("Interrupted — partial results saved.")
                sys.exit(1)
            except Exception as exc:  # noqa: BLE001
                cli.warn(f"Failed to scrape {fw_cfg['display_name']}: {exc}")
                continue

        cli.done("All frameworks scraped.", {"Output": str(output_dir.resolve())})
        return

    # ── --framework ──────────────────────────────────────────────────────
    framework = args.framework.lower()
    if framework not in FRAMEWORK_CONFIG:
        cli.fatal(
            f"Unknown framework: {framework!r}",
            hint="Run --list to see supported frameworks.",
        )

    version = args.version or FRAMEWORK_CONFIG[framework]["default_version"]

    try:
        entries = scrape_framework(
            framework,
            version,
            output_dir,
            delay=args.delay,
            max_pages=args.max_pages,
        )
    except KeyboardInterrupt:
        cli.warn("Interrupted — partial results saved.")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        cli.fatal(f"Scrape failed: {exc}")

    total_chars = sum(e["chars"] for e in entries)
    cli.done(
        f"Scraped {len(entries)} pages ({total_chars / 1_000:.0f} KB)",
        extras={
            "Framework": framework,
            "Version": version,
            "Output": str((output_dir / framework / version).resolve()),
        },
    )


if __name__ == "__main__":
    main()
