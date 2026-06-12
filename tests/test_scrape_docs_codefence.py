"""DATA-039: docs HTML→Markdown must tag code fences with the right language.

The highlight-language CSS class lives on either <pre> or <code> depending on
the highlighter. The converter only read <code>'s class, so the common case
where it's on <pre> (highlight.js / MDX / Docusaurus) produced an UNTAGGED
fence — the model couldn't tell the example's language. Now both are checked,
plus the `lang-x` shorthand.
"""

import importlib.util
from pathlib import Path

import pytest

bs4 = pytest.importorskip("bs4")
_SCRIPT = Path(__file__).parent.parent / "scripts" / "scrape_docs.py"


def _md(html: str) -> str:
    spec = importlib.util.spec_from_file_location("scrape_docs_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    soup = bs4.BeautifulSoup(html, "html.parser")
    return module._element_to_markdown(soup)


class TestCodeFenceLanguage:
    def test_language_on_code_element(self):
        md = _md('<pre><code class="language-ts">const x = 1;</code></pre>')
        assert "```ts" in md and "const x = 1;" in md

    def test_language_on_pre_element(self):
        # highlight.js / MDX style: class on <pre>, not <code>.
        md = _md('<pre class="language-python"><code>x = 1</code></pre>')
        assert "```python" in md

    def test_lang_shorthand(self):
        md = _md('<pre><code class="lang-tsx">const A = () => null;</code></pre>')
        assert "```tsx" in md

    def test_pre_without_code(self):
        md = _md('<pre class="language-js">let y = 2;</pre>')
        assert "```js" in md and "let y = 2;" in md

    def test_untagged_when_no_language_class(self):
        md = _md("<pre><code>plain code</code></pre>")
        assert "```\nplain code" in md  # fence opens with no language token

    def test_code_text_preserved(self):
        # Multi-line code (indentation/newlines) must survive get_text().
        html = "<pre><code class=\"language-ts\">function f() {\n  return 1;\n}</code></pre>"
        md = _md(html)
        assert "function f() {\n  return 1;\n}" in md

    def test_standalone_inline_code_backticked(self):
        # A <code> walked directly (not buried inside a flattened <p>/<li>) is
        # backticked. (Inline code INSIDE a <p> is currently flattened — DATA-040.)
        md = _md("<code>npm install</code>")
        assert "`npm install`" in md


class TestListCodeBlocks:
    """DATA-040: a <pre> code block inside a list item must render as a proper
    fenced block, not be flattened (get_text) into a single broken bullet line."""

    def test_code_block_in_ordered_item_preserved(self):
        html = (
            "<ol>"
            "<li>Install the package:"
            "<pre><code class=\"language-bash\">npm install zod</code></pre>"
            "</li>"
            "<li>Import it:"
            "<pre><code class=\"language-ts\">import { z } from 'zod';</code></pre>"
            "</li>"
            "</ol>"
        )
        md = _md(html)
        # Descriptions are numbered bullets...
        assert "1. Install the package:" in md
        assert "2. Import it:" in md
        # ...and the code blocks are real fences (not flattened into the bullet).
        assert "```bash\nnpm install zod\n```" in md
        assert "```ts\nimport { z } from 'zod';\n```" in md

    def test_multiline_code_in_list_keeps_formatting(self):
        html = (
            "<ul><li>Example:"
            "<pre><code class=\"language-ts\">function f() {\n  return 1;\n}</code></pre>"
            "</li></ul>"
        )
        md = _md(html)
        assert "- Example:" in md
        assert "function f() {\n  return 1;\n}" in md  # newlines survived

    def test_plain_list_unchanged(self):
        md = _md("<ul><li>first</li><li>second</li></ul>")
        assert "- first" in md and "- second" in md

    def test_ordered_numbering_unchanged_for_plain_items(self):
        md = _md("<ol><li>one</li><li>two</li><li>three</li></ol>")
        assert "1. one" in md and "2. two" in md and "3. three" in md
