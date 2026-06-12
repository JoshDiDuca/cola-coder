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
