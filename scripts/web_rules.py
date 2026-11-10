#!/usr/bin/env python3
"""Put RULES.md into the challenge web page, between its RULES:START and RULES:END markers.

    python3 scripts/web_rules.py ~/projects/site-sprout-challenge/ig/sprout-challenge/index.html

Needs the markdown package (pip install markdown). Relative links (INSTALL.md, README.md, files in the repository)
become links to the repository on GitHub. Heading ids match GitHub's, so links like #7-scoring work on both.
Each numbered rule is a <details> section; the page's script opens it when a link points into it.
Re-run it whenever RULES.md changes, and copy RULES.pdf to the page's rulebook.pdf.
"""
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/shantanu-builds-robots/sprout-challenge/blob/main/"
START, END = "<!-- RULES:START -->", "<!-- RULES:END -->"


def slug(text: str, sep: str) -> str:
    """GitHub's heading ids: lowercase, punctuation dropped, spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", sep)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    page = Path(sys.argv[1])
    md = (ROOT / "RULES.md").read_text()
    md = re.sub(r"\A# .*\n", "", md)  # the page has its own title
    md = re.sub(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)", lambda m: f"]({REPO_URL}{m.group(1)})", md)
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc"],
                             extension_configs={"toc": {"slugify": slug}})
    body = re.sub(r'<a href="(https?://)', r'<a target="_blank" rel="noopener" href="\1', body)
    # Each numbered rule becomes a section that opens on tap; the intro and contents before rule 1 stay open.
    parts = re.split(r'(?=<h2 id=")', body)
    body = parts[0] + "".join(
        re.sub(r'<h2 id="([^"]+)">(.*?)</h2>', r'<details class="rule" id="\1"><summary><h2>\2</h2></summary>', p, count=1)
        + "</details>" for p in parts[1:])
    html = page.read_text()
    if html.count(START) != 1 or html.count(END) != 1:
        sys.exit(f"{page}: needs exactly one {START} and one {END}")
    head, rest = html.split(START)
    _, tail = rest.split(END)
    page.write_text(f"{head}{START}\n{body}\n{END}{tail}")
    print(f"wrote the rulebook into {page}")


if __name__ == "__main__":
    main()
