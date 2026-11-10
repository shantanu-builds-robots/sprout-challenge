#!/usr/bin/env python3
"""Build RULES.pdf from RULES.md: a cover page, then the rulebook with page numbers.

    python3 scripts/pdf/make_rulebook.py            # writes RULES.pdf at the repository root

Needs pandoc and WeasyPrint (pip install weasyprint), and the network once for the web fonts. Relative links
(INSTALL.md, README.md, files in the repository) become links to the repository on GitHub, so they work in the PDF.
Re-run it whenever RULES.md changes, and commit both.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_URL = "https://github.com/shantanu-builds-robots/sprout-challenge/blob/main/"
HERE = Path(__file__).resolve().parent

COVER = """<section class="cover">
<img class="hero" src="{hero}"><div class="fade"></div>
<div class="inner">
<div class="kicker">1K FOLLOWER COMMUNITY CHALLENGE</div>
<h1>Sprout Challenge</h1>
<div class="sub">Rulebook. Code a robot dog through an obstacle course.<br>Fastest robot wins a $20 gift card.</div>
<table>
<tr><td>Task</td><td>Write <code>act(observation) -&gt; 12 joint angles</code>, called 50 times a second</td></tr>
<tr><td>Deadline</td><td>30 November 2026, 23:59 IST (18:29 UTC)</td></tr>
<tr><td>Results</td><td>by 3 December 2026, on Instagram</td></tr>
<tr><td>Enter</td><td>DM a public GitHub repo link to @shantanu.builds.robots</td></tr>
<tr><td>Code</td><td><a href="https://github.com/shantanu-builds-robots/sprout-challenge">github.com/shantanu-builds-robots/sprout-challenge</a></td></tr>
</table>
</div>
<div class="foot">Version of {date}. The latest rules are always RULES.md in the repository; if this PDF and RULES.md
differ, RULES.md applies.</div>
</section>
"""


def main():
    md = (ROOT / "RULES.md").read_text()
    # Repository-relative links -> GitHub links (keep in-page anchors and absolute URLs).
    md = re.sub(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)", lambda m: f"]({REPO_URL}{m.group(1)})", md)
    md = md.replace("(README.md#", f"({REPO_URL}README.md#")
    body = subprocess.run(["pandoc", "-f", "gfm", "-t", "html5"], input=md, capture_output=True, text=True,
                          check=True).stdout
    # The inline "Contents" line becomes its own page as a list.
    m = re.search(r"<p><strong>Contents:</strong>(.*?)</p>", body, re.S)
    if m:
        items = re.findall(r'<a\s+href="(#[^"]+)">([^<]+)</a>', m.group(1))
        names = [re.sub(r"^[0-9]+[.]\s*", "", " ".join(t.split())) for _, t in items]
        toc = '<section class="toc"><h2>Contents</h2><ol>' + "".join(
            f'<li><a href="{h}">{n}</a></li>' for (h, _), n in zip(items, names)) + "</ol></section>"
        body = body.replace(m.group(0), "")
    else:
        toc = ""
    import datetime
    date = datetime.date.today().strftime("%-d %B %Y")
    html = (f'<!doctype html><html><head><meta charset="utf-8"><title>Sprout Challenge rulebook</title>'
            f'<link rel="stylesheet" href="{HERE / "rulebook.css"}"></head><body>'
            f'{COVER.replace("{date}", date).replace("{hero}", (HERE / "cover.jpg").as_uri())}{toc}{body}</body></html>')
    out = ROOT / "RULES.pdf"
    tmp = HERE / "_rulebook.html"
    tmp.write_text(html)
    try:
        subprocess.run(["weasyprint", str(tmp), str(out)], check=True)
    finally:
        tmp.unlink(missing_ok=True)
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
