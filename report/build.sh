#!/usr/bin/env bash
# Build report/report.pdf: fill tables from results/ -> pandoc HTML -> headless Chrome PDF.
set -euo pipefail
cd "$(dirname "$0")"
uv run python make_tables.py
uv run python - <<'PY'
import re, pathlib
src = pathlib.Path("report.md").read_text(encoding="utf-8")
def fill(m):
    p = pathlib.Path("generated") / f"{m.group(1)}.md"
    return p.read_text(encoding="utf-8") if p.exists() else "*(table not available)*"
pathlib.Path("_built.md").write_text(re.sub(r"<!--TABLE:(\w+)-->", fill, src), encoding="utf-8")
PY
pandoc _built.md -s --from markdown+raw_html+pipe_tables --to html5 --css style.css \
  --embed-resources --resource-path=.:.. --metadata lang=en -o report.html
CHROME="${CHROME:-/c/Program Files/Google/Chrome/Application/chrome.exe}"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$(cygpath -w "$PWD/report.pdf")" "file:///$(cygpath -m "$PWD/report.html")" 2>/dev/null
rm -f _built.md
echo "wrote report/report.pdf"
