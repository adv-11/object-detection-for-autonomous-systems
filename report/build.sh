#!/usr/bin/env bash
# Build a report PDF: fill tables from results/ -> pandoc HTML -> headless Chrome PDF.
#   bash report/build.sh              # report.md            -> report.pdf
#   bash report/build.sh extension    # extension.md         -> extension.pdf
set -euo pipefail
cd "$(dirname "$0")"
NAME="${1:-report}"
uv run python make_tables.py
NAME="$NAME" uv run python - <<'PY'
import os, re, pathlib
name = os.environ["NAME"]
src = pathlib.Path(f"{name}.md").read_text(encoding="utf-8")
def fill(m):
    for d in (pathlib.Path("generated"), pathlib.Path("../results/tables")):
        p = d / f"{m.group(1)}.md"
        if p.exists():
            return p.read_text(encoding="utf-8")
    return "*(table not available)*"
pathlib.Path("_built.md").write_text(re.sub(r"<!--TABLE:(\w+)-->", fill, src), encoding="utf-8")
PY
pandoc _built.md -s --from markdown+raw_html+pipe_tables --to html5 --css style.css \
  --embed-resources --resource-path=.:.. --metadata lang=en -o "$NAME.html"
CHROME="${CHROME:-/c/Program Files/Google/Chrome/Application/chrome.exe}"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$(cygpath -w "$PWD/$NAME.pdf")" "file:///$(cygpath -m "$PWD/$NAME.html")" 2>/dev/null
rm -f _built.md
echo "wrote report/$NAME.pdf"
