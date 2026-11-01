#!/usr/bin/env bash
# Build report/report.pdf from report/report.md (pandoc -> HTML -> headless Chrome PDF).
set -euo pipefail
cd "$(dirname "$0")"
pandoc report.md -s --from markdown+raw_html+pipe_tables --to html5 --css style.css \
  --embed-resources --resource-path=.:.. --metadata lang=en -o report.html
CHROME="${CHROME:-/c/Program Files/Google/Chrome/Application/chrome.exe}"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf-no-header \
  --print-to-pdf="$(cygpath -w "$PWD/report.pdf")" "file:///$(cygpath -m "$PWD/report.html")" 2>/dev/null
echo "wrote report/report.pdf"
