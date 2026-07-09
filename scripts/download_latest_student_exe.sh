#!/usr/bin/env bash
# Скачать последний IUP Student.exe с GitHub Releases
set -euo pipefail

REPO="${IUP_GITHUB_REPO:-Impulse3e/IUP}"
OUT="${1:-IUP-Student.exe}"

url=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for asset in data.get('assets', []):
    name = asset.get('name', '')
    if name.endswith('.exe'):
        print(asset['browser_download_url'])
        break
else:
    sys.exit(1)
")

echo "Скачивание: $url"
curl -fL "$url" -o "$OUT"
chmod +x "$OUT" 2>/dev/null || true
echo "Готово: $(pwd)/$OUT"
