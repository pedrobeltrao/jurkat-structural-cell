#!/bin/sh
# Build the viewer (web/app) and install it plus the encoded data into the served viewer folder.
# Usage: scripts/deploy_viewer.sh [viewer_dir]   (default: the folder that contains this bundle)
# The previous data/ and assets/ are kept as *_prev (one level of backup).
set -e
cd "$(dirname "$0")/.."
BUNDLE=$(pwd)
VIEWER=${1:-$(dirname "$BUNDLE")}
export PATH="$BUNDLE/.node/bin:$PATH"
DATA=$(.venv/bin/python -c "import sys; sys.path.insert(0, 'pipeline'); import config; print(config.WEB_OUT)")
[ -f "$DATA/library/types.json" ] || { echo "no encoded data at $DATA (run s5_encode.py)"; exit 1; }

cd web/app
[ -d node_modules ] || npm ci
# data is copied separately below; a public/data link would be duplicated into dist/ (~0.5 GB)
mkdir -p public && rm -f public/data
[ -d public/molstar ] || cp -r node_modules/molstar/build/viewer public/molstar
npx vite build
cd "$VIEWER"
for d in data assets; do [ -e "$d" ] && { rm -rf "${d}_prev"; mv "$d" "${d}_prev"; }; done
cp -r "$BUNDLE/web/app/dist/assets" assets
cp "$BUNDLE/web/app/dist/index.html" index.html
cp -r "$DATA" data
cp "$BUNDLE/MODEL.md" MODEL.md
# build stamp: the viewer appends it to every data URL, so browsers never mix cached files from two builds
printf '{"v": "%s"}\n' "$(date +%Y%m%d%H%M%S)" > data/version.json
[ -d molstar ] || cp -r "$BUNDLE/web/app/dist/molstar" molstar
echo "deployed to $VIEWER"
