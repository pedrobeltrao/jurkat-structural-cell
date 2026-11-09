#!/bin/sh
# Build the Jurkat viewer (app/) and assemble the deployable site in web/ (index.html, assets/, molstar/, MODEL.md).
# web/data/ is written separately by jenc.py and is never touched here.
set -e
cd "$(dirname "$0")"
VIEW=$(pwd)
cd app
[ -d node_modules ] || npm ci
mkdir -p public
[ -d public/molstar ] || cp -r node_modules/molstar/build/viewer public/molstar
npx vite build
mkdir -p "$VIEW/web"
cd "$VIEW/web"
rm -rf index.html assets molstar
cp "$VIEW/app/dist/index.html" index.html
cp -r "$VIEW/app/dist/assets" assets
cp -r "$VIEW/app/dist/molstar" molstar
cp "$VIEW/MODEL.md" MODEL.md
[ -d data ] || echo "note: web/data/ does not exist yet (run jenc.py)"
echo "site assembled in $VIEW/web: $(du -sh --exclude=data . | cut -f1) without data"
