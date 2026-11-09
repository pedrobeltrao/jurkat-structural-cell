#!/bin/sh
# Rebuild the whole model: inventory -> geometry -> packing -> validation -> web encoding.
# Usage: scripts/run_all.sh [first_stage]   e.g. "3" to restart from packing. Logs go to $YC_WORK/sN.log.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
: "${YC_DIMERS:=$HOME/Projects/bioemu_docker/afdb_yeast_dimers}"
: "${YC_AFDB_TAR:=$HOME/Downloads/UP000002311_559292_YEAST_v6.tar}"
export YC_DIMERS YC_AFDB_TAR
WORK=$($PY -c "import sys; sys.path.insert(0, '../pipeline'); import config; print(config.WORK)")
first=${1:-1}
for s in s1_inventory s2_geometry s3_pack s4_validate s5_encode; do
  n=${s#s}; n=${n%%_*}
  [ "$n" -lt "$first" ] && continue
  echo "== $s"
  $PY "$s.py" 2>&1 | tee "$WORK/$s.log"
done
