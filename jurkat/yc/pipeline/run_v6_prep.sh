#!/bin/bash
# v6 tile measures (new volume region: speckle), counts (inventory v6, Golgi shell area factor), pre-pass index (chromatin
# dropped inside speckles). Geometry: jspeckles.py already applied to jgeom_cell1.npz (v5 copy: jgeom_cell1_v5.npz).
# Membranes are unchanged from v5, so the lipid step (jlipids.py) is not rerun. Log: logs/run_v6_prep.log
set -e
V=${VCELL_ROOT:-.}
PY=$V/conda_envs/vcell/bin/python; P=$V/jurkat/yc/pipeline
cd $V
export NUMBA_THREADING_LAYER=workqueue
echo "== tiles $(date)"; NUMBA_NUM_THREADS=1 JWC_NPROC=24 $PY $P/jwc.py tiles
echo "== counts $(date)"; NUMBA_NUM_THREADS=1 $PY $P/jwc.py counts
echo "== index $(date)"; NUMBA_NUM_THREADS=8 $PY $P/jwc.py index
echo "== done $(date)"
