#!/bin/bash
# v5 geometry post-processing and tile/count/index steps (after jgeom_build.py). Log: logs/run_v5_prep.log
set -e
V=${VCELL_ROOT:-.}
PY=$V/conda_envs/vcell/bin/python; P=$V/jurkat/yc/pipeline
cd $V
export NUMBA_THREADING_LAYER=workqueue
echo "== cristae $(date)"; NUMBA_NUM_THREADS=8 $PY $P/jcristae.py
echo "== face cap $(date)"; NUMBA_NUM_THREADS=8 $PY $P/jface_cap.py
echo "== tiles $(date)"; NUMBA_NUM_THREADS=1 JWC_NPROC=8 $PY $P/jwc.py tiles
echo "== counts $(date)"; NUMBA_NUM_THREADS=1 $PY $P/jwc.py counts
echo "== index $(date)"; NUMBA_NUM_THREADS=8 $PY $P/jwc.py index
echo "== lipids $(date)"; $PY $P/jlipids.py
echo "== done $(date)"
