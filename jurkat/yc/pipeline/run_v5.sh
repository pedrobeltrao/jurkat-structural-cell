#!/bin/bash
# v5 full pack (after run_v5_prep.sh and the test tiles were removed): 24 workers, then summary. Log: logs/jwc_full_v5.log
set -e
V=${VCELL_ROOT:-.}
PY=$V/conda_envs/vcell/bin/python; P=$V/jurkat/yc/pipeline
cd $V
export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue JWC_NPROC=24
echo "== pack $(date)"; $PY $P/jwc.py pack
echo "== summary $(date)"; $PY $P/jwc.py summary
echo "== done $(date)"
