#!/bin/bash
# v6 full pack (after run_v6_prep.sh and the test tiles were moved away by v6_qc.py clean): 24 workers, then summary.
# Log: logs/jwc_full_v6.log
set -e
V=${VCELL_ROOT:-.}
PY=$V/conda_envs/vcell/bin/python; P=$V/jurkat/yc/pipeline
cd $V
export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue JWC_NPROC=24
echo "== pack $(date)"; $PY $P/jwc.py pack
echo "== summary $(date)"; $PY $P/jwc.py summary
cp -p data/derived/wc/summary.json data/derived/wc/summary_v6.json; cp -p data/derived/wc/libs.json data/derived/wc/libs_v6.json
echo "== done $(date)"
