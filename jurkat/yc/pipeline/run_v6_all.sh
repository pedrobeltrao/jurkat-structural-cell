#!/bin/bash
# v6: reapply inventory (R6a shares 0.3/0.15), recount, move test tiles away, full pack, then the post-pack chain.
V=${VCELL_ROOT:-.}; PY=$V/conda_envs/vcell/bin/python; P=$V/jurkat/yc/pipeline
cd $V; export NUMBA_THREADING_LAYER=workqueue
set -e
echo "== inventory $(date)"; $PY $P/jinventory_v6.py --apply > logs/jinventory_v6_apply.log 2>&1
echo "== counts $(date)"; NUMBA_NUM_THREADS=1 $PY $P/jwc.py counts
echo "== clean test tiles $(date)"; mv data/derived/wc/tiles/*.npz data/derived/v6/test_tiles_round2/ ; mv data/derived/wc/pack_log.tsv data/derived/v6/test_tiles_round2/pack_log_test.tsv
df -h data/derived | tail -1
$P/run_v6.sh > logs/jwc_full_v6.log 2>&1
set +e
$P/run_v6_post.sh > logs/run_v6_post.log 2>&1
echo "== all done $(date)"
