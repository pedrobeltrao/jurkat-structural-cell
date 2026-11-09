#!/bin/bash
# Whole-cell pack v3 (2026-11-09): v2 + polysomes v3 (tighter chains) + TM monomers re-oriented (jlib_reorient_tm.py, run
# before this) + adaptive sphere trees for membrane libraries. v2 results archived to tiles_v2 / pack_log_v2 / summary_v2.
set -e
cd "${VCELL_ROOT:-.}"
P=conda_envs/vcell/bin/python; W=data/derived/wc
INCLUDE_MEMBRANE=1 REFINE_LOG=data/derived/lib_refine_st_log_membrane.tsv $P jurkat/yc/pipeline/lib_refine_st.py $(cat data/derived/membrane_lib_ids.txt)
[ -d $W/tiles_v2 ] || mv $W/tiles $W/tiles_v2
[ -f $W/pack_log_v2.tsv ] || mv $W/pack_log.tsv $W/pack_log_v2.tsv
[ -f $W/summary_v2.json ] || cp $W/summary.json $W/summary_v2.json
cp $W/counts.npz $W/counts_v2.npz
cd jurkat/yc/pipeline; P=../../../conda_envs/vcell/bin/python
$P -u jwc.py counts
$P -u jwc.py index
JWC_NPROC=26 $P -u jwc.py pack
$P -u jwc.py summary
