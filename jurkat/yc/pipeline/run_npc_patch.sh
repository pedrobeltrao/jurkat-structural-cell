#!/bin/bash
# After pack v3: NPC pre-pass at the NE pores (jnpc.py, already written to prepass/npc_cell1.npz) patched into the
# finished pack (jwc_patch_npc.py), then summary, viewer encode + check, density validation.
set -e
cd "${VCELL_ROOT:-.}"
W=data/derived/wc
# (v3 finished 05:35; summary_v3_prenpc.json, counts_v3_prenpc.npz, libs_v3.json saved.) Also repacks the tiles holding the
# 50 IM monomers re-oriented after v3 (matrix side rule fix, data/derived/reorient_matrix_ids.txt).
cd jurkat/yc/pipeline; P=../../../conda_envs/vcell/bin/python
JWC_LIBS_FROM=../../../$W/libs_v3.json $P -u jwc.py counts
cmp <($P -c "import json;print(json.load(open('../../../$W/libs.json'))['libs'])") <($P -c "import json;print(json.load(open('../../../$W/libs_v3.json'))['libs'])")
$P -u jwc.py index
PATCH_LIBS=../../../data/derived/reorient_matrix_ids.txt PATCH_OLD_COUNTS=../../../$W/counts_v3_prenpc.npz JWC_NPROC=26 $P -u jwc_patch_npc.py
$P -u jwc.py summary
cd ../../..; P=conda_envs/vcell/bin/python
$P -u jurkat/viewer/jenc.py --cube 1600 --out jurkat/viewer/web/data
$P jurkat/viewer/check_data.py --ref
$P jurkat/validate/densities.py
echo ALL_DONE
