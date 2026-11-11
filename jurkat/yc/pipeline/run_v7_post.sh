#!/bin/bash
# v7 after the full pack: standard validation (v6 outputs kept as *_v6), Tier 2, gallery, composite xz, ledger. No encode.
# Log: logs/run_v7_post.log
V=${VCELL_ROOT:-.}
PY=$V/conda_envs/vcell/bin/python; VD=$V/jurkat/validate
cd $V
export NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=8 LOC_VERSION=v7 JWC_REGIONS=v7
export VAL_ING=$V/data/derived/ingredients_v7.tsv VAL_GEOM=$V/data/derived/jgeom_cell1_v7.npz VAL_POLY=$V/data/derived/prepass_v7/polysomes_cell1.npz
for f in summary.json libs.json counts.npz pack_log.tsv; do b=${f%.*}; e=${f##*.}; cp -p data/derived/wc/$f data/derived/wc/${b}_v7.$e; done
echo "== densities $(date)"; $PY $VD/densities.py
echo "== ribosome_nn $(date)"; $PY $VD/ribosome_nn.py
echo "== crowding $(date)"; $PY $VD/crowding.py
echo "== loc_compare model $(date)"; $PY $VD/loc_compare.py model
echo "== loc_compare compare $(date)"; $PY $VD/loc_compare.py compare
echo "== loc_compare holdout $(date)"; $PY $VD/loc_compare.py holdout
echo "== fluor $(date)"; $PY $VD/fluor.py all
echo "== tier2 $(date)"; NUMBA_NUM_THREADS=1 TIER2_NPROC=16 $PY $VD/tier2.py collect --ver v7 && NUMBA_NUM_THREADS=1 TIER2_NPROC=16 $PY $VD/tier2.py model --ver v7 && $PY $VD/tier2.py compare --ver v7
echo "== gallery $(date)"; NUMBA_NUM_THREADS=1 $PY $VD/jurkat_gallery.py collect && $PY $VD/jurkat_gallery.py render && $PY $VD/jurkat_gallery.py figure
echo "== composite xz $(date)"; PLANE=xz $PY $VD/composite_gallery.py && for f in composite_gallery composite_gallery_xz; do [ -e data/derived/validation/$f.png ] && cp -p data/derived/validation/$f.png data/derived/validation/${f}_v7.png; done
echo "== ledger $(date)"; $PY $V/jurkat/provenance/build_ledger.py v7 assemble figures
echo "== done $(date)"
