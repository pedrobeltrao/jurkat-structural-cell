#!/bin/bash
# v6 after the full pack: validation reruns (v5 outputs kept as *_v5), Tier 2, gallery, then the whole-cell viewer encode.
# Log: logs/run_v6_post.log
V=${VCELL_ROOT:-.}
PY=$V/conda_envs/vcell/bin/python; VD=$V/jurkat/validate
cd $V
export NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=8 LOC_VERSION=v6
echo "== densities $(date)"; $PY $VD/densities.py
echo "== ribosome_nn $(date)"; $PY $VD/ribosome_nn.py
echo "== crowding $(date)"; $PY $VD/crowding.py
echo "== loc_compare model $(date)"; $PY $VD/loc_compare.py model
echo "== loc_compare compare $(date)"; $PY $VD/loc_compare.py compare
echo "== loc_compare holdout $(date)"; $PY $VD/loc_compare.py holdout
echo "== fluor $(date)"; $PY $VD/fluor.py all
echo "== tier2 $(date)"; NUMBA_NUM_THREADS=1 TIER2_NPROC=16 $PY $VD/tier2.py collect --ver v6 && NUMBA_NUM_THREADS=1 TIER2_NPROC=16 $PY $VD/tier2.py model --ver v6 && $PY $VD/tier2.py compare --ver v6
echo "== gallery $(date)"; NUMBA_NUM_THREADS=1 $PY $VD/jurkat_gallery.py collect && $PY $VD/jurkat_gallery.py render && $PY $VD/jurkat_gallery.py figure
echo "== encode $(date)"; NUMBA_NUM_THREADS=1 $PY $V/jurkat/viewer/jenc_full.py --tiles data/derived/wc/tiles --libs data/derived/wc/libs.json \
  --meta data/derived/wc/tiles_meta.json --parquet data/derived/wc/tiles.parquet --geom data/derived/jgeom_cell1.npz \
  --out data/derived/web_full_v6 --workers 6
echo "== check_data $(date)"; $PY $V/jurkat/viewer/check_data.py data/derived/web_full_v6 --ref --sample 300
echo "== done $(date)"
