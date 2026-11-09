# Pipeline scripts

These scripts replace the notebook cells in `notebook/pipeline_driver_notebook.ipynb`. Paths come from `pipeline/config.py`, which you can override with `YC_*` environment variables. All output goes to `work/`.

| Script | Notebook cells | Output | Time (12 cores) |
|---|---|---|---|
| `s1_inventory.py` | 1–4, 22 + `pipeline/assemblies.py` | `work/results/inventory/{ingredients.csv, oligo_map.json, complexes_*.csv, oligomers.csv}`; new library entries in `pipeline/lib/` | ~1 h on the first run (RCSB queries), ~1 min once `work/rcsb_cache.json` exists |
| `s2_geometry.py` (`--sphere`: analytic cell instead of the real shapes) | 24, 25 (+ re-index from 28) | `work/yc3d/{geometry.json, tile_measure.parquet, tile_labvol.parquet, geom/}` | 15 s |
| `s3_pack.py` (`--random`: no organisation layer) | 27–29 | `work/yc3d/{tiles_raw/, tile_counts.parquet, prepass.npz, expected_totals.parquet, pack_summary.parquet}` | 1 min |
| `s4_validate.py` | 31–36 | `work/results/wholecell3d_validation.json`, `work/yc3d/libpack_meta.npz` | 30 s |
| `s5_encode.py` | 38–45 | `work/yc3d/web/data/` | 30 s (needs the AFDB tar for per-residue pLDDT) |
| `s6_motion.py` (optional) | — | `work/conformers/{kept_residues, pool_modes}.npz`; `library/modes.{bin,json}` in the encoded data. Needs `POOL_DB` (pool predictions, foldcomp) and `POOL_PY` (a Python with foldcomp); `MOTION_CAP` sets the amplitude cap in nm (default 0 = none). `--skip-fit` only re-encodes | ~100 min fit (10 cores); 6 s re-encode |
| `cell_meshes.py` (optional) | — | `pipeline/cell_meshes.npz`: the segmented meshes of the model cell (`YC_ORGANELLE_MESHES`); the file is in the repository, so this only re-extracts it | 5 s |
| `organelle_stats.py` (optional) | — | `pipeline/organelle_stats.json` from segmented cryo-FIB/SEM meshes (`YC_ORGANELLE_MESHES`); the file is in the repository, so this only re-derives it | 5 s |
| `deploy_viewer.sh` | — | builds `web/app` (Node in `.node/`) and installs `index.html`, `assets/` and `data/` into the viewer folder; keeps the previous version as `*_prev` | 10 s |
| `check_inventory.py` | — | copy conservation of the inventory against the proteome | 5 s |

To run the whole pipeline: `scripts/run_all.sh` (or `scripts/run_all.sh 3` to restart from packing), then `scripts/deploy_viewer.sh`.

**Reproducing the shipped model.** Run `s2_geometry.py`. Then run `s3_pack.py --from-intermediates ../intermediates`, `s4_validate.py` and `s5_encode.py --tiles-only`. This reproduces all 6,416 tiles of the shipped viewer data byte for byte. The geometry agrees to 1e-13 nm, so for an exact comparison copy `intermediates/geometry.json` into `work/yc3d/` first.

**Inputs outside the bundle** (defaults in `run_all.sh`):
- `YC_AFDB_TAR`: AlphaFold DB yeast proteome v6 tar.
- `YC_DIMERS`: AFDB predicted yeast dimers (`homodimers/`, `heterodimers/` with `candidates.csv`).
