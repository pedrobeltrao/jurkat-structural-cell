# Jurkat structural cell

A molecule-by-molecule structural model of a whole human Jurkat T cell (cell 1 of the OpenOrganelle FIB-SEM volume
`jrc_jurkat-1`). It holds about 960 million macromolecules (887 M packed plus 76 M pre-placed chromatin, polysome
and cytoskeleton units), each drawn from an experimental (PDB) or predicted (AlphaFold DB) structure. A WebGPU viewer streams the model into the browser at residue resolution.

- **Viewer:** https://pedrobeltrao.github.io/jurkat-structural-cell/ (GitHub Pages, built from `jurkat/viewer/app`).
- **Data:** https://huggingface.co/datasets/pedrobeltrao/jurkat-structural-cell-data. The site reads `web_full_v4/` (16.7 GB in
  2,365 files, none larger than 34 MB); the earlier v3 pack stays in `web_full_v3/` (open it with `?data=`).
- **Status:** work in progress. The current data are **pack v4**: procedural cristae, inventory v4, no interaction-driven
  attraction. See "Known limitations".

## What the model is

- **Geometry.** Cell 1 of `jrc_jurkat-1` (Heinrich et al. 2021; Xu et al. 2021), refined at 16 nm. It includes the plasma
  membrane, the nuclear envelope (fitted with HMFF) and its 1,000 nuclear pores, and the mitochondria, ER and
  endo-lysosomal compartments from the segmentations. The Golgi is procedural, because cell 1 has no segmented Golgi.
  The cell is about 712 µm³.
- **Inventory.** Protein copy numbers come from Jurkat E6-1 RNA consensus and Jurkat proteomics (Geiger et al. 2012),
  scaled to 100 pg of protein. Localisation comes from HPA, UniProt and GO, with nucleo-cytoplasmic partition from
  Itzhak et al. 2016. Complexes come from the Complex Portal, PDB homo-oligomers and AlphaFold DB homo- and
  heterodimers, calibrated against SEC data (Heusel et al. 2019, 2020).
- **Pre-placed assemblies.** These are built before packing:
  - chromatin (31 M nucleosomes plus linker histones, in territories);
  - polysomes (2.1 M ribosomes, EM-seeded) with mRNA threads (250k mRNAs);
  - microtubules from a centrosome, and cortical F-actin;
  - one NPC (PDB 7R5K) in every pore.
- **Packing.** Every other macromolecule is packed into 7,485 tiles of 500 nm. A sphere-tree collision packer
  (forked from the yeast whole-cell pipeline) works in checkerboard phases. Membrane proteins are packed in their
  bilayer with UniProt topology. There is no interaction-driven attraction: placement is random within
  compartment, membrane and orientation rules, so the model can be tested against proximity data.
- **Lipids.** These are not stored. The viewer generates bilayers near the camera with a generic mammalian
  composition.

Full details (assumptions, decisions, validation) are in `jurkat/viewer/MODEL.md` (shown in the viewer under "Model
details"). The changes to the viewer are in `jurkat/viewer/CHANGES.md`.

## Repository layout

| Path | Contents |
|---|---|
| `jurkat/em/` | EM label handling, cell segmentation and refinement |
| `jurkat/inventory/` | RNA consensus, inventory v0, complexes, homo-oligomers, AFDB dimers, SEC calibration |
| `jurkat/structures/` | AlphaFold DB helpers |
| `jurkat/yc/pipeline/` | Packing pipeline: geometry (`cell3d.py`, `jgeom_build.py`, `jcristae.py`), libraries (`jlib_*.py`), pre-pass assemblies (`jchromatin.py`, `jpolysome.py`, `jmrna.py`, `jcytoskeleton.py`, `jnpc.py`), packer (`pack3d.py`, `jpack_test.py`) and the whole-cell driver (`jwc.py`, `run_v3.sh`) |
| `jurkat/yc/scripts/` | Stage scripts inherited from the yeast pipeline |
| `jurkat/validate/` | Densities, crowding, ribosome neighbours, tracer Brownian dynamics |
| `jurkat/viewer/` | Encoders (`jenc.py` for a 1.6 µm cube, `jenc_full.py` for the whole cell), data checks (`check_data.py`), local server (`serve.py`), build script and the WebGPU app (`app/`) |

The data are not in this repository. This includes the EM volumes, reference tables, structure libraries, packing
output and viewer files. Scripts read them from `$VCELL_ROOT/data/...`, where `VCELL_ROOT` defaults to the current
directory, so run them from the repository root or set `VCELL_ROOT`. The AlphaFold DB human proteome tar is located
with `AFDB_HUMAN_TAR`.

## Running the viewer locally

You need Node 20+ and a WebGPU browser (current Chrome or Edge on desktop; Safari 26+; Firefox 141+ on Windows).

```sh
cd jurkat/viewer
./build_web.sh                   # npm ci, copy Mol*, vite build -> web/ (web/data is never touched)
```

Then point the viewer at a data folder in one of these ways:

1. **Local data.** Put or symlink the viewer data at `web/data/` (for example a download of the Hugging Face dataset
   folder `web_full_v4/`; the older v3 pack stays in `web_full_v3/`), then run:
   ```sh
   python serve.py web --port 8000           # static server with HTTP Range + CORS
   ```
   Open http://localhost:8000/. `python -m http.server` also works, but it has no Range support, so the viewer
   downloads whole block shards instead of single tiles.
2. **Hosted data, no rebuild.** Add a `config.json` next to `index.html`:
   ```json
   {"data_base": "https://huggingface.co/datasets/pedrobeltrao/jurkat-structural-cell-data/resolve/main/web_full_v4/"}
   ```
3. **Per visit.** Use the URL parameter `?data=<url>`, which overrides `config.json`. For example:
   `http://localhost:8000/?data=https://huggingface.co/datasets/pedrobeltrao/jurkat-structural-cell-data/resolve/main/web_full_v4/`

The order is `?data=`, then `config.json`, then the default `data/`. Relative URLs resolve against the page. The data
host must allow CORS and should support HTTP Range requests. Hugging Face `resolve/main/` URLs support both.

`?view=<n|name>` opens a bookmark, for example `?view=Nuclear%20pore`.

### Two data modes

- **Whole cell** (`jenc_full.py`, about 16 GB). This is the default for the public site.
  - Viewer tiles of 200 nm are stored as gzip members of one shard per 1 µm block (`tiles/s_BX_BY_BZ.bin`). The
    viewer reads them with Range requests, clipped to the cut slab, within a budget of 4, 8 or 16 M molecules
    (Performance panel "Detail", or `?detail=low|medium|high`), centre of the view first, evicting the least
    recently used tiles. A 60 nm section through the whole cell fits the 8 M budget.
  - Until a tile's molecules arrive, the cut face shows per-tile *overview stand-ins* (`overview/*.bin.gz`) and
    compartment colours.
- **1.6 µm cube** (`jenc.py`, about 180 MB). One sub-volume with every compartment. It is used by the validation pages
  and by the single-file artifact (`artifact_pack.py`).

## Rebuilding the data

The pipeline runs on a cluster: about 2.5 h on 26 cores for the packing, with up to 128 GB per job. Inputs are listed
in `MODEL.md`. The order is:

1. Geometry: `jurkat/em/*`, then `jgeom_build.py` (and `jcristae.py` from v4 on).
2. Inventory and complexes: `jurkat/inventory/*`, `jingredients.py` (and `jinventory_v4.py`).
3. Libraries: `jlib_build.py`, `lib_refine_st.py`, `jlib_reorient_tm.py`.
4. Pre-pass assemblies: `jchromatin.py`, `jpolysome.py`, `jmrna.py`, `jcytoskeleton.py`, `jnpc.py`.
5. Packing: `jwc.py` steps tiles, counts, index, pack and summary (`run_v3.sh`, `run_npc_patch.sh`).
6. Viewer data:
   ```sh
   python jurkat/viewer/jenc_full.py --tiles data/derived/wc/tiles_v3 --libs data/derived/wc/libs_v3npc.json \
       --meta data/derived/wc/tiles_meta_v3.json --parquet data/derived/wc/tiles_v3.parquet \
       --geom data/derived/jgeom_cell1_pre_cristae.npz --out data/derived/web_full_v3 --workers 6
   python jurkat/viewer/check_data.py data/derived/web_full_v3
   ```
   The encoder streams one 1 µm slab per worker (bounded memory) and resumes after an interruption (state in
   `<out>_work/`).

## Data sources and licences

| Source | Used for | Licence / terms |
|---|---|---|
| OpenOrganelle `jrc_jurkat-1` (Janelia; Heinrich et al. 2021, *Nature*; Xu et al. 2021, *Nature*) | Cell and organelle geometry, ribosome and NPC positions | CC BY 4.0 |
| Protein Data Bank (wwPDB) | Experimental structures (e.g. 7R5K NPC, ribosome, nucleosome) | CC0 1.0 |
| AlphaFold DB (EMBL-EBI / Google DeepMind) | Predicted monomers and dimers | CC BY 4.0 |
| UniProt | Sequences, topology, localisation | CC BY 4.0 |
| Complex Portal (EMBL-EBI) | Curated complexes | *to confirm* (EMBL-EBI terms of use) |
| STRING | Interaction evidence for dimer selection | CC BY 4.0 |
| Human Protein Atlas | Subcellular localisation, cell-line RNA | CC BY-SA 4.0 (*to confirm version*) |
| DepMap / CCLE | Jurkat RNA expression | CC BY 4.0 (*to confirm*) |
| Geiger et al. 2012 (PRIDE PXD002395, Expression Atlas E-PROT-20) | Jurkat protein abundances | Publication / EMBL-EBI terms (*to confirm*) |
| Itzhak et al. 2016; Heusel et al. 2019, 2020; PaxDb | Partition, SEC assembly state, abundance cross-checks | Publications / database terms (*to confirm*) |
| Mol* (molstar) | Atomic model viewer embedded in the app | MIT |
| Yeast whole-cell viewer and pipeline (G. Magnusson, github.com/zoidpuff/yeast_cell_viewer) | Code the Jurkat pipeline and viewer were forked from | *no licence file; permission / licence to confirm* |

Please cite the original data sources when you use the model. The derived viewer data on Hugging Face inherit the
attribution requirements of the CC BY sources above.

## Known limitations (pack v4)

- **Placement shortfalls.** 95.9% of the target mass was placed. Thin compartments jam with the current random
  sequential packer: mitochondrial matrix 69.5% (between cristae), ER lumen 65.4%, NE 88%, endo-lysosomal membranes
  87%. Cytosol, nucleoplasm, PM, ER membrane and inner mitochondrial membrane are at 98–99.7%.
- **Truncated cell.** The EM volume cuts cell 1 at one face (about 3–4% of the cell missing). The model currently
  puts a plasma membrane on that face; the next geometry revision makes it an open boundary ("edge of EM data").
- **Cristae** are procedural (lamellae perpendicular to each mitochondrion's long axis, 70–100 nm apart), not
  segmented. The viewer's cut-face colouring does not yet know them (crista lumens are coloured as matrix).
- **No interaction-driven organisation.** Placement is random within compartments and membranes, so the model can be
  compared with proximity data without circularity. Pre-built heterodimers (AFDB models selected by SEC co-elution
  or STRING) are placed as complexes.
- **Viewer.** The whole-cell distance fields for the cut-face colours are resampled at 48 nm (32 nm for ER, Golgi and
  endo-lysosomes), which agrees with the 16 nm geometry at 98.4% of points inside the cell. The per-tile membrane
  meshes are 8 nm.
- **GPU memory.** The whole-cell build needs a WebGPU storage-buffer binding of about 175 MB (library 158 MB, fields
  173 MB). Desktop GPUs expose ≥ 1 GB; devices at the WebGPU minimum (128 MiB) get an error message.
- **Browser support.** Needs WebGPU (recent Chrome or Edge on desktop). Tested on an Intel integrated GPU (Iris Xe,
  "Detail: medium"); other GPUs and Safari/Firefox are untested.

## Licence

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) for the code, documentation and model data (see `LICENSE`). Non-commercial reuse with attribution; please cite this repository. Third-party sources keep their own terms (table above).
