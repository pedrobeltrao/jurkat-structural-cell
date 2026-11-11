# Jurkat structural cell

A molecule-by-molecule structural model of a whole human Jurkat T cell (cell 1 of the OpenOrganelle FIB-SEM volume
`jrc_jurkat-1`). It holds about 950 million macromolecules (880 M packed plus 70 M pre-placed chromatin, polysome,
cytoskeleton, lamin and nuclear-pore units), each drawn from an experimental (PDB) or predicted (AlphaFold DB) structure. A WebGPU viewer streams the model into the browser at residue resolution.

- **Viewer:** https://pedrobeltrao.github.io/jurkat-structural-cell/ (GitHub Pages, built from `jurkat/viewer/app`).
- **Data:** https://huggingface.co/datasets/pedrobeltrao/jurkat-structural-cell-data. The site reads `web_full_v7/` (16.7 GB in
  2,367 files, none larger than 37 MB). The earlier packs stay in `web_full_v3/` and `web_full_v4/`; open them with `?data=`.
- **Status:** work in progress. The site shows pack **v7**. Compared with v4, the model now has:
  - an open boundary where the EM volume truncates the cell, a nucleolus, nuclear speckles, and membrane-adjacent
    shells for anchored proteins, the actin cortex, the lamina and the nuclear pores (v5-v6);
  - genome-mapped chromatin: 46 chromosome chains with genome coordinates, with RNA polymerase II, CTCF and cohesin
    at their mapped sites, and nascent RNA (v7);
  - a stack of Golgi cisternae next to the centrosome, and microtubules traced in the EM volume (v7);
  - a lamin rod lattice, a pericentriolar matrix, a vimentin cage and microtubule-associated protein bands (v7).

  The app also reads the v4-v6 data. See "Known limitations".

## What the model is

- **Geometry.** Cell 1 of `jrc_jurkat-1` (Heinrich et al. 2021; Xu et al. 2021), refined at 16 nm. It includes the plasma
  membrane, the nuclear envelope (fitted with HMFF) and its 1,000 nuclear pores, and the mitochondria, ER and
  endo-lysosomal compartments from the segmentations. Cell 1 has no segmented Golgi. In v7 the Golgi is a modelled stack
  of cisternae (20 nm lumens every 40 nm, 4.2 µm³ in total) placed around the centrioles found in the EM.
  The cell is about 712 µm³.
- **Inventory.** Protein copy numbers come from Jurkat E6-1 RNA consensus and Jurkat proteomics (Geiger et al. 2012),
  scaled to 100 pg of protein. Localisation comes from HPA, UniProt and GO, with nucleo-cytoplasmic partition from
  Itzhak et al. 2016. Complexes come from the Complex Portal, PDB homo-oligomers and AlphaFold DB homo- and
  heterodimers, calibrated against SEC data (Heusel et al. 2019, 2020).
- **Pre-placed assemblies.** These are built before packing:
  - chromatin: 28.2 M nucleosomes plus linker histones on 46 genome-mapped chains, with Pol II, CTCF and cohesin;
  - polysomes (2.1 M ribosomes, EM-seeded) with mRNA threads (250k mRNAs plus 13k nascent transcripts);
  - microtubules traced in the EM volume (1.46 mm), centrioles, and cortical F-actin;
  - a lamin rod lattice under the nuclear envelope (about 1 M lamin copies);
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
   folder `web_full_v7/`; older packs stay in `web_full_v3/` and `web_full_v4/`), then run:
   ```sh
   python serve.py web --port 8000           # static server with HTTP Range + CORS
   ```
   Open http://localhost:8000/. `python -m http.server` also works, but it has no Range support, so the viewer
   downloads whole block shards instead of single tiles.
2. **Hosted data, no rebuild.** Add a `config.json` next to `index.html`:
   ```json
   {"data_base": "https://huggingface.co/datasets/pedrobeltrao/jurkat-structural-cell-data/resolve/main/web_full_v7/"}
   ```
3. **Per visit.** Use the URL parameter `?data=<url>`, which overrides `config.json`. For example:
   `http://localhost:8000/?data=https://huggingface.co/datasets/pedrobeltrao/jurkat-structural-cell-data/resolve/main/web_full_v7/`

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
   python jurkat/viewer/jenc_full.py --tiles data/derived/wc/tiles --libs data/derived/wc/libs_v7.json \
       --meta data/derived/wc/tiles_meta.json --parquet data/derived/wc/tiles.parquet \
       --geom data/derived/jgeom_cell1_v7.npz --prepass data/derived/wc --regions v7 \
       --mrna data/derived/prepass/mrna_cell1.npz --mrna-drop data/derived/prepass_v7/mrna_drop.npy \
       --mrna-extra data/derived/chromatin/v8/prepass/mrna_nascent_cell1.npz \
       --ingredients data/derived/ingredients_v7.tsv --out data/derived/web_full_v7 --workers 6
   python jurkat/viewer/check_data.py data/derived/web_full_v7 --ref --geom data/derived/jgeom_cell1_v7.npz
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

## Known limitations (pack v7)

- **Placement shortfalls.** 96.4% of the target mass was placed. Crowded and thin compartments jam with the current
  random sequential packer:
  - mitochondrial matrix 86% and ER lumen 82%;
  - nucleolus 87% (rRNA is not modelled);
  - NE membrane 80% (crowded by the lamin lattice and the chromatin);
  - pericentriolar matrix 84%, Golgi peripheral shell 86% and NPC channel shell 77%.

  Cytosol, nucleoplasm, PM, cortex, ER membrane and inner mitochondrial membrane are at 97–99.6%.
- **Chromatin.** The genome-mapped chromatin matches the EM peripheral density, territory order, compartments and
  LAD/NAD positions. It misses TAD insulation and the fine texture of DAPI images.
- **Compartment codes.** The viewer has 32 region codes. Proteins in the new v7 packing regions (pericentriolar
  matrix, vimentin cage, microtubule-associated bands) are coloured as cytosol, and the NPC channel shell as nuclear
  pore.
- **Truncated cell.** The EM volume cuts cell 1 at its bottom face (41 µm² footprint). That face is an open boundary:
  no membrane, nothing placed beyond it. A side-wall extrapolation puts the missing cap at about 7 µm³ (1% of the
  cell; a sphere-shaped cap would be 29 µm³, but the measured walls are much steeper), and copy numbers are scaled
  to the modelled part (×0.990 for volumes, ×0.943 for plasma-membrane proteins).
- **Cristae** are procedural (lamellae perpendicular to each mitochondrion's long axis, 70–100 nm apart), not
  segmented.
- **No interaction-driven organisation.** Placement is random within compartments and membranes, so the model can be
  compared with proximity data without circularity. Pre-built heterodimers (AFDB models selected by SEC co-elution
  or STRING) are placed as complexes.
- **Viewer.** The whole-cell distance fields for the cut-face colours are resampled at 48 nm (cell, NE) and 32 nm
  (mitochondria, ER, Golgi, endo-lysosomes, nucleolus). The Golgi cisternae use a 16 nm field. Labels agree with
  the 16 nm geometry at 98.8% of points inside the cell (cristae, nucleolus, speckles and Golgi cisternae included).
  The per-tile membrane meshes are 8 nm. The whole-cell overview mesh shows only the outer surface of the Golgi stack.
- **GPU memory.** The whole-cell build needs a WebGPU storage-buffer binding of about 230 MB (library about 160 MB, fields
  about 230 MB). Desktop GPUs expose ≥ 1 GB; devices at the WebGPU minimum (128 MiB) get an error message.
- **Browser support.** Needs WebGPU (recent Chrome or Edge on desktop). Tested on an Intel integrated GPU (Iris Xe,
  "Detail: medium"); other GPUs and Safari/Firefox are untested.

## Licence

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) for the code, documentation and model data (see `LICENSE`). Non-commercial reuse with attribution; please cite this repository. Third-party sources keep their own terms (table above).
