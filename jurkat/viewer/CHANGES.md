# Jurkat viewer: changes from the yeast viewer

The source is `external/yeast_cell_viewer/source/web/app`, copied to `app/` (nothing under `external/` was modified).
Build: `./build_web.sh` runs `npm ci` if needed, copies Mol* into `app/public/molstar`, runs `vite build` and assembles
`web/` (index.html, assets/, molstar/, MODEL.md). `web/data/` comes from `jenc.py` and the script never touches it.
Check data: `conda_envs/vcell/bin/python check_data.py [web/data] [--ref]`.

## Data / constants (`src/data.ts`)
- `REGION_NAMES` (18 codes, jenc order), `MEM_NAMES` (PM, NE, ER, MITO_OM, MITO_IM, GOLGI, ENDO) and `LABEL_NAMES` (11 labels, cell3d order) are now readable versions of the Jurkat lists.
  `LIPID_CLASSES` = PI, PE, PC, SM, PS, (unused sterol slot), other. New `MEM_REGIONS` (7..13) and `SOLUBLE_REGIONS` (0..6).
- New palette `PAL`: regions 0..17 (pale-blue cytosol, lavender nucleoplasm, orange matrix, green ER, yellow Golgi, brown endo, one colour per membrane,
  violet chromatin, pink polysome ribosomes, green cytoskeleton, grey-blue NPC); lipids 32..42; organelle-mode membranes 56..62;
  membranes-only view 66..72; compartment fill 80..90; traced cell surface 91.
- `SPECIAL` colours are keyed by the type id string with prefix patterns (`RIBO*`, `NUCLEOSOME`, `H1-*`, `NPC`, `MT_*`, `F_*`), resolved per type at load (`specialFor`, `idMatches`).
- `loadTileIndex`: missing columns (`vol_glucan_nm3`, `vol_mannan_nm3`) give zeros, and `wall` is null when absent. Loading messages no longer quote yeast bead counts.
- `TGRID` kept at 26 (the tiles -4..3 fit in it).

## Geometry (`src/fields.ts`, `src/geometry.ts`)
- fields.ts: the reader is unchanged (same layout as yeast `web_fields`). The field constants are now `F_CELL, F_NE, F_MITO, F_ER, F_GOLGI, F_ENDO`.
- geometry.ts has been rewritten:
  - `prepareGeometry(geo)` completes a `jurkat_fields` geometry.json for the renderer. It sets `G[0..2]` to the cube half edge, sets empty primitive lists, keeps only the pores whose influence reaches the cube, builds an 8^3 pore lookup grid (`grid.pore_off/pore_idx`), and records `half` and `cellR`.
  - `Geometry.memSdfs` / `labelFrom` port `cell3d.mem_sdfs` / `label_from` exactly, including the pore torus. This is used by the minimap.

## Shaders (`src/shaders.ts`)
- FILL has been rewritten for Jurkat:
  - The 6 fields come from `fieldSdf` (it returns 1e9 for a missing field), and `memSdfs` / `labelOf` follow cell3d exactly. The NE uses `|ne_mid| - PERI_W/2` with the pore-torus correction from the pore grid. The IM is `-(om + IMS_W)`.
  - Membrane bands are drawn for the 7 membranes. The lipid-droplet monolayer case has been removed.
  - Rays are clipped to the sub-volume cube (`u.cell.x` = half edge) instead of the enclosing sphere. A cube face inside the cell is shaded like a cut face. Where molecules are loaded, the backdrop is pushed 40 nm behind the face.
  - The exterior trace finds the plasma membrane inside the cube. It must converge, otherwise the pixel shows background. The yeast cell-wall backdrop pass has been removed, the surface colour is palette 91 and its id is 0x200 (PM).
  - Organelle mode without a cut shows background (meshes only).
- `animInst`: the membrane-protein region range is now 7..13 (it was 14..21).
- DRAW_MESH: the LD monolayer offset (mem 8) and the cell-surface discard (mem 9) have been removed.
- `freeRadius`: the glucan case has been removed.
- **Draw-list offsets widened from 22 to 26 bits** (`OFF_BITS`/`OFF_MASK`, up to 67 M beads or spheres; the count of up to 64 sits in the top 6 bits). Type ids stay at 14 bits.
- All shaders validate with naga (naga-wasm 30.2, a WGSL to WGSL translate with validation), as do the yeast originals.

## Renderer (`src/renderer.ts`)
- Gives a clear error if the bead or sphere-tree buffer exceeds the device's storage-binding limit, or 2^26 entries. Everything else is unchanged.

## Streaming (`src/tiles.ts`)
- Cell-wall detail is requested only when index.json has `wall` (never for Jurkat). The worker no longer receives wall parameters.
- Sub-volume box taken from the tile index. Without a cut, the loaded tiles are those on cube faces that face the camera, or every tile in view when the camera is inside the cube, plus plasma-membrane shell tiles. Before, only the yeast exterior shell was loaded.

## Worker (`src/worker.ts`)
- The procedural lipid composition is a mammalian default: PI 5, PE 20, PC 45, SM 10, PS 8, other 12 mol%. No sterol headgroup is drawn.
- The LD monolayer logic (mem 8) has been removed; lipids always go on two leaflets.
- β-glucan/mannan generation has been removed.

## UI (`src/main.ts`, `index.html`, `src/about.ts`, `src/camera.ts`)
- Class switches resolve id patterns: actin `F_*`, microtubules `MT_*`, ribosomes `RIBO*`, chromatin `NUCLEOSOME` + `H1-*`, pores `NPC`. The no-motion set is the same.
- Show list: tRNA, FG repeats, β-glucan and Mannan have been removed. mRNA and Linker DNA appear only when the index has those beads. Classes with no matching types are hidden. Soluble and membrane protein switches use the Jurkat region sets.
- Legend: the Goodsell legend lists Jurkat classes. The compartment legend lists all 18 regions.
- The pLDDT colour option is hidden when every `plddt_mean` is null. The STRING-module option is hidden without organisation.json.
- Inspect panel:
  - The SGD link has been removed. The UniProt link stays (`https://www.uniprot.org/uniprotkb/<acc>`), and a gene-name UniProt search is shown when there is no accession.
  - "Copies here … in the sub-volume".
  - The source label shows 'assembly unit' for pre-pass building blocks. "Load atomic model" is disabled when there is no single PDB or AlphaFold entry.
  - The lipid text gives the mammalian composition.
- Titles and text now describe the Jurkat T cell sub-volume. This covers the title, h1, subtitle, loading text, search placeholder, help (zoom range), "Sweep through cube", the procedural-detail label, the toast for the whole-cell surfaces and the screenshot file name. The preview text no longer says "6,416 tiles". The About text uses the given wording, and `MODEL.md` (now `jurkat/viewer/MODEL.md`) is a short Jurkat description.
- Camera: the start distance is 3000 nm and `maxDist` is 25000 nm. At load it is raised to `3 * (R + |centre|)` so the whole-cell organelle meshes fit (about 44 µm for the test data). The projection is infinite reversed-Z, so there is no far plane.
- Minimap: the extent is 1.25 × the cube half edge (fields exist only there). It uses Jurkat label colours, and the area outside the cube is drawn semi-transparent.
- The sweep range is ± the cube half edge (via `G[0]`).
- `prepareGeometry` is called after `loadFields`.

## Tracer dynamics overlay (`src/dynamics.ts`, `src/renderer.ts`, `src/shaders.ts`, `src/main.ts`, `index.html`)
- `data/dynamics.json` and the `.bin` trajectories (uint16 (frames, n, 3), `box_lo + q / quant * box_nm`, written by `jurkat/validate/dynamics.py`) are fetched only on first use. That happens when a bookmark with `"dynamics": "<region>"` is applied, on the Dynamics button (shown only when such a bookmark exists), or on key G. When the data is missing, the viewer is unchanged, and G gives a toast.
- Rendering:
  - Tracers are sphere impostors (`DRAW_TRACERS`, using the shared `emitQuad`/`shadeFrag`). They are depth-tested against the molecules and clipped by the cut plane and slab like every sphere.
  - Their picking ids are free-sphere category, kind 6, so hover and click give the tracer name, radius and D/D0.
  - Trails and the 12 box edges use a new screen-space line pipeline (`DRAW_LINES`). It is depth-tested with no depth write and writes colour only, so it adds no outlines. Trails are clipped to the slab. The box is not.
  - Both are drawn in the molecule view only, after the free spheres. `Renderer.setTracers` uploads them (the buffers grow as needed).
- Panel (top centre, while active):
  - Play/pause, a frame slider and a frame/time readout (ns/µs).
  - Speed from 5 to 150 frames/s.
  - Tracer selector (each tracer / Both), a region selector (applies that region's bookmark), and trails (off / 10 / 50 / 200 frames).
  - The legend shows radius, n and D/D0 from the json.
  - Any other bookmark or the × button hides the overlay.
- Keys: G show/hide, K play/pause. While playing, the loop sets `dirty` each frame (`dyn.tick`). Positions are linearly interpolated between frames, and the run loops back to the start at the end. When paused, it stops requesting redraws.

## Unchanged
Rendering, LOD, culling, tile format and decoding, picking, outlines, the inspect panel layout, Mol* embedding, bookmarks and tour, and the single-file config.
Remaining `tsc --noEmit` errors: 9 `Uint32Array<ArrayBufferLike>` vs `GPUAllowSharedBufferSource` typing errors in main.ts (5) and renderer.ts (4). The untouched yeast source has the same errors (TS 5.9 lib typing), and vite builds regardless.

## 2026-11-09: artifact bundle
- `data.ts` `fetchBin` and `worker.ts` `getBin` read an optional `data/split.json`:
  - `{path: n}` means the file is served as `path.000` … `path.NNN`, which are concatenated;
  - `"_b64": 1` means every non-JSON file is served as base64 text at `path.txt`.

  Without `split.json` nothing changes.
- `artifact_pack.py DATA_DIR OUT_DIR` builds a claude.ai Artifact bundle:
  - strips the document wrapper from the page;
  - splits and base64-encodes the data;
  - copies the dynamics sets and the Diffusion bookmarks whose boxes lie inside the cube;
  - drops bookmarks with NaN (regions absent from the cube).

## 2026-11-09: whole-cell mode, configurable data location, public repository skeleton
- **Data location** (`data.ts`).
  - `DATA_BASE` is resolved once at start, with top-level await: `?data=<url>`, then `config.json` next to
    `index.html` (`{"data_base": "<url>"}`), then `data/`. Relative URLs resolve against the page.
  - The worker gets the same base from `tiles.ts`. `split.json` / `_b64` work as before; the artifact bundle has no
    `config.json` and keeps `data/`.
- **Whole-cell builds** (`jenc_full.py`; `index.json` `whole_cell`, `shard`, `overview`, column `off`):
  - **Worker** (`getTileBytes`): reads a tile as the byte range `[off, off + gz_bytes)` of its block shard
    `tiles/s_BX_BY_BZ.bin` with an HTTP Range request. On status 200 (Range ignored) or a failed ranged fetch, it uses
    whole shards (cache of 3).
  - **Tile requests**: `TileManager.tileMsg` adds the shard fields. Cube builds still use `tiles/ix_iy_iz.bin.gz`.
  - **Streaming**: `TileManager.whole` means no sub-volume box. With a cut, candidates are ranked by distance to the
    view centre (plus 0.1 × eye distance), within the existing budget (94% of the instance pool, about 8 M; 4,088
    slots) and with LRU eviction. Without a cut, the yeast PM-shell rule applies. The HUD shows when the budget is
    reached.
  - **Overview stand-ins** (`tiles.ts` `Overview`):
    - One `overview/BX_BY_BZ.bin.gz` per 1 µm block, holding up to 128 spheres per viewer tile (instances sampled by
      mass; radius scaled so the stand-ins fill half the molecular envelope volume, capped at 6×). They are loaded
      for the blocks that meet the slab, nearest to the view centre first, within 40% of the free-sphere pool.
    - They live in the free-sphere pool as kind 7 (radius and region in the record). When a tile's molecules load,
      its stand-ins switch to kind 15 and are culled; they switch back when the tile is evicted. Stand-ins are
      dropped when the cut is off and in the organelle and membrane views.
    - Picking a stand-in gives "Overview stand-in" with its compartment.
  - **Shaders**: `freeRadius(kind, w)` and `vsFree` handle kind 7 (palette colour of the region, no motion), and
    `cullSpheres` skips kind 15.
  - **Tile-state grid**: now 128³ (tiles −64..63, ±12.8 µm; `data.TGRID`, `TOFF`; the shader's `tileLoaded` reads
    the same layout). Bit 0 means molecules loaded, bit 1 means stand-ins shown; either pushes the cut-face backdrop
    back. `setTileBits` uploads only the changed word (before, the whole grid was uploaded per tile).
  - **Pores**: `prepareGeometry` uses a 48³ pore lookup grid for whole-cell geometry (8³ for the cube).
  - **`types_tiles.bin`**: with `index.json` `types_tiles_u32`, tile indices are uint32 (more than 65,535 tiles).
    The encoder keeps up to 256 tiles per type, one per block, so "Fly to nearest" goes to a nearby copy.
  - **Text**: in the subtitle, help, inspect panel ("in the cell") and sweep button, and a whole-cell About text
    (`ABOUT_HTML_WHOLE`). `index.html` has spans `.subvolWord` and `#zoomHelp`. The unbuilt cubeSize / title fix is
    kept.
- **Typecheck**: `tsc --noEmit` is clean. The 9 inherited `Uint32Array<ArrayBufferLike>` errors were fixed with
  `<ArrayBuffer>` typed arrays (`typeFlags`, `Library.beads/st/modes`, `paletteArray`). All shaders validate with
  naga-wasm 30.2.
- **New files**:
  - `jenc_full.py`: streaming whole-cell encoder (see its docstring and PLANNING_NOTES §7g).
  - `serve.py`: local static server with HTTP Range and CORS.
  - `make_site_repo.sh`: refreshes the code in `jurkat/site_repo`.
- **`check_data.py`**:
  - reads `split.json` parts;
  - accepts an http(s) URL, reading tiles with Range requests (must answer 206);
  - decodes whole-cell shards, uint32 `types_tiles` and one overview block;
  - checks the index `n_inst` against library copies;
  - samples 300 tiles for the `label_mask` check on large indices.
- **Dynamics gated on the data** (Pedro: the diffusion example is an offline validation, not part of the public
  viewer).
  - `dynamics.json` is probed at start. Without it, these are all off:
    - the Dynamics button;
    - the G/K keys and their help row (`.dynHelp`);
    - bookmarks with a `dynamics` field;
    - the "Dynamics example" section of Model details.
  - The code is unchanged, and the 1.6 µm cube build with `dynamics.json` behaves as before. The whole-cell encoder
    writes no dynamics data.
- **`MODEL.md`** now describes both builds (whole cell and 1.6 µm sub-volume).
- **Whole-cell fields and buffer check.**
  - The whole-cell fields are 48 nm for cell, NE and mito, and 32 nm for ER, Golgi and endo (`--field-sp`,
    `--field-sp-thin`; `--fields-only` rewrites just the fields). That is 173 MB float32.
  - `Renderer.init` now gives a clear error when the fields exceed the storage-binding limit, as it already did for
    the library.
- **Tests run** (no browser: headless Chromium crashes on this node):
  - `tsc --noEmit` is clean, naga validates every shader, and `build_web.sh` succeeds; `web/data` is untouched.
  - A Node harness (esbuild bundles of `data.ts`, `worker.ts` and `tiles.ts`) ran against `serve.py` (Range) and
    `python -m http.server` (no Range). It covered:
    - `config.json` and `?data=` resolution;
    - `split.json` parts;
    - worker init and tile loads (instance, vertex and polymer counts equal the index);
    - lipid detail generation;
    - the Overview block parser, hide/show and eviction, with a stubbed GPU queue.
  - `check_data.py` passes on the folder and over HTTP Range.

## 2026-11-09 (2): whole-cell section fits the budget; plane slider; Hugging Face rate limit
Pedro's test of the live site: only a central patch had molecules ("7.96 M molecules in 900 tiles, memory budget
reached"). Each tile uploaded all its instances although a 40–60 nm slab shows 20–30% of them.
- **Slab-clipped uploads** (`worker.ts` `clipTile`, `tiles.ts`). With a cut in whole-cell builds, the worker keeps
  only what reaches the slab ± a margin:
  - instances whose bounding sphere reaches it;
  - polymer beads inside it;
  - mesh triangles that reach it.

  Ghosts and lipids stay worker-side: lipid detail is generated for the slab range as before.
- **Margin**: the widest of 100, 60, 30, 15 or 6 nm whose estimated section total fits the budget (the estimate
  includes half a page of waste per tile). For the whole cell at 60 nm with 8 M, that is 6 nm: a 200 nm margin would
  triple the section.
- **Caches** (worker):
  - compressed tile bytes in an LRU of 200–500 MB in total (navigator.deviceMemory × 64 MB), split over the
    workers;
  - decoded tiles in an LRU of 48.
- **Re-clipping**: when the slab leaves a tile's clipped range, the tile is re-clipped from these caches with no
  refetch. This happens once the plane has rested 250 ms, or at once for uploads that no longer reach the slab
  (sweeps, drags). The old upload stays until its replacement arrives, so the section has no holes while it reloads.
- **Desired set**: in cut mode, every tile meeting the slab in view, nearest to the view centre first. The budget
  counts clipped instances: the loaded count, or the tile's estimated slab share.
- **Budget scaled to the device** (`Renderer.detail`; Performance panel "Detail" selector, saved in localStorage;
  `?detail=`): low 4 M, medium 8 M, high 16 M molecules, capped by the storage-binding limit. Auto is medium, or low
  on fallback / software adapters (SwiftShader).
- **Stand-ins only where needed**: they are loaded only for blocks with section tiles whose molecules are missing,
  and evicted otherwise.
- **HUD**: "section N% of M tiles", the number of stand-ins actually shown, and the budget value when it is reached.
  "MB streamed" counts network bytes only.
- **Hugging Face HTTP 429**: per-tile Range requests to `resolve/main` hit the rate limit after about 2,800
  requests.
  - The worker now remembers each shard's redirect target (the signed CDN URL; 20 min, re-resolved on refusal), so
    later ranges go to the CDN.
  - Each shard is resolved once per worker, because tiles go to workers by block.
  - 429/503 are retried with Retry-After or back-off (worker and `data.ts`).
  - Result: one section is 365 hub requests and 3,075 CDN requests, with no 429.
- **404s**: `library/modes.json`, `library/organisation.json` and `dynamics.json` were requested although the build
  has none. `version.json` now lists `"absent"` files (`jenc_full.py write_version`), and `data.ts isAbsent` skips
  them. `data/derived/web_full_v3/version.json` was patched with the same `v`; it must be re-uploaded to Hugging
  Face.
- **Cut-plane Position slider**: it was hard-coded at ±2,200 nm (a yeast value). It now spans the model box along
  the current plane normal ± 100 nm, updated on every plane change (axis buttons, Align to view).
  - Whole cell: Z −7,372..7,996, X −8,116..8,212, Y −5,164..5,212 nm. Cube: ±900 nm.
  - Sweep runs over the same range.
  - Keyboard and wheel steps, bookmarks, pan and fly-to were already unclamped (checked).
  - The section map fits the whole cell, and the help text gives the cell size ("about 16 µm across").
- **Before / after** (Z plane 0, 60 nm slab, 8 M budget; headless SwiftShader):

  | | Before (live site) | After (local server) | After (Hugging Face data) |
  |---|---|---|---|
  | Tiles loaded | 827 | 2,902 | 2,902 |
  | Molecules | 7.49 M | 6.15 M | 6.2 M |
  | Section with detail | 28.5% | 100% | 100% |
  | Settled after | 38 s | 27 s | 68 s |
  | Downloaded | 117 MB | 470 MB | 470 MB |
  | GPU pools / JS heap | 758 / 561 MB | 758 / 527 MB | 758 / 497 MB |

  Notes:
  - After a 40 nm plane move, re-clipping all 2,902 tiles from the caches took 22 s under SwiftShader, with no
    network.
  - Auto on SwiftShader (low, 4 M) covers 53% of the section, with stand-ins for the rest.
  - Without a cut (camera at 40 µm), the plasma-membrane shell loads to the budget (1,257 tiles, 7.5 M) and the HUD
    reports the budget.
  - Cube mode loads all 128 tiles in view as before.

## Model v5 (2026-11-09): nucleolus, open face, cristae, v5 regions

Backups of every changed file: `*.bak_v4`.

- **Region ids** (`jenc.REGION_NAMES`, `data.ts REGION_NAMES`, `check_data.py`). Pack v5 numbers its regions as
  `jpack_test.RNAMES` (8 volumes incl. nucleolus, 7 membranes, 12 shells). The viewer keeps its v4 ids 0-17 unchanged
  (volumes 0-6, membranes 7-13, pre-pass 14-17) and appends 18 nucleolus and 19-30 the shells (31 ids, fit the 5-bit
  region field of instances and overview stand-ins). `jenc.region_map` maps the tile `job_region` to viewer ids
  (`--regions v5`, the default; `--regions v4` for older packs); index.json records it as `pack_regions`. v4 data stay
  decodable (they never use ids >= 18), and `animInst`'s membrane range 7..13 is unchanged; shell proteins are not in
  the bilayer and move like soluble ones. The UI lists only the regions present in the library; a new class switch
  "Anchored, cortex, lamina" toggles the shells. `regionMask` widened to 32 bits.
- **Fields** (`jenc.write_fields_bin`, `jenc_full.web_fields_whole`, `fields.ts loadFields`). The 7th field
  (nucleolus) is exported with band (40, 40). Whole cell: cell and ne_mid at 48 nm; mito, ER, Golgi, endo and nucleolus
  at 32 nm (`--field-sp-thin`; `--field-sp-of name=nm` overrides; mito went from 48 to 32 nm for the cristae: crista
  lumens 97.8 -> 99.2% labelled IMS, matrix 98.0 -> 99.3%, for +13 MB on the GPU).
  - **Bug fix: offsets into F are now multiples of 16.** The app holds the fields in a Float32Array and reads each
    field's index/data offsets from it; odd offsets above 2^25 were rounded (v4: Golgi index 36,021,519 and data
    37,196,091), shifting the Golgi field by one word, so only 67% of points inside the Golgi read as Golgi lumen in
    the v4 app (now 98.9%). The file is still the sections back to back; geometry.json `fields.total` gives the length
    of F, and the loaders place each section at its offset.
  - **Cristae section** (`jenc.cristae_section`): cell3d's per-mitochondrion lamella parameters (86 x 12 float32) and
    the 48 nm id grid in bricks of 4^3 cells (16.7 k of 216 k bricks stored, int16; 2.1 MB file, 4.4 MB GPU), after
    the fields; `geometry.json fields.cristae` gives its layout. `loadFields` appends a 32-word header to F (offset
    `F.crOff`, geo.G[5] for the shader).
  - **Open face**: geometry.json `face` = {on, z}; geo.G[3] = on, geo.G[4] = z (uniform `Geo.G0.w`, `G1.x`).
- **Label rule** (`geometry.ts`, `shaders.ts FILL`, `check_data.py Geo`): exact ports of v5 `cell3d.label_from`:
  `outside_EM_volume` (label 11) below the face, nucleolus (label 4) inside the nucleoplasm branch when field 6 < 0,
  and `crista_lumen` (matrix = max(om + IMS_W, -lumen) where om + IMS_W < CR_W).
- **Rendering the face**: label 11 is not filled (exterior colour, so nothing beyond the data); on cut faces a grey
  line (palette 92, at least 4 nm and about 2 px wide at any zoom) marks where the section meets the face; the exterior sphere trace runs on the domain depth
  min(PM depth, z - z_face), so the flat face appears as a flat grey surface (id 0x10B, picked as "Edge of the EM
  data"); the legend and About text say "flat face: edge of the EM data". Section map: label 11 transparent.
- **Meshes**: per-tile membrane meshes and the whole-cell organelle meshes drop triangles whose centroid is below the
  face (`jenc.clip_mesh`; the v5 mask fields were extruded below the face, so the PM and organelle surfaces continue
  there). Procedural lipids are generated on the tile mesh triangles (`worker.ts genDetail`), so they follow.
- **Colours**: nucleolus region 18 `#8A4F9E` (deep violet), nucleolus fill 84 `#D6B9DE` (darker than nucleoplasm),
  shells 19-30 shades of their membrane's colour (cortex tan `#D9A26B`, lamina `#6A4C9C`), face 92 `#8D939C`.
- **Bookmarks** (`jenc_full.extra_bookmarks`): "Nucleolus" (section through the deepest nucleolar point) and "Edge of
  the EM data" (vertical section showing the flat bottom).
- **Tests**: WGSL validated by naga (wgpu-native) for every shader; the FILL `labelOf` compiled with a compute entry
  point and run on the GPU agrees 100% with `Geometry.label` run in Node on the same data (fetched over HTTP through
  `data.fetchBin`, split files, `loadFields`, `prepareGeometry`); `fsFill` rendered offscreen with wgpu (edge-of-data
  section, the flat face seen from below, cristae, nucleolus) looks as intended. Agreement with cell3d on the v5
  geometry (`check_data.py DATA --fields-only --ref`; 20,000 random points plus 2,000 per rare label): 99.62% of random
  points, 98.81% of those inside the cell; per label: cytosol 99.5%, nucleoplasm 99.4%, nucleolus 99.4%, matrix 99.4%,
  IMS 98.3% (crista lumens 99.4% IMS), Golgi 98.9%, endo 94.0%, ER lumen 84.8%, pore channel 71%, perinuclear 56%
  (34 nm wide at a 48 nm field; `--field-sp-of ne_mid=32` gives about 70% for +13 MB), outside_EM_volume 100%.
  Membrane triangles of tiles at the face: none below it. GPU size of the v5 fields: 51.2 M words = 205 MB (v4 173 MB).
