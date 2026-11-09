# Jurkat T cell model

Jurkat T cell (OpenOrganelle jrc_jurkat-1, cell 1): a molecular model of the whole cell with ~960 million placed
macromolecules (887 M packed plus 76 M pre-placed chromatin, polysome and cytoskeleton units). The viewer streams
either the whole cell (public site) or one 1.6 µm sub-volume (detail build) at residue resolution, plus whole-cell
organelle outlines.

## What the viewer shows

- **Whole-cell build** (`jenc_full.py`): every instance of the cell in 200 nm tiles, loaded around the view centre
  within a GPU memory budget (about 8 million molecules at once) and evicted when out of view. Until a tile has
  loaded, the section shows overview stand-ins (about 100 molecules per tile, sampled by mass, enlarged and coloured
  by compartment). Copy numbers in the inspect panel are counts in the whole cell. The compartment fields for cut
  faces are sampled at 48 nm; membrane meshes per tile are 8 nm. The current data are pack v3 (no cristae).
- **Sub-volume build** (`jenc.py`). A cube of 1.6 µm (8 x 8 x 8 tiles of 200 nm) chosen to contain plasma membrane, cytoplasm, nuclear
  envelope and nucleus (and mitochondria / ER where possible), re-centred on the origin (`jenc.py`).
- **Molecules.** Every instance in the cube, as placed by the Jurkat packing pipeline, with its PDB or AlphaFold DB
  structure at one bead per residue (nucleotide). Copy numbers in the inspect panel are counts in the sub-volume.
- **Compartments.** Six signed-distance fields cropped to the cube (cell, perinuclear midsurface, mitochondria, ER,
  Golgi, endo-lysosomal). Labels follow `cell3d.label_from` of the pipeline: perinuclear space within 17 nm of the
  nuclear-envelope midsurface (except in pores), mitochondrial intermembrane space 8 nm wide inside the outer membrane.
- **Whole-cell context.** Organelle surfaces of the entire cell at 48 nm resolution (Organelles and Membranes views).
- **Lipids.** Generated in the browser near the camera, generic mammalian composition (PC 45%, PE 20%, SM 10%,
  PS 8%, PI 5%, other 12%); cholesterol is not drawn.

## Dynamics example

When the build has `data/dynamics.json` (from `jurkat/validate/dynamics.py`), the "Diffusion" guided views and the
Dynamics button (key G) show Brownian-dynamics trajectories of inert tracers in one 150 nm box of cytosol and one of
nucleoplasm. The tracers are GFP-sized (r 2.3 nm, green) and IgG-sized (r 4.5 nm, magenta). They move among the
frozen crowd of placed macromolecules. A step is rejected if it would overlap a crowder or leave the box (reflecting
walls), and the step size uses D0 from Stokes-Einstein with a hydrodynamic correction for the box's volume fraction.
Frames are 20 ns apart. The panel gives the predicted long-time D/D0 per tracer. The crowders do not move, there is no
binding, and tracers do not interact, so D/D0 for large probes is a lower bound. The tracers are drawn at their real
radius and cut by the slab like molecules, with optional trails. The box outline is orange.

## Not in this build

mRNA polylines, linker DNA, AlphaFold pLDDT per residue, conformational motion modes and STRING interaction modules.
