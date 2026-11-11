# Jurkat T cell model

Jurkat T cell (OpenOrganelle jrc_jurkat-1, cell 1): a molecular model of the whole cell with ~950 million placed
macromolecules (model v7: 880 M packed plus 70 M pre-placed chromatin, polysome, cytoskeleton, lamin and NPC units). The viewer streams
either the whole cell (public site) or one 1.6 µm sub-volume (detail build) at residue resolution, plus whole-cell
organelle outlines.

## What the viewer shows

- **Whole-cell build** (`jenc_full.py`): every instance of the cell in 200 nm tiles, loaded around the view centre
  within a GPU memory budget (about 8 million molecules at once) and evicted when out of view. Until a tile has
  loaded, the section shows overview stand-ins (about 100 molecules per tile, sampled by mass, enlarged and coloured
  by compartment). Copy numbers in the inspect panel are counts in the whole cell. The compartment fields for cut
  faces are sampled at 48 nm (cell, nuclear envelope) and 32 nm (mitochondria, ER, Golgi, endo-lysosomal,
  nucleolus); membrane meshes per tile are 8 nm. The current data are model v7 (the viewer also reads v4-v6 data).
- **Sub-volume build** (`jenc.py`). A cube of 1.6 µm (8 x 8 x 8 tiles of 200 nm) chosen to contain plasma membrane, cytoplasm, nuclear
  envelope and nucleus (and mitochondria / ER where possible), re-centred on the origin (`jenc.py`).
- **Molecules.** Every instance in the cube, as placed by the Jurkat packing pipeline, with its PDB or AlphaFold DB
  structure at one bead per residue (nucleotide). Copy numbers in the inspect panel are counts in the sub-volume.
- **Compartments.** Seven signed-distance fields (cell, perinuclear midsurface, mitochondria, ER, Golgi,
  endo-lysosomal, nucleolus) plus the procedural cristae. Labels follow `cell3d.label_from` of the pipeline:
  perinuclear space within 17 nm of the nuclear-envelope midsurface (except in pores), the nucleolus inside the
  nucleoplasm (no membrane; from the EM nucleolus segmentation), mitochondrial intermembrane space 8 nm wide inside the
  outer membrane, and lamellar cristae (intracristal lumen 20 nm wide, lamellae 70-100 nm apart, joined to the
  intermembrane space by junction ribbons; the crista lumen counts as intermembrane space). Model v6 adds nuclear
  speckles as spheres (radius 250 nm) inside the nucleoplasm (not in the nucleolus or the pore channels).
- **Edge of the EM data.** The imaged FIB-SEM volume ends at the bottom of cell 1 and truncates it (about 3-4% of the
  cell was not imaged). The model treats this flat face as an open boundary: no plasma membrane, no membrane proteins
  or lipids on it, nothing modelled beyond it. The viewer draws the face in flat grey (a grey line on cut faces) and
  labels points below it "edge of the EM data". Copy numbers are scaled to the modelled part of the cell.
- **Model v5 placement classes.** Nucleolar proteins (HPA main location nucleoli or fibrillar centre) have a share in
  the nucleolus. Lipid-anchored proteins (myristoyl, palmitoyl, prenyl, GPI) sit in a thin shell on the correct side of
  their membrane (GPI-anchored ones outside the cell). Actin-binding proteins annotated at the plasma membrane fill a
  cortical shell up to 150 nm under it, and the lamins a lamina shell under the inner nuclear membrane. Transmembrane
  proteins and complexes requested in soluble regions are routed to their membrane, and small proteins (< 60 kDa)
  annotated in both nucleus and cytoplasm get a nuclear pool. In the viewer each shell is its own compartment
  (Colour by: Compartment).
- **Model v6 placement classes.** Proteins annotated to nuclear speckles (e.g. SRSF splicing factors, SON, SRRM2) have a
  share in the speckle spheres (Colour by: Compartment, "nuclear speckle", magenta). Nuclear transport receptors and
  pore basket proteins fill a shell next to the nuclear pores, on both faces of the nuclear envelope, within 150 nm of a
  pore axis. The viewer has no separate compartment code left for it, so these proteins count as "nuclear pore (NPC and
  pore-proximal)".
- **Model v7.** Genome-mapped chromatin (46 chains with genome coordinates; RNA polymerase II, CTCF and cohesin
  pre-placed on it and shown as chromatin; nascent RNA drawn with the mRNA threads). A cisternal Golgi next to the
  centrosome: inside the EM Golgi volume, lumen layers 20 nm thick every 40 nm parallel to its surface; the space
  between cisternae is "Golgi intercisternal cytosol" (filled like cytosol). Microtubules traced in the EM volume. A
  lamin rod lattice under the inner nuclear membrane (shown as nuclear lamina). The new packing regions for the
  pericentriolar matrix, the vimentin cage and the microtubule-associated bands are shown as cytosol, and the NPC
  channel shell as nuclear pore, because the 5-bit region code has no free id. The whole-cell context mesh shows the
  Golgi envelope (cisternae are drawn in the 8 nm tile meshes only).
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
