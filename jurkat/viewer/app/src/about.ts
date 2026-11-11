export const ABOUT_HTML = `
<h2>About this model</h2>
<p><b>Jurkat T cell (OpenOrganelle jrc_jurkat-1, cell 1)</b>: a <span class="cubeSize">1.6 µm</span> sub-volume of a molecular model with ~900 million
macromolecules in total; this view streams the sub-volume at residue resolution plus whole-cell organelle outlines.</p>
<p>The cell shape and its organelles (plasma membrane, nuclear envelope with nuclear pores, mitochondria, ER, Golgi and
endo-lysosomal compartments) come from the segmented FIB-SEM volume of the human Jurkat T cell in OpenOrganelle
(jrc_jurkat-1). Every macromolecule is drawn from its experimental (PDB) or predicted (AlphaFold DB) coordinates, at one
bead per amino-acid residue or nucleotide when it is large enough on screen. Further away, molecules switch to a
sphere-tree stand-in and then to one sphere each. Click any molecule to inspect it and to load its full-atom model in Mol*.</p>
<p>The model of the whole cell is too large to stream into a browser, so this viewer shows one <span class="cubeSize">1.6 µm</span> cube that contains
plasma membrane, cytoplasm, nuclear envelope and nucleus, re-centred on the origin. The faces of the cube behave like cut
faces. The Organelles and Membranes views show decimated surfaces of the entire cell for context (zoom out to see them).</p>
<h3>Colours and controls</h3>
<ul>
<li>Cut faces are coloured by compartment (cytosol, nucleoplasm, nucleolus, nuclear speckles, perinuclear space,
mitochondrial matrix and intermembrane space with the cristae, ER, Golgi and endo-lysosomal lumen), with membranes as
light bands.</li>
<li><b>Flat bottom face = edge of the EM data.</b> The imaged FIB-SEM volume ends at the bottom of the cell and
cuts it off (about 3–4% of the cell was not imaged). The model stops there: the face has no membrane, is drawn in
flat grey (a grey line on cut faces), and nothing is modelled beyond it.</li>
<li>Molecules are coloured by the compartment they were placed in; ribosomes are pink (rRNA darker), chromatin violet,
microtubules green, actin orange, nuclear pores grey-blue. Nucleolar proteins are a deep violet; lipid-anchored
proteins (on the side of their membrane given by the anchor; GPI-anchored ones outside the cell), the actin cortex
under the plasma membrane and the nuclear lamina have their own shades, and nuclear speckle proteins are magenta
(Colour by: Compartment lists them all).</li>
<li>Lipids are generated in the browser near the camera on each membrane with one generic mammalian composition
(PC 45%, PE 20%, SM 10%, PS 8%, PI 5%, other 12%; cholesterol is not drawn).</li>
</ul>
<p>More detail is under <a href="#" id="aboutToModel" style="color:#8fc3ff">Model details</a>.</p>
`;

// whole-cell build (geometry.json whole_cell, jurkat/viewer/jenc_full.py)
export const ABOUT_HTML_WHOLE = `
<h2>About this model</h2>
<p><b>Jurkat T cell (OpenOrganelle jrc_jurkat-1, cell 1)</b>: a molecular model of the whole cell with ~900 million
macromolecules, streamed at residue resolution around the part you look at.</p>
<p>The cell shape and its organelles (plasma membrane, nuclear envelope with nuclear pores, mitochondria, ER, Golgi and
endo-lysosomal compartments) come from the segmented FIB-SEM volume of the human Jurkat T cell in OpenOrganelle
(jrc_jurkat-1). Every macromolecule is drawn from its experimental (PDB) or predicted (AlphaFold DB) coordinates, at one
bead per amino-acid residue or nucleotide when it is large enough on screen. Further away, molecules switch to a
sphere-tree stand-in and then to one sphere each. Click any molecule to inspect it and to load its full-atom model in Mol*.</p>
<p>The whole cell is far too large for a browser at once, so molecules are loaded in 200 nm tiles within a memory budget,
starting at the centre of the view; tiles further away are evicted. Until a tile's molecules arrive, the section shows
<i>overview stand-ins</i> (about a hundred molecules per tile, sampled by mass and enlarged, coloured by compartment) over
the compartment colours of the cut face. The Organelles and Membranes views show decimated surfaces of the entire cell.
A thinner slab loads a larger area at full detail.</p>
<p><b>Model v5</b> adds the nucleolus (from the EM nucleolus segmentation; proteins annotated as nucleolar get a share
there), procedural lamellar cristae in the mitochondria, lipid-anchored proteins placed next to their membrane, a cortical
shell under the plasma membrane for actin-binding proteins annotated there, the nuclear lamina under the inner nuclear
membrane, transmembrane proteins and complexes routed to their membrane, and a nuclear pool for small proteins annotated
in both nucleus and cytoplasm. The face where the EM volume truncates the cell is an open boundary without membrane.</p>
<p><b>Model v6</b> adds nuclear speckles: membraneless spheres in the nucleoplasm (about 0.5 µm across, between the
chromatin) holding proteins annotated to nuclear speckles, such as pre-mRNA splicing factors. It also adds a shell next
to the nuclear pores, on both faces of the nuclear envelope, for nuclear transport receptors and pore basket proteins.
These are shown with the nuclear pores.</p>
<p><b>Model v7</b> replaces the chromatin with genome-mapped chromatin: 46 continuous chromosome chains with genome
coordinates, RNA polymerase II, CTCF and cohesin at their mapped sites, and nascent RNA. The Golgi is now a stack of
cisternae next to the centrosome; the space between the cisternae is cytosol. The microtubules are traced from the EM
volume. v7 also adds a lattice of lamin rods under the nuclear envelope, a pericentriolar matrix, a vimentin cage
around the nucleus and bands of microtubule-associated proteins. Proteins in the last three are coloured as cytosol,
because the viewer has no compartment code left for them.</p>
<h3>Colours and controls</h3>
<ul>
<li>Cut faces are coloured by compartment (cytosol, nucleoplasm, nucleolus, nuclear speckles, perinuclear space,
mitochondrial matrix and intermembrane space with the cristae, ER, Golgi and endo-lysosomal lumen), with membranes as
light bands.</li>
<li><b>Flat bottom face = edge of the EM data.</b> The imaged FIB-SEM volume ends at the bottom of the cell and
cuts it off (about 3–4% of the cell was not imaged). The model stops there: the face has no membrane, is drawn in
flat grey (a grey line on cut faces), and nothing is modelled beyond it.</li>
<li>Molecules are coloured by the compartment they were placed in; ribosomes are pink (rRNA darker), chromatin violet,
microtubules green, actin orange, nuclear pores grey-blue. Nucleolar proteins are a deep violet; lipid-anchored
proteins (on the side of their membrane given by the anchor; GPI-anchored ones outside the cell), the actin cortex
under the plasma membrane and the nuclear lamina have their own shades, and nuclear speckle proteins are magenta
(Colour by: Compartment lists them all).</li>
<li>Lipids are generated in the browser near the camera on each membrane with one generic mammalian composition
(PC 45%, PE 20%, SM 10%, PS 8%, PI 5%, other 12%; cholesterol is not drawn).</li>
</ul>
<p>More detail is under <a href="#" id="aboutToModel" style="color:#8fc3ff">Model details</a>.</p>
`;
