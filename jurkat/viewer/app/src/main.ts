import { Renderer, FrameState, ID_INST, ID_LIPID, ID_FREE } from './renderer';
import { Camera, CamState } from './camera';
import { TileManager, LoadedTile } from './tiles';
import { Geometry, prepareGeometry } from './geometry';
import { loadLibrary, loadTileIndex, fetchJson, fetchBin, Library, TileIndex, REGION_NAMES, MEM_NAMES, LABEL_NAMES, PAL, specialFor, idMatches, b64ToBytes, moduleColor, MEM_REGIONS, SOLUBLE_REGIONS } from './data';
import { loadFields } from './fields';
import { V3, dot, norm, cross, sub, add, scale, len } from './math';
import { ABOUT_HTML, ABOUT_HTML_WHOLE } from './about';
import { markdownToHtml } from './markdown';
import { Dynamics } from './dynamics';
import MODEL_MD from '../../MODEL.md?raw';     // jurkat/viewer/MODEL.md

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const fmt = (n: number) => n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 1 : 2) + ' M' : n >= 1e4 ? (n / 1e3).toFixed(0) + ' k' : n.toLocaleString('en-US');
const fmtNm = (n: number) => n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 2).replace(/\.?0+$/, '') + ' µm' : (n >= 10 ? n.toFixed(0) : n.toFixed(1)) + ' nm';

declare global { interface Window { yc: any; } }

interface Bookmark { name: string; caption: string; target: V3; normal: V3; up: V3; width: number; slab: number; dynamics?: string }

let lib: Library, idx: TileIndex, geo: any, geom: Geometry, bookmarks: Bookmark[], typesTiles: { off: Uint32Array; tiles: Uint16Array | Uint32Array };
// whole-cell build (jenc_full.py) or the 1.6 um sub-volume (jenc.py): wording of counts
const scopeWord = () => (geo?.whole_cell ? 'in the cell' : 'in the sub-volume');
let R: Renderer, T: TileManager, dyn: Dynamics;
const cam = new Camera();
const canvas = $<HTMLCanvasElement>('gpu');

const fs: FrameState = {
  cutOn: true, plane: [0, 0, 1, 0], slab: 60, colorMode: 0, regionMask: 0xFFFFFF, freeHide: 0, highlight: false,
  selId: 0, shaded: true, organelle: false, motion: 0, l0px: 1.6, l1px: 2.5, minpx: 0.3, outline: 1, showMeshes: true, showLipids: true, renderScale: 1, memColors: false,
};
let planeFollow = false;
// tracer dynamics (offline validation, jurkat/validate/dynamics.py): UI, keys and "Diffusion" bookmarks only when the
// data folder has dynamics.json (the whole-cell public build has none)
let hasDyn = false;
let dirty = true, lastUpdate = 0, lastFrameT = 0, quality = 'auto';
let typeFlags: Uint32Array<ArrayBuffer>;
// molecule classes with their own Show switch (library type ids of the Jurkat build; a trailing '*' matches a prefix)
const CLASS_TYPES: Record<string, string[]> = {
  actin: ['F_*'],
  microtubules: ['MT_*'],
  ribosomes: ['RIBO*'],
  nucleosomes: ['NUCLEOSOME', 'H1-*'],
  pores: ['NPC'],
};
// connected assemblies (filament subunits, pores, chromatin): illustrative motion would pull them apart
const NO_MOTION = [...CLASS_TYPES.actin, ...CLASS_TYPES.microtubules, 'NPC', ...CLASS_TYPES.nucleosomes];
/** library type indices whose id matches one of the patterns */
function typesMatching(patterns: string[]): number[] {
  const out: number[] = [];
  lib.types.forEach((t, i) => { if (idMatches(t.id, patterns)) out.push(i); });
  return out;
}
let selType = -1;
let selection: any = null;
let tour: { i: number; phase: 'fly' | 'wait' | 'hold'; t: number } | null = null;
let sweep: { t0: number; d0: number; d1: number; dur: number } | null = null;
const frameTimes: number[] = [];

/** molecule budget: ?detail=low|medium|high|auto, else the saved choice (Performance panel), else auto */
function detailPref(): string {
  const ok = ['auto', 'low', 'medium', 'high'];
  const q = new URLSearchParams(location.search).get('detail');
  if (q && ok.includes(q)) return q;
  try { const v = localStorage.getItem('yc.detail'); if (v && ok.includes(v)) return v; } catch { /* storage unavailable */ }
  return 'auto';
}
function setLoading(msg: string, f: number) {
  ($('loading').querySelector('.m') as HTMLElement).textContent = msg;
  ($('loading').querySelector('.pb div') as HTMLElement).style.width = `${Math.round(f * 100)}%`;
}
function fail(msg: string) {
  const e = $('loading').querySelector('.err') as HTMLElement;
  e.style.display = 'block'; e.innerHTML = msg;
  ($('loading').querySelector('.t') as HTMLElement).textContent = 'Could not start the viewer';
}
function toast(msg: string, ms = 2200) {
  const t = $('toast'); t.textContent = msg; t.style.display = 'block';
  clearTimeout((toast as any)._t); (toast as any)._t = setTimeout(() => (t.style.display = 'none'), ms);
}

// ---------------------------------------------------------------- boot
async function boot() {
  if (!('gpu' in navigator)) {
    fail(`This browser does not expose <b>WebGPU</b>. Use a current Chrome or Edge (version 113 or later), or Safari 26 or later.
      On Linux Chrome, enable <code>chrome://flags/#enable-unsafe-webgpu</code> and <code>#enable-vulkan</code>.
      Firefox 141+ supports WebGPU on Windows.<br><br>Also check that the page is served over <code>http://localhost</code>
      (for example <code>python -m http.server</code>). Opening the file directly with <code>file://</code> does not work.`);
    return;
  }
  try {
    setLoading('Loading cell geometry', 0.01);
    [geo, bookmarks] = await Promise.all([fetchJson('geometry.json'), fetchJson('bookmarks.json')]);
    geo.F = await loadFields(geo, fetchBin);      // distance fields of the sub-volume
    prepareGeometry(geo);                         // Jurkat: renderer/shader members (G, pore grid) from the fields geometry
    geom = new Geometry(geo);
    // whole-cell organelle surfaces (shifted by the sub-volume centre) must fit in the view when zoomed out
    if (geo.R) cam.maxDist = Math.max(cam.maxDist, 3 * (geo.R + Math.hypot(...((geo.centre_nm ?? [0, 0, 0]) as [number, number, number]))));
    idx = await loadTileIndex();
    lib = await loadLibrary(setLoading);
    const tt = await fetchBin('types_tiles.bin');
    const NT = lib.types.length;
    typesTiles = { off: new Uint32Array(tt, 0, NT + 1), tiles: idx.typesTilesU32 ? new Uint32Array(tt, (NT + 1) * 4) : new Uint16Array(tt, (NT + 1) * 4) };
    setLoading('Starting WebGPU', 0.7);
    R = new Renderer(canvas);
    R.detail = detailPref();
    R.onLost = (m) => { toast(m, 10000); };
    await R.init(lib, geo);
    typeFlags = new Uint32Array(NT);
    for (const ids of Object.values(CLASS_TYPES)) for (const i of typesMatching(ids)) typeFlags[i] |= 4;
    for (const i of typesMatching(NO_MOTION)) typeFlags[i] |= 8;
    R.device.queue.writeBuffer(R.typeFlags, 0, typeFlags);
    setLoading('Compiling shaders and starting tile streaming', 0.9);
    let embedded: Record<string, Uint8Array> | undefined;
    if (window.__YC_EMBED__) {
      embedded = {};
      for (const [k, v] of Object.entries(window.__YC_EMBED__)) if (k.startsWith('tiles/') || k.startsWith('library/')) embedded[k] = b64ToBytes(v);
    }
    const nW = Math.max(2, Math.min(4, (navigator.hardwareConcurrency || 4) - 1));
    T = new TileManager(idx, R, lib, geo, nW, embedded);
    T.onChange = () => { dirty = true; };
    dyn = new Dynamics(R, () => { dirty = true; }, toast);
    dyn.onRegion = (r) => { const i = bookmarks.findIndex(b => b.dynamics === r); if (i >= 0) { stopTour(); applyBookmark(i); } else dyn.show(r); };
    hasDyn = !!(await dyn.load());
    if (!hasDyn) bookmarks = bookmarks.filter(b => !b.dynamics);
    buildUI();
    const qv = new URLSearchParams(location.search).get('view');
    const v0 = qv === null ? 0 : (isNaN(Number(qv)) ? Math.max(0, bookmarks.findIndex(b => b.name.toLowerCase().startsWith(qv.toLowerCase()))) : Math.min(bookmarks.length - 1, Math.max(0, Number(qv))));
    applyBookmark(v0, false);
    $('loading').style.display = 'none';
    requestAnimationFrame(loop);
    window.yc = { R, T, cam, fs, lib, idx, applyBookmark, setType, setModule, pickAt, dyn, bookmarks, setQuality, describe, ready: true, state: () => ({ pending: T.pendingForView(), detailPending: T.detInflight, loaded: T.loaded.size, totals: T.totals(), stats: R.stats, cam: { t: cam.target, d: cam.dist }, plane: fs.plane, gpuMs, gpuBusy, dirty }), render: () => { dirty = true; } };
  } catch (e: any) {
    console.error(e);
    fail(`${String(e?.message || e)}<br><br>See the browser console for details.`);
  }
}

// ---------------------------------------------------------------- plane helpers
function planeN(): V3 { return [fs.plane[0], fs.plane[1], fs.plane[2]]; }
function setPlane(n: V3, d: number) { n = norm(n); fs.plane = [n[0], n[1], n[2], d]; syncPlaneUI(); dirty = true; }
function alignToView() { const n = cam.fwd; setPlane(n, dot(n, cam.target)); }
// move the plane (and the camera with it) along its normal; nudgeStep is the user's fine step in nm
let nudgeStep = 0.5;
try { const s = Number(localStorage.getItem('yc.nudgeStep')); if (s > 0) nudgeStep = s; } catch { /* storage unavailable */ }
function movePlane(s: number) { const n = planeN(); cam.target = add(cam.target, scale(n, s)); setPlane(n, fs.plane[3] + s); }

// ---------------------------------------------------------------- bookmarks / tour
function bookmarkState(b: Bookmark): CamState {
  const aspect = canvas.clientWidth / canvas.clientHeight;
  const n = norm(b.normal);
  let up = norm(sub(b.up, scale(n, dot(b.up, n))));
  return { target: [...b.target] as V3, dist: cam.distForWidth(b.width, aspect), fwd: n, up };
}
function applyBookmark(i: number, animate = true, onDone?: () => void) {
  const b = bookmarks[i];
  if (!b) return;
  if ((fs.organelle || fs.memColors) && i !== 0) setDisplayMode('molecules');
  if (b.dynamics) dyn.show(b.dynamics); else dyn.hide();
  const st = bookmarkState(b);
  const n1 = norm(b.normal), d1 = dot(n1, b.target);
  const n0 = planeN(), d0 = fs.plane[3], s0 = fs.slab;
  fs.cutOn = true; ($('cutOn') as HTMLInputElement).checked = true;
  planeFollow = false; $('followView').classList.remove('on');
  if (!animate) { cam.set(st); setPlane(n1, d1); fs.slab = b.slab; syncPlaneUI(); dirty = true; onDone?.(); return; }
  cam.flyTo(st, 2600, () => { setPlane(n1, d1); fs.slab = b.slab; syncPlaneUI(); onDone?.(); }, (e) => {
    let n = norm(add(scale(n0, 1 - e), scale(n1, e)));
    if (len(add(scale(n0, 1 - e), scale(n1, e))) < 1e-3) n = n1;
    fs.plane = [n[0], n[1], n[2], d0 + (d1 - d0) * e];
    fs.slab = Math.exp(Math.log(s0) + (Math.log(b.slab) - Math.log(s0)) * e);
    syncPlaneUI();
  });
  showCaption(b);
  dirty = true;
}
function showCaption(b: Bookmark | null) {
  const c = $('caption');
  if (!b) { c.style.display = 'none'; return; }
  c.innerHTML = `<b>${b.name}</b>${b.caption}`; c.style.display = 'block';
  clearTimeout((showCaption as any)._t);
  if (!tour) (showCaption as any)._t = setTimeout(() => (c.style.display = 'none'), 9000);
}
function startTour() {
  tour = { i: 0, phase: 'fly', t: performance.now() };
  $('tourBtn').textContent = '■ Stop'; $('tourBtn').classList.add('on');
  applyBookmark(0, true, () => { if (tour) { tour.phase = 'wait'; tour.t = performance.now(); } });
}
function stopTour() {
  tour = null; $('tourBtn').textContent = '▶ Tour'; $('tourBtn').classList.remove('on');
  showCaption(null);
}
function tourStep(now: number) {
  if (!tour) return;
  if (tour.phase === 'wait' && (T.pendingForView() === 0 || now - tour.t > 14000)) { tour.phase = 'hold'; tour.t = now; }
  else if (tour.phase === 'hold' && now - tour.t > 6500) {
    tour.i = (tour.i + 1) % bookmarks.length; tour.phase = 'fly';
    const i = tour.i;
    applyBookmark(i, true, () => { if (tour && tour.i === i) { tour.phase = 'wait'; tour.t = performance.now(); } });
  }
}

// ---------------------------------------------------------------- selection / search
let selModule = -1;
/** highlight every type of one interaction module (legend of the module colouring); -1 clears */
function setModule(m: number) {
  if (selType >= 0) { typeFlags[selType] &= ~1; selType = -1; renderSelType(); }
  selModule = m;
  for (let i = 0; i < typeFlags.length; i++) {
    if (m >= 0 && lib.moduleOf[i] === m) typeFlags[i] |= 1; else typeFlags[i] &= ~1;
  }
  fs.highlight = m >= 0;
  R.device.queue.writeBuffer(R.typeFlags, 0, typeFlags);
  renderLegend();
  dirty = true;
}
function setType(t: number) {
  if (selModule >= 0) setModule(-1);
  if (selType >= 0) typeFlags[selType] &= ~1;
  selType = t;
  if (t >= 0) typeFlags[t] |= 1;
  fs.highlight = t >= 0;
  R.device.queue.writeBuffer(R.typeFlags, 0, typeFlags);
  renderSelType();
  dirty = true;
}
function renderSelType() {
  const el = $('selType');
  if (selType < 0) { el.style.display = 'none'; return; }
  const t = lib.types[selType];
  const loaded = T.countLoaded(selType);
  el.style.display = 'block';
  el.innerHTML = `<div><b>${t.gene}</b> ${t.name}</div>
    <div class="sub" style="margin:3px 0 6px">${fmt(t.copies)} copies ${scopeWord()} · ${fmt(loaded)} in loaded tiles · ${srcLabel(t)}</div>
    <div class="grid2"><button class="small" id="flyNearest">Fly to nearest</button><button class="small" id="clearType">Clear highlight</button></div>`;
  $('flyNearest').onclick = () => flyToNearest(selType);
  $('clearType').onclick = () => setType(-1);
}
function searchTypes(q: string) {
  q = q.trim().toLowerCase();
  if (!q) return [];
  const out: { i: number; s: number }[] = [];
  lib.types.forEach((t, i) => {
    const g = t.gene.toLowerCase(), id = t.id.toLowerCase(), u = (t.uniprot || '').toLowerCase(), nm = (t.name || '').toLowerCase(), gs = (t.genes || '').toLowerCase();
    let s = 0;
    if (g === q || id === q || u === q) s = 100;
    else if (g.startsWith(q) || id.startsWith(q)) s = 60;
    else if (gs.split(/[;,\s]+/).includes(q)) s = 55;
    else if (nm.includes(q)) s = 30 + (nm.startsWith(q) ? 10 : 0);
    else if (gs.includes(q) || u.startsWith(q) || (t.pdb || '').toLowerCase() === q) s = 20;
    if (s) out.push({ i, s: s + Math.log10(t.copies + 1) });
  });
  out.sort((a, b) => b.s - a.s);
  return out.slice(0, 14).map(x => x.i);
}
function flyToNearest(type: number) {
  const o0 = typesTiles.off[type], o1 = typesTiles.off[type + 1];
  if (o1 <= o0) { toast('No copies of this molecule were placed.'); return; }
  let best = -1, bd = 1e18;
  for (let k = o0; k < o1; k++) {
    const ti = typesTiles.tiles[k], c = T.tileCenter(ti);
    const d = len(sub(c, cam.target));
    if (d < bd) { bd = d; best = ti; }
  }
  const c = T.tileCenter(best);
  const t = lib.types[type];
  const width = Math.max(90, t.br * 9);
  const goTo = (tile: LoadedTile) => {
    const k = T.nearestInTile(tile, type, c);
    if (k < 0) return;
    const d = T.decode(tile, k);
    const n = planeN();
    const st = cam.state(); st.target = d.p; st.dist = cam.distForWidth(width, canvas.clientWidth / canvas.clientHeight);
    cam.flyTo(st, 900, () => { setPlane(n, dot(n, d.p) - t.br * 0.4); });
    fs.plane[3] = dot(n, d.p) - t.br * 0.4;
    const pid = T.poolIndex(tile, k);
    select({ kind: 'inst', id: (ID_INST | pid) >>> 0, info: T.instanceAt(pid) });
  };
  const loaded = T.loaded.get(best);
  const n = planeN();
  const st = cam.state(); st.target = c; st.dist = cam.distForWidth(width * 2.5, canvas.clientWidth / canvas.clientHeight);
  fs.cutOn = true; ($('cutOn') as HTMLInputElement).checked = true;
  if (fs.organelle || fs.memColors) setDisplayMode('molecules');
  cam.flyTo(st, 2200, () => {
    setPlane(n, dot(n, c) - 20);
    const L = T.loaded.get(best);
    if (L) goTo(L);
    else {
      const h = (tile: LoadedTile) => { if (tile.idx === best) { T.onTileLoaded.splice(T.onTileLoaded.indexOf(h), 1); goTo(tile); } };
      T.onTileLoaded.push(h);
    }
  });
  setPlane(n, dot(n, c) - 20);
  if (loaded) { /* handled on arrival of flight */ }
}

// ---------------------------------------------------------------- picking & inspection
async function pickAt(cx: number, cy: number) {
  const dpr = R.width / canvas.clientWidth;
  return R.pick(cx * dpr, cy * dpr);
}
function srcLabel(t: Library['types'][number]) {
  if (t.src === 'PDB') return 'PDB ' + t.pdb;
  if (t.src === 'AF-multimer') return 'AlphaFold complex';
  if (t.src === 'AF') return t.genes && t.genes.includes(';') ? 'AlphaFold complex' : 'AlphaFold';
  return 'assembly unit';      // pre-placed building blocks (filament subunits, chromatin units) without one source entry
}
/** whether an atomic model can be fetched for the type (PDB entry, or an AlphaFold DB entry by model or UniProt id) */
function hasAtomicModel(t: Library['types'][number]) {
  return (t.src === 'PDB' && !!t.pdb) || !!t.model || (t.src === 'AF' && !!t.uniprot && !(t.genes || '').includes(';'));
}
function describe(id: number): any {
  const cat = id >>> 30;
  if (id === 0) return { kind: 'none', title: 'Outside the cell' };
  if (cat === 1) {
    const info = T.instanceAt(id & 0x3FFFFFFF);
    if (!info) return { kind: 'none', title: '' };
    return { kind: 'inst', id, info, title: lib.types[info.type].gene, sub: REGION_NAMES[info.region] };
  }
  if (cat === 2) {
    const mem = (id >>> 1) & 15, leaf = id & 1;
    const name = MEM_NAMES[mem] || 'membrane';
    const side = leaf ? 'cytoplasmic leaflet' : (mem === 0 ? 'exoplasmic (outer) leaflet' : 'lumenal leaflet');
    return { kind: 'lipid', id, mem, leaf, title: 'Lipid bilayer', sub: `${name} · ${side}` };
  }
  if (cat === 3) {
    const kind = (id >>> 15) & 7, chain = id & 0x7FFF;
    if (kind === 6) return { kind: 'free', id, fk: kind, chain, ...dyn.describe(chain) };     // tracer dynamics overlay
    if (kind === 7) return { kind: 'free', id, fk: kind, chain, title: 'Overview stand-in',     // whole-cell overview (tiles.ts Overview)
      sub: `${REGION_NAMES[chain & 31] ?? ''} · one sampled molecule standing for its neighbourhood; zoom in or wait for the molecules to load` };
    const nm = ['mRNA', 'Linker DNA', 'β-1,3-glucan fibril', 'Mannan chain', 'pre-rRNA', 'FG repeats'][kind] ?? 'polymer';
    const sub = ['nucleotide-resolution thread (schematic path)', 'between consecutive nucleosomes, 2 bp per bead', 'procedural, 36 glucose per bead',
      'procedural α-mannan on a cell-wall mannoprotein', 'nascent ribosomal RNA', 'disordered domain of an FG nucleoporin (NSP1, NUP49, NUP57, NUP159), one bead per residue'][kind] ?? '';
    return { kind: 'free', id, fk: kind, chain, title: nm, sub };
  }
  if (id >= 0x200 && id < 0x210) return { kind: 'band', id, title: 'Membrane (cut face)', sub: MEM_NAMES[id - 0x200] };
  if (id >= 0x100 && id < 0x110) return { kind: 'label', id, title: LABEL_NAMES[id - 0x100], sub: 'compartment (not loaded at this distance, or beyond the slab)' };
  return { kind: 'none', title: '' };
}
function select(d: any) {
  selection = d;
  fs.selId = d && d.kind === 'inst' ? d.id >>> 0 : 0;
  renderInspect();
  dirty = true;
}
function renderInspect() {
  const el = $('inspect');
  const d = selection;
  if (!d || d.kind === 'none') { el.style.display = 'none'; return; }
  el.style.display = 'block';
  let h = `<button class="close" id="inspClose">×</button>`;
  if (d.kind === 'inst') {
    const t = lib.types[d.info.type];
    const isPDB = t.src === 'PDB';
    const nb = t.nb, nnt = t.nnt;
    h += `<h2>${t.gene}</h2><div>${t.name}</div><div style="margin-top:6px"><span class="badge ${isPDB ? 'pdb' : 'af'}">${srcLabel(t)}</span>
      ${t.oligomer > 1 ? `<span class="badge" style="background:#666">${t.oligomer}-mer</span>` : ''}</div>
      <div class="kv">
      ${t.genes && t.genes !== t.gene ? `<div>Subunits</div><div style="word-break:break-word">${t.genes.split(/;\s*/).slice(0, 40).join(', ')}${t.genes.split(';').length > 40 ? ' …' : ''}</div>` : ''}
      ${t.uniprot ? `<div>UniProt</div><div>${t.uniprot}</div>` : ''}
      <div>Location</div><div>${REGION_NAMES[d.info.region]}</div>
      <div>Copies here</div><div>${t.copies.toLocaleString('en-US')} ${scopeWord()}</div>
      <div>Mass</div><div>${(t.mass / 1000).toFixed(1)} kDa</div>
      <div>Residues drawn</div><div>${(nb - nnt).toLocaleString('en-US')} aa${nnt ? ` + ${nnt.toLocaleString('en-US')} nt` : ''}</div>
      <div>Radius</div><div>${t.br.toFixed(1)} nm (Rg ${t.rg.toFixed(1)} nm)</div>
      ${!isPDB && t.plddt_mean != null ? `<div>Mean pLDDT</div><div>${t.plddt_mean.toFixed(1)} (${Math.round((t.plddt_frac_kept ?? 1) * 100)}% of residues kept, pLDDT ≥ 50)</div>` : ''}
      <div>Position</div><div>${d.info.p.map((x: number) => x.toFixed(1)).join(', ')} nm</div>
      </div>
      <div class="grid2"><button class="small" id="loadAtoms"${hasAtomicModel(t) ? '' : ' disabled title="No single PDB or AlphaFold DB entry for this building block"'}>Load atomic model</button><button class="small" id="hlType">Highlight all</button>
      <button class="small" id="centerOn">Center view</button><button class="small" id="hideType">${typeFlags[d.info.type] & 2 ? 'Show' : 'Hide'} this type</button></div>
      <div style="margin-top:9px">${t.uniprot ? `<a href="https://www.uniprot.org/uniprotkb/${t.uniprot}" target="_blank">UniProt</a>` : ''}
      ${isPDB ? `<a href="https://www.rcsb.org/structure/${t.pdb}" target="_blank">RCSB ${t.pdb}</a>` : ''}
      ${hasAtomicModel(t) && !isPDB ? `<a href="https://alphafold.ebi.ac.uk/entry/${t.model || t.uniprot}" target="_blank">AlphaFold DB</a>` : ''}
      ${!t.uniprot && t.genes && !t.genes.includes(';') && /^[A-Z0-9-]+$/.test(t.gene) && !specialFor(t.id) ? `<a href="https://www.uniprot.org/uniprotkb?query=gene_exact:${encodeURIComponent(t.gene)}+AND+organism_id:9606" target="_blank">UniProt search</a>` : ''}</div>`;
  } else {
    h += `<h2>${d.title}</h2><div class="sub" style="font-size:12.5px">${d.sub || ''}</div>`;
    if (d.kind === 'lipid') h += `<p class="sub">Lipids are generated in your browser on the mesh of this membrane, with one generic mammalian composition (PC 45%, PE 20%, SM 10%, PS 8%, PI 5%, other 12%; cholesterol is not drawn). They sit at 0.65 nm² per lipid and are removed where they would overlap protein.</p>`;
  }
  el.innerHTML = h;
  $('inspClose').onclick = () => select(null);
  if (d.kind === 'inst') {
    $('loadAtoms').onclick = () => openMolstar(d.info.type);
    $('hlType').onclick = () => setType(d.info.type);
    $('centerOn').onclick = () => { const st = cam.state(); st.target = d.info.p; cam.flyTo(st, 700); };
    $('hideType').onclick = () => { typeFlags[d.info.type] ^= 2; R.device.queue.writeBuffer(R.typeFlags, 0, typeFlags); renderInspect(); dirty = true; };
  }
}

// ---------------------------------------------------------------- Mol*
let molViewer: any = null, molLoading: Promise<void> | null = null;
function loadMolstarLib() {
  if (molLoading) return molLoading;
  const CDN = 'https://cdn.jsdelivr.net/npm/molstar@4.18.0/build/viewer/';
  const load = (base: string) => new Promise<void>((res, rej) => {
    const l = document.createElement('link'); l.rel = 'stylesheet'; l.href = base + 'molstar.css'; document.head.appendChild(l);
    const s = document.createElement('script'); s.src = base + 'molstar.js'; s.onload = () => res(); s.onerror = () => { l.remove(); s.remove(); rej(new Error('Mol* failed to load')); };
    document.head.appendChild(s);
  });
  // local copy first (offline use); CDN fallback for the single-file preview
  molLoading = (window.__YC_EMBED__ ? load(CDN) : load('molstar/').catch(() => load(CDN))).catch((e) => { molLoading = null; throw e; });
  return molLoading;
}
async function openMolstar(type: number) {
  const t = lib.types[type];
  $('molModal').style.display = 'flex';
  $('molTitle').textContent = `${t.gene} · ${t.name} · loading…`;
  try {
    await loadMolstarLib();
    const molstar = (window as any).molstar;
    if (!molViewer) {
      molViewer = await molstar.Viewer.create('molHost', {
        layoutIsExpanded: false, layoutShowControls: false, layoutShowRemoteState: false, layoutShowSequence: true,
        layoutShowLog: false, layoutShowLeftPanel: false, viewportShowExpand: false, viewportShowSelectionMode: false, viewportShowAnimation: false,
        collapseLeftPanel: true, pdbProvider: 'rcsb', emdbProvider: 'rcsb',
      });
    } else await molViewer.plugin.clear();
    let url = '', link = '';
    if (t.src === 'PDB' && t.pdb) {
      url = `https://files.rcsb.org/download/${t.pdb}-assembly1.cif`; link = `https://www.rcsb.org/structure/${t.pdb}`;
      try { const h = await fetch(url, { method: 'HEAD' }); if (!h.ok) url = `https://files.rcsb.org/download/${t.pdb}.cif`; } catch { url = `https://files.rcsb.org/download/${t.pdb}.cif`; }
    } else {
      const entry = t.model || t.uniprot;   // predicted complexes are AFDB entries of their own
      link = `https://alphafold.ebi.ac.uk/entry/${entry}`;
      try {
        const r = await fetch(`https://alphafold.ebi.ac.uk/api/prediction/${entry}`);
        const j = await r.json(); url = (Array.isArray(j) ? j[0] : j)?.cifUrl;
      } catch { /* fall through */ }
      if (!url) url = t.model ? `https://alphafold.ebi.ac.uk/files/${t.model}-model_v1.cif` : `https://alphafold.ebi.ac.uk/files/AF-${t.uniprot}-F1-model_v6.cif`;
    }
    await molViewer.loadStructureFromUrl(url, 'mmcif', false);
    $('molTitle').textContent = `${t.gene} · ${t.name} · ${t.src === 'PDB' ? 'PDB ' + t.pdb + ' (biological assembly)' : t.model ? `AlphaFold DB predicted ${t.oligomer}-mer (full length, incl. low-pLDDT regions)` : 'AlphaFold DB model (full length, incl. low-pLDDT regions)'}`;
    const a = $<HTMLAnchorElement>('molLink'); a.href = link; a.textContent = 'Open source page';
  } catch (e: any) {
    $('molTitle').textContent = `Could not load the atomic model: ${e?.message || e}. Internet access to RCSB/AlphaFold DB is needed.`;
  }
}

// ---------------------------------------------------------------- display modes
let orgMeshesRequested = false;
function loadOrganelleMeshes() {
  if (orgMeshesRequested) return;
  orgMeshesRequested = true;
  toast('Loading whole-cell organelle surfaces…');
  fetchBin('organelle_meshes.bin.gz').then(b => { R.uploadOrganelleMeshes(b); dirty = true; });
}
// membranes view = the Show list with only Membranes ticked (fs.memColors); set up with the Show list
let setMembraneView: (on: boolean) => void = () => {};
function syncModeButtons() {
  const mem = fs.memColors && !fs.organelle;
  $('modeSection').classList.toggle('on', !fs.organelle && !mem); $('modeOrg').classList.toggle('on', fs.organelle);
  $('modeMem').classList.toggle('on', mem);
}
function setOrganelle(on: boolean) {
  fs.organelle = on;
  if (on) loadOrganelleMeshes();
  syncModeButtons();
  dirty = true;
}
function setDisplayMode(m: 'molecules' | 'organelles' | 'membranes') {
  if (m !== 'membranes' && fs.memColors) setMembraneView(false);
  setOrganelle(m === 'organelles');
  if (m === 'membranes') setMembraneView(true);
}
function setQuality(q: string) {
  quality = q;
  const P: Record<string, [number, number]> = { low: [2.6, 0.75], med: [1.6, 1], high: [1.0, 1], ultra: [0.6, 1], auto: [1.6, 1] };
  const [l0, sc] = P[q];
  fs.l0px = l0; fs.renderScale = sc; R.resize(sc); dirty = true;
}
function renderLegend() {
  const el = $('legend');
  const sw = (c: string, l: string) => `<div><span class="sw" style="background:${c}"></span>${l}</div>`;
  const m = fs.colorMode;
  if (m === 0) el.innerHTML = [sw(PAL[0], 'cytosolic protein'), sw(PAL[1], 'nuclear protein'), sw('#EE9BC0', 'ribosome protein'), sw('#C9558F', 'rRNA'),
    sw('#A98BCB', 'histone'), sw('#7A4FA0', 'DNA'), sw('#8E6CB0', 'linker histone H1'), sw(PAL[17], 'nuclear pore'), sw('#E0893A', 'actin filament'), sw('#74C476', 'microtubule'),
    sw(PAL[7], 'PM protein'), sw(PAL[8], 'NE protein'), sw(PAL[9], 'ER membrane protein'), sw(PAL[10], 'mito membrane protein'),
    sw(PAL[2], 'mito matrix'), sw(PAL[3], 'ER lumen'), sw(PAL[4], 'Golgi'), sw(PAL[5], 'endo-lysosomal'), sw('#F3C643', 'lipid head'), sw('#F8E3A0', 'lipid tail')].join('');
  else if (m === 1) el.innerHTML = sw('#4C8CD9', 'experimental (PDB)') + sw('#F5A84C', 'AlphaFold model');
  else if (m === 2) el.innerHTML = sw('#0053D6', 'pLDDT > 90') + sw('#65CBF3', '70–90') + sw('#FFDB13', '50–70') + sw('#B8B8B8', 'experimental (PDB)');
  else if (m === 3) el.innerHTML = `<div style="grid-column: 1 / 3"><div style="height:8px;border-radius:3px;background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)"></div><div style="display:flex;justify-content:space-between"><span>1</span><span>10³</span><span>10⁶ copies</span></div></div>`;
  else if (m === 6) {
    el.innerHTML = lib.modules.length
      ? '<div class="sub" style="grid-column: 1 / 3">Click a module to highlight it</div>' +
        lib.modules.map((x, i) => `<div class="modsw${selModule === i ? ' on' : ''}" data-m="${i}" title="${x.n} proteins"><span class="sw" style="background:${moduleColor(i)}"></span>${x.label}</div>`).join('') +
        sw('#D1D1CC', 'no module')
      : '<div style="grid-column: 1 / 3">No interaction modules in this build</div>';
    el.querySelectorAll<HTMLElement>('.modsw').forEach(d => d.onclick = () => { const i = Number(d.dataset.m); setModule(selModule === i ? -1 : i); });
  }
  else if (m === 5) el.innerHTML = REGION_NAMES.map((n, i) => sw(PAL[i], n)).join('');
  else el.innerHTML = '';
}

// ---------------------------------------------------------------- UI construction
/** box holding the model: the whole cell's bounding box (cell distance field) or the sub-volume cube */
function modelBox(): { lo: V3; hi: V3 } {
  const h = geo?.fields?.head?.[0];
  if (geo?.whole_cell && h) {
    const lo: V3 = [h[0], h[1], h[2]], e = h[3] * h[7];
    return { lo, hi: [lo[0] + h[4] * e, lo[1] + h[5] * e, lo[2] + h[6] * e] };
  }
  const a = geo?.half ?? 2200;
  return { lo: [-a, -a, -a], hi: [a, a, a] };
}
/** range of the plane offset d = n . p over the model box (+ 100 nm) for normal n: the Position slider and the sweep */
function planeRange(n: V3): [number, number] {
  const B = modelBox(); let lo = Infinity, hi = -Infinity;
  for (let k = 0; k < 8; k++) {
    const p: V3 = [k & 1 ? B.hi[0] : B.lo[0], k & 2 ? B.hi[1] : B.lo[1], k & 4 ? B.hi[2] : B.lo[2]];
    const v = dot(n, p); lo = Math.min(lo, v); hi = Math.max(hi, v);
  }
  return [Math.floor(lo - 100), Math.ceil(hi + 100)];
}
function syncPlaneUI() {
  const d = fs.plane[3];
  const [rlo, rhi] = planeRange(planeN()), pd = $('planeD') as HTMLInputElement;
  pd.min = String(rlo); pd.max = String(rhi);           // the slider follows the plane normal (axis buttons, Align to view)
  pd.value = String(Math.round(d));
  $('planeDv').textContent = `${Number(d.toFixed(nudgeStep < 0.5 ? 2 : 1))} nm`;
  const s = fs.slab;
  ($('slab') as HTMLInputElement).value = String(Math.round(Math.log(s / 5) / Math.log(4400 / 5) * 1000));
  $('slabv').textContent = fmtNm(s);
  ['axX', 'axY', 'axZ'].forEach((id, k) => $(id).classList.toggle('on', Math.abs(fs.plane[k]) > 0.999));
  mmDirty = true;
}
function buildUI() {
  const NT = lib.types.length;
  let total = 0; for (const t of lib.types) total += t.copies;
  const cubeUm = geo.cube_nm ? (geo.cube_nm / 1000).toFixed(1) : '1.6';
  $('subCount').textContent = geo.whole_cell
    ? `${(total / 1e6).toFixed(0)} M macromolecules of ${NT.toLocaleString('en-US')} species in the whole cell (${idx.n.toLocaleString('en-US')} tiles, streamed around the view), each a PDB or AlphaFold structure`
    : `${(total / 1e6).toFixed(1)} M macromolecules of ${NT.toLocaleString('en-US')} species in this ${cubeUm} µm sub-volume (${idx.n} tiles), each a PDB or AlphaFold structure`;
  // features of the yeast build that the Jurkat data do not have: hide their options
  const cm = $<HTMLSelectElement>('colorMode');
  const hideOpt = (v: string) => { const o = cm.querySelector(`option[value="${v}"]`) as HTMLOptionElement | null; if (o) o.remove(); };
  if (lib.types.every(t => t.plddt_mean == null)) hideOpt('2');      // no pLDDT in this build
  if (!lib.modules.length) hideOpt('6');                              // no organisation.json
  if ((window as any).__YC_PREVIEW__) {
    const nI = idx.nInst.reduce((a, b) => a + b, 0);
    $('subCount').innerHTML = `<b>Preview:</b> ${idx.n} tiles (${(nI / 1e3).toFixed(0)} k molecules) of the ${(total / 1e6).toFixed(1)} M-molecule sub-volume.`;
    ($('modeOrg') as HTMLButtonElement).disabled = true; $('modeOrg').title = 'Organelle surfaces are only in the full app';
    ($('modeMem') as HTMLButtonElement).disabled = true; $('modeMem').title = 'Membrane surfaces are only in the full app';
  }
  // views
  const v = $('views');
  bookmarks.forEach((b, i) => {
    const btn = document.createElement('button'); btn.textContent = `${(i + 1) % 10 === 0 && i < 10 ? 0 : i < 9 ? i + 1 : ''}${i < 10 ? ' · ' : ''}${b.name}`;
    btn.title = b.caption; btn.onclick = () => { stopTour(); applyBookmark(i); };
    v.appendChild(btn);
  });
  $('tourBtn').onclick = () => (tour ? stopTour() : startTour());
  document.querySelectorAll('.dynHelp').forEach(e => ((e as HTMLElement).style.display = hasDyn ? '' : 'none'));
  if (hasDyn) {      // tracer dynamics example in this build (dynamics.json present; trajectories are fetched on first use)
    const db = document.createElement('button'); db.className = 'small'; db.textContent = 'Dynamics'; db.title = 'Show / hide the tracer diffusion example (G)';
    db.onclick = () => toggleDynamics(); $('tourBtn').parentElement!.prepend(db, ' ');
  }
  // search
  const inp = $<HTMLInputElement>('search'), res = $('results');
  let act = 0, cur: number[] = [];
  const draw = () => {
    res.innerHTML = cur.map((i, k) => { const t = lib.types[i]; return `<div data-i="${i}" class="${k === act ? 'act' : ''}"><span class="nm"><b>${t.gene}</b> ${t.name}</span><span class="n">${fmt(t.copies)}</span></div>`; }).join('');
    res.querySelectorAll('div').forEach(d => d.addEventListener('click', () => { setType(Number((d as HTMLElement).dataset.i)); cur = []; draw(); inp.value = ''; }));
  };
  inp.addEventListener('input', () => { cur = searchTypes(inp.value); act = 0; draw(); });
  inp.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { act = Math.min(cur.length - 1, act + 1); draw(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { act = Math.max(0, act - 1); draw(); e.preventDefault(); }
    else if (e.key === 'Enter' && cur.length) { setType(cur[act]); flyToNearest(cur[act]); cur = []; draw(); inp.value = ''; inp.blur(); }
    else if (e.key === 'Escape') { cur = []; draw(); inp.blur(); }
    e.stopPropagation();
  });
  // cut
  $<HTMLInputElement>('cutOn').onchange = (e) => { fs.cutOn = (e.target as HTMLInputElement).checked; dirty = true; };
  const ax = (k: number) => () => { const n: V3 = [0, 0, 0]; n[k] = dot(cam.fwd, [k === 0 ? 1 : 0, k === 1 ? 1 : 0, k === 2 ? 1 : 0]) >= 0 ? 1 : -1; fs.cutOn = true; ($('cutOn') as HTMLInputElement).checked = true; setPlane(n, dot(n, cam.target)); };
  $('axX').onclick = ax(0); $('axY').onclick = ax(1); $('axZ').onclick = ax(2);
  $('alignView').onclick = () => { fs.cutOn = true; ($('cutOn') as HTMLInputElement).checked = true; alignToView(); };
  $('followView').onclick = () => { planeFollow = !planeFollow; $('followView').classList.toggle('on', planeFollow); if (planeFollow) alignToView(); };
  $<HTMLInputElement>('planeD').oninput = (e) => { const d = Number((e.target as HTMLInputElement).value); const n = planeN(); const dd = d - fs.plane[3]; cam.target = add(cam.target, scale(n, dd)); setPlane(n, d); };
  const stepSel = $<HTMLSelectElement>('nudgeStep');
  if ([...stepSel.options].some(o => Number(o.value) === nudgeStep)) stepSel.value = String(nudgeStep);
  stepSel.onchange = () => {
    nudgeStep = Number(stepSel.value); syncPlaneUI(); stepSel.blur();
    try { localStorage.setItem('yc.nudgeStep', String(nudgeStep)); } catch { /* storage unavailable */ }
  };
  // -/+ buttons: one step per click, repeating while held
  for (const [id, sgn] of [['nudgeBack', -1], ['nudgeFwd', 1]] as const) {
    const b = $(id); let t0 = 0, t1 = 0;
    const stop = () => { clearTimeout(t0); clearInterval(t1); };
    b.addEventListener('pointerdown', (e) => {
      e.preventDefault(); stop(); sweep = null; $('sweepBtn').classList.remove('on');
      movePlane(sgn * nudgeStep);
      t0 = window.setTimeout(() => { t1 = window.setInterval(() => movePlane(sgn * nudgeStep), 60); }, 400);
    });
    for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) b.addEventListener(ev, stop);
  }
  $<HTMLInputElement>('slab').oninput =(e) => { const x = Number((e.target as HTMLInputElement).value) / 1000; fs.slab = 5 * Math.pow(4400 / 5, x); syncPlaneUI(); dirty = true; };
  $('sweepBtn').onclick = () => {
    if (sweep) { sweep = null; $('sweepBtn').classList.remove('on'); return; }
    const n = planeN(); fs.cutOn = true; ($('cutOn') as HTMLInputElement).checked = true;
    const [d0, d1] = planeRange(n);
    sweep = { t0: performance.now(), d0, d1, dur: Math.max(30000, (d1 - d0) / 4400 * 30000 / 2) };
    cam.target = add(cam.target, scale(n, d0 - fs.plane[3]));
    $('sweepBtn').classList.add('on');
  };
  $('flipBtn').onclick = () => { const n = planeN(); const st = cam.state(); const tgt = cam.target; const nn = scale(n, -1) as V3; setPlane(nn, -fs.plane[3]);
    st.fwd = scale(st.fwd, -1) as V3; st.target = tgt; cam.flyTo(st, 1200); };
  // display
  $('modeSection').onclick = () => setDisplayMode('molecules');
  $('modeOrg').onclick = () => setDisplayMode('organelles');
  $('modeMem').onclick = () => setDisplayMode('membranes');
  $<HTMLSelectElement>('colorMode').onchange = (e) => {
    fs.colorMode = Number((e.target as HTMLSelectElement).value);
    if (fs.colorMode !== 6 && selModule >= 0) setModule(-1);
    renderLegend(); dirty = true;
  };
  $<HTMLInputElement>('shaded').onchange = (e) => { fs.shaded = (e.target as HTMLInputElement).checked; dirty = true; };
  $<HTMLSelectElement>('motion').onchange = (e) => { fs.motion = Number((e.target as HTMLSelectElement).value); (e.target as HTMLElement).blur(); dirty = true; };
  $<HTMLInputElement>('outlines').onchange = (e) => { fs.outline = (e.target as HTMLInputElement).checked ? 1 : 0; dirty = true; };
  renderLegend();
  // show list: named classes have their own switch (and ignore the compartment filter); "only" hides everything else
  const show = $('showList');
  const cls = (k: string) => (on: boolean) => setTypeHidden(CLASS_TYPES[k], !on);
  const has = (k: string) => typesMatching(CLASS_TYPES[k]).length > 0;
  const sumIdx = (a: Uint32Array) => a.reduce((x, y) => x + y, 0);
  const classes: [string, string, (on: boolean) => void][] = ([
    has('actin') && ['Actin', '#E0893A', cls('actin')],
    has('microtubules') && ['Microtubules', '#74C476', cls('microtubules')],
    has('ribosomes') && ['Ribosomes', '#EE9BC0', cls('ribosomes')],
    has('nucleosomes') && ['Chromatin', '#A98BCB', cls('nucleosomes')],
    has('pores') && ['Nuclear pores', PAL[17], cls('pores')],
    sumIdx(idx.nMr) > 0 && ['mRNA', '#B0437A', (on: boolean) => setFree(0, on)],
    sumIdx(idx.nLk) > 0 && ['Linker DNA', '#E07636', (on: boolean) => setFree(1, on)],
    ['Lipids', '#F3C643', (on: boolean) => { fs.showLipids = on; dirty = true; }],
    ['Membranes', '#F8E3A0', (on: boolean) => { fs.showMeshes = on; dirty = true; }],
    ['Soluble proteins', PAL[0], (on: boolean) => setRegions(SOLUBLE_REGIONS, on)],
    ['Membrane proteins', PAL[7], (on: boolean) => setRegions(MEM_REGIONS, on)],
  ] as any[]).filter(Boolean);
  const boxes: HTMLInputElement[] = [];
  const allRegions = REGION_NAMES.map((_, i) => i);
  // only membranes shown: colour them by the compartment they enclose (the section map's colours)
  const mem = classes.findIndex(([name]) => name === 'Membranes');
  // (the whole cell's membrane surfaces, as in organelle mode, without the cut: see renderer.frame)
  const syncMemColors = () => {
    fs.memColors = boxes.every((b, k) => b.checked === (k === mem)) && !(window as any).__YC_PREVIEW__;   // preview: no surfaces
    if (fs.memColors) loadOrganelleMeshes();
    syncModeButtons();
    dirty = true;
  };
  const setAll = (on: (k: number) => boolean) => {
    classes.forEach(([, , f], k) => { boxes[k].checked = on(k); f(on(k)); });
    const regionsOn = classes.every((_, k) => on(k));       // "only": every compartment off (e.g. cell-wall proteins too)
    setRegions(allRegions, regionsOn);
    syncMemColors();
  };
  classes.forEach(([name, col, f], k) => {
    const l = document.createElement('label'); l.className = 'chk';
    l.innerHTML = `<input type="checkbox" checked><span class="sw" style="background:${col}"></span><span class="nm">${name}</span><a class="only" title="Show only ${name.toLowerCase()}">only</a>`;
    const box = l.querySelector('input') as HTMLInputElement; boxes.push(box);
    box.onchange = () => { f(box.checked); syncMemColors(); };
    (l.querySelector('.only') as HTMLElement).onclick = (e) => { e.preventDefault(); e.stopPropagation(); setAll(j => j === k); };
    show.appendChild(l);
  });
  setMembraneView = (on) => setAll(on ? (j => j === mem) : (() => true));
  $('showAll').onclick = () => setAll(() => true);
  $('hideAll').onclick = () => setAll(() => false);
  const rl = $('regionList');
  REGION_NAMES.forEach((n, i) => {
    const l = document.createElement('label'); l.className = 'chk';
    l.innerHTML = `<input type="checkbox" checked data-r="${i}"><span class="sw" style="background:${PAL[i]}"></span>${n}`;
    (l.querySelector('input') as HTMLInputElement).onchange = (e) => setRegions([i], (e.target as HTMLInputElement).checked);
    rl.appendChild(l);
  });
  $<HTMLSelectElement>('quality').onchange = (e) => setQuality((e.target as HTMLSelectElement).value);
  const dsel = $<HTMLSelectElement>('detailBudget');
  dsel.value = R.detail; dsel.title = `Molecules kept on the GPU at once (now ${R.detailUsed}). Changing it reloads the page.`;
  dsel.onchange = () => {
    try { localStorage.setItem('yc.detail', dsel.value); } catch { /* storage unavailable */ }
    const u = new URL(location.href); u.searchParams.delete('detail'); location.href = u.href;
  };
  $<HTMLInputElement>('detailOn').onchange = (e) => {
    T.detailEnabled = (e.target as HTMLInputElement).checked;
    if (!T.detailEnabled) for (const t of T.loaded.values()) if (t.det) T.dropDetail(t);
    dirty = true;
  };
  $('shotBtn').onclick = screenshot;
  $('helpBtn').onclick = () => ($('helpModal').style.display = 'flex');
  $('aboutBtn').onclick = () => ($('aboutModal').style.display = 'flex');
  $('aboutBody').innerHTML = geo?.whole_cell ? ABOUT_HTML_WHOLE : ABOUT_HTML;
  document.querySelectorAll('.cubeSize').forEach(e => { if (geo?.cube_nm && !geo.whole_cell) e.textContent = fmtNm(geo.cube_nm); });
  if (geo?.whole_cell) {
    $('sweepBtn').textContent = 'Sweep through cell'; document.title = 'Jurkat Cell Model: whole cell';
    document.querySelectorAll('.cubeSize').forEach(e => (e.textContent = 'whole cell'));
    document.querySelectorAll('.subvolWord').forEach(e => (e.textContent = ''));
    const B = modelBox(), L = Math.max(B.hi[0] - B.lo[0], B.hi[1] - B.lo[1], B.hi[2] - B.lo[2]) / 1000;
    $('zoomHelp').textContent = `Zoom (from the whole cell, about ${L.toFixed(0)} µm across, down to single residues; molecules stream in around the view centre)`;
  }
  $('modelBtn').onclick = () => {
    // the dynamics section describes the offline tracer example: only for builds that ship it
    if (!$('modelBody').innerHTML) $('modelBody').innerHTML = markdownToHtml(hasDyn ? MODEL_MD : MODEL_MD.replace(/\n## Dynamics example[\s\S]*?(?=\n## |$)/, '\n'));
    $('modelModal').style.display = 'flex';
  };
  $('aboutModal').addEventListener('click', (e) => {   // "Model details" link inside the About text
    if ((e.target as HTMLElement).id === 'aboutToModel') { e.preventDefault(); $('aboutModal').style.display = 'none'; $('modelBtn').click(); }
  });
  document.querySelectorAll('.modal').forEach(m => m.addEventListener('click', (e) => {
    if (e.target === m || (e.target as HTMLElement).classList.contains('close')) (m as HTMLElement).style.display = 'none';
  }));
  $('molClose').onclick = () => ($('molModal').style.display = 'none');
  $('panelToggle').onclick = () => { $('panel').classList.toggle('collapsed'); $('panelToggle').textContent = $('panel').classList.contains('collapsed') ? '›' : '‹'; };
  syncPlaneUI();
  setupInput();
  setupMinimap();
}
function toggleDynamics() {
  if (dyn.visible) { dyn.hide(); return; }
  const i = bookmarks.findIndex(b => b.dynamics && (!dyn.data || b.dynamics === dyn.cur.region));
  if (i >= 0) { stopTour(); applyBookmark(i); } else dyn.show();
}
function setTypeHidden(ids: string[], hidden: boolean) {
  for (const i of typesMatching(ids)) { if (hidden) typeFlags[i] |= 2; else typeFlags[i] &= ~2; }
  R.device.queue.writeBuffer(R.typeFlags, 0, typeFlags); dirty = true;
}
function setFree(kind: number, on: boolean) { if (on) fs.freeHide &= ~(1 << kind); else fs.freeHide |= 1 << kind; dirty = true; }
function setRegions(rs: number[], on: boolean) {
  for (const r of rs) { if (on) fs.regionMask |= 1 << r; else fs.regionMask &= ~(1 << r); }
  fs.regionMask >>>= 0;
  document.querySelectorAll<HTMLInputElement>('#regionList input').forEach(inp => { inp.checked = ((fs.regionMask >>> Number(inp.dataset.r)) & 1) === 1; });
  dirty = true;
}

// ---------------------------------------------------------------- input
function setupInput() {
  let drag: null | { x: number; y: number; btn: number; moved: number; shift: boolean } = null;
  const keys = new Set<string>();
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId); canvas.focus();
    drag = { x: e.clientX, y: e.clientY, btn: e.button, moved: 0, shift: e.shiftKey };
    canvas.classList.add('dragging');
    if (tour) stopTour();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) { hover(e.clientX, e.clientY); return; }
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY; drag.moved += Math.abs(dx) + Math.abs(dy);
    if (drag.btn === 2 || drag.btn === 1 || drag.shift) {
      const pxPerNm = cam.pxPerNmAtUnit(canvas.clientHeight) / cam.dist;
      cam.pan(dx, dy, pxPerNm);
    } else {
      cam.orbit(dx * 0.005, dy * 0.005);
      if (planeFollow) alignToView();
    }
    cam.anim = null; dirty = true;
  });
  const end = (e: PointerEvent) => {
    if (!drag) return;
    const wasClick = drag.moved < 4 && drag.btn === 0;
    drag = null; canvas.classList.remove('dragging');
    if (wasClick) {
      pickAt(e.offsetX, e.offsetY).then(r => { const d = describe(r.id); select(d.kind === 'inst' || d.kind === 'lipid' || d.kind === 'free' || d.kind === 'band' ? d : null); });
    }
  };
  canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('dblclick', (e) => {
    pickAt(e.offsetX, e.offsetY).then(r => {
      if (r.depth <= 0) return;
      const near = cam.near(), zv = near / r.depth;   // view-space distance along -z
      const aspect = canvas.clientWidth / canvas.clientHeight;
      const ray = cam.ray(e.offsetX / canvas.clientWidth, e.offsetY / canvas.clientHeight, aspect);
      const t = zv / dot(ray, cam.fwd);
      const p = add(cam.eye, scale(ray, t));
      const st = cam.state(); st.target = p; st.dist = Math.max(cam.minDist * 4, cam.dist * 0.55);
      cam.flyTo(st, 900);
    });
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    // macOS turns Shift+wheel on a mouse into horizontal scrolling: use whichever axis moved
    const raw = e.deltaY || e.deltaX;
    const dy = e.deltaMode === 1 ? raw * 30 : raw;
    if (e.altKey) {
      // fine cut-plane step: one fine step per mouse-wheel notch (~100 px), proportional for trackpads, at most 4 per event
      movePlane(Math.sign(dy) * nudgeStep * Math.min(4, Math.abs(dy) / 100));
    } else if (e.shiftKey) {
      const n = planeN(), step = Math.max(1, Math.min(fs.slab, cam.dist * 0.05)) * Math.sign(dy) * Math.min(4, Math.abs(dy) / 60 + 0.5);
      cam.target = add(cam.target, scale(n, step)); setPlane(n, fs.plane[3] + step);
    } else {
      // zoom towards the cursor (Cmd: 8x slower; Ctrl is left alone because trackpad pinch arrives as Ctrl+wheel)
      const f = Math.exp(dy * 0.0015 * (e.metaKey ? 0.125 : 1));
      const aspect = canvas.clientWidth / canvas.clientHeight;
      const ray = cam.ray(e.offsetX / canvas.clientWidth, e.offsetY / canvas.clientHeight, aspect);
      const tgtDepth = cam.dist;
      const pc = add(cam.eye, scale(ray, tgtDepth / dot(ray, cam.fwd)));
      const old = cam.dist; cam.zoom(f);
      const k = 1 - cam.dist / old;
      cam.target = add(cam.target, scale(sub(pc, cam.target), k));
    }
    cam.anim = null; if (tour) stopTour(); dirty = true;
  }, { passive: false });
  window.addEventListener('keydown', (e) => {
    if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'SELECT') return;
    const k = e.key.toLowerCase();
    keys.add(k);
    if (k === 'c') { fs.cutOn = !fs.cutOn; ($('cutOn') as HTMLInputElement).checked = fs.cutOn; dirty = true; }
    else if (k === 'v') { fs.cutOn = true; ($('cutOn') as HTMLInputElement).checked = true; alignToView(); }
    else if (k === '[') { fs.slab = Math.max(5, fs.slab / 1.25); syncPlaneUI(); dirty = true; }
    else if (k === ']') { fs.slab = Math.min(4400, fs.slab * 1.25); syncPlaneUI(); dirty = true; }
    else if (k === ',' || k === '.' || e.code === 'Comma' || e.code === 'Period') {   // Alt: one fine step (Option changes e.key on macOS)
      const fwd = k === '.' || e.code === 'Period';
      movePlane((fwd ? 1 : -1) * (e.altKey ? nudgeStep : Math.max(2, fs.slab * 0.25))); }
    else if (k === 'o') setDisplayMode(fs.organelle ? 'molecules' : 'organelles');
    else if (k === 'm') setDisplayMode(fs.memColors && !fs.organelle ? 'molecules' : 'membranes');
    else if (k === 't') (tour ? stopTour() : startTour());
    else if (k === 'p') screenshot();
    else if (k === 'g' && hasDyn) toggleDynamics();
    else if (k === 'k' && hasDyn && dyn.visible) dyn.setPlaying(!dyn.playing);
    else if (k === 'escape') { select(null); setType(-1); }
    else if (/^[0-9]$/.test(k)) { const i = k === '0' ? 9 : Number(k) - 1; if (i < bookmarks.length) { stopTour(); applyBookmark(i); } }
  });
  window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
  window.addEventListener('blur', () => keys.clear());
  (setupInput as any).keys = keys;
  window.addEventListener('resize', () => { R.resize(fs.renderScale); dirty = true; });
}
let hoverT = 0, hoverBusy = false;
function hover(x: number, y: number) {
  const now = performance.now();
  if (hoverBusy || now - hoverT < 90) return;
  hoverT = now; hoverBusy = true;
  dirty = true;
  pickAt(x, y).then(r => {
    hoverBusy = false;
    const d = describe(r.id), el = $('hover');
    if (!d.title || d.kind === 'none') { el.style.display = 'none'; return; }
    let txt = `<b>${d.title}</b>`;
    if (d.kind === 'inst') { const t = lib.types[d.info.type]; txt += ` · ${t.name} <span style="color:#aaa">· ${d.sub}</span>`; }
    else txt += ` <span style="color:#aaa">· ${d.sub}</span>`;
    el.innerHTML = txt; el.style.display = 'block';
    el.style.left = `${x + 14}px`; el.style.top = `${y + 12}px`;
  });
}
canvas.addEventListener('pointerleave', () => ($('hover').style.display = 'none'));

// ---------------------------------------------------------------- minimap (analytic section through the cut plane)
let mmDirty = true, mmLastT = 0;
let mmBasis: { u: V3; v: V3; c: V3; ext: number } | null = null;
const mmImg: { data: ImageData | null } = { data: null };
function setupMinimap() {
  const c = $<HTMLCanvasElement>('minimap');
  c.addEventListener('click', (e) => {
    if (!mmBasis) return;
    const r = c.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width * 2 - 1) * mmBasis.ext, y = (1 - (e.clientY - r.top) / r.height * 2) * mmBasis.ext;
    const p = add(mmBasis.c, add(scale(mmBasis.u, x), scale(mmBasis.v, y)));
    const st = cam.state(); st.target = p; cam.flyTo(st, 1200);
    if (tour) stopTour();
  });
}
function updateMinimap(now: number) {
  const c = $<HTMLCanvasElement>('minimap'), g = c.getContext('2d')!;
  const W = c.width, H = c.height;
  const n: V3 = fs.cutOn ? planeN() : cam.fwd;
  const d = fs.cutOn ? fs.plane[3] : dot(cam.fwd, cam.target);
  if (mmDirty && now - mmLastT > 250) {
    mmDirty = false; mmLastT = now;
    let u = sub(cam.right, scale(n, dot(cam.right, n)));
    if (len(u) < 1e-3) u = norm(cross(n, [0, 0, 1])); u = norm(u);
    let v = cross(n, u); if (dot(v, cam.up) < 0) v = scale(v, -1) as V3;
    // Jurkat: the sub-volume (fields exist only there), or the whole cell's bounding box
    const MB = modelBox(), ext = geo.whole_cell ? 1.05 * Math.max(...MB.lo.map(Math.abs), ...MB.hi.map(Math.abs)) : geo.half ? geo.half * 1.25 : 2300, c0 = scale(n, d) as V3;
    mmBasis = { u, v, c: c0, ext };
    const N = 120, img = g.createImageData(N, N);
    // per label (LABEL_NAMES): exterior, cytosol, perinuclear, nucleoplasm, nucleolus, IMS, matrix, ER, Golgi, endo, pore
    const cols = [PAL[80], '#C4D6EB', PAL[6], PAL[1], '#A27DB8', '#E8B48F', PAL[2], PAL[3], PAL[4], PAL[5], '#C9C2DC']
      .map(h => { const x = parseInt(h.slice(1), 16); return [(x >> 16) & 255, (x >> 8) & 255, x & 255]; });
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const x = ((i + 0.5) / N * 2 - 1) * ext, y = (1 - (j + 0.5) / N * 2) * ext;
      const p = add(c0, add(scale(u, x), scale(v, y)));
      const lab = geom.label(p[0], p[1], p[2]);
      const inCube = !geo.half || (Math.abs(p[0]) <= geo.half && Math.abs(p[1]) <= geo.half && Math.abs(p[2]) <= geo.half);
      const col = cols[lab] ?? cols[0], o = 4 * (j * N + i);
      img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = lab === 0 ? 0 : inCube ? 255 : 110;
    }
    mmImg.data = img;
  }
  if (!mmImg.data || !mmBasis) return;
  const tmp = (updateMinimap as any).tmp || ((updateMinimap as any).tmp = document.createElement('canvas'));
  tmp.width = mmImg.data.width; tmp.height = mmImg.data.height; tmp.getContext('2d')!.putImageData(mmImg.data, 0, 0);
  g.clearRect(0, 0, W, H); g.imageSmoothingEnabled = true; g.drawImage(tmp, 0, 0, W, H);
  // view footprint at the target
  const B = mmBasis, toPx = (p: V3) => { const q = sub(p, B.c); return [(dot(q, B.u) / B.ext + 1) / 2 * W, (1 - dot(q, B.v) / B.ext) / 2 * H]; };
  const aspect = canvas.clientWidth / canvas.clientHeight;
  const hh = cam.dist * Math.tan(cam.fovy / 2), hw = hh * aspect;
  const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => toPx(add(cam.target, add(scale(cam.right, a * hw), scale(cam.up, b * hh)))));
  g.strokeStyle = '#FF9400'; g.lineWidth = 3; g.beginPath();
  corners.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.stroke();
  const [tx, ty] = toPx(cam.target);
  g.fillStyle = '#FF9400'; g.beginPath(); g.arc(tx, ty, 5, 0, 7); g.fill();
  $('mmLabel').textContent = fs.cutOn ? `plane ${Math.round(d)} nm` : 'view plane';
}

// ---------------------------------------------------------------- screenshot
function screenshot() {
  const sc = fs.renderScale;
  R.resize(2 * sc);
  fs.renderScale = 2 * sc;
  renderNow(true);
  canvas.toBlob((b) => {
    if (b) {
      const a = document.createElement('a'); a.href = URL.createObjectURL(b);
      a.download = `jurkat_subvolume_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.png`; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    }
    fs.renderScale = sc; R.resize(sc); dirty = true;
  }, 'image/png');
  toast('Screenshot saved (2× resolution)');
}

// ---------------------------------------------------------------- HUD / scale bar
function updateHUD() {
  const tt = T.totals(), s = R.stats;
  const fps = frameTimes.length > 3 ? 1000 / (frameTimes.reduce((a, b) => a + b) / frameTimes.length) : 0;
  const pend = T.pendingForView();
  $('hud').innerHTML = `<b>${fmt(tt.inst)}</b> molecules in ${tt.tiles} tiles${pend ? ` · <span style="color:#FF9400">streaming ${pend}</span>` : ''}${T.starved ? ` · <span style="color:#FF9400">budget of ${fmt(tt.instCap)} molecules reached (thinner slab or higher Detail loads more)</span>` : ''}${T.whole && fs.cutOn && tt.secWant ? ` · section ${Math.round(100 * tt.secHave / tt.secWant)}% of ${tt.secWant} tiles` : ''}${tt.ovSpheres ? ` · ${fmt(tt.ovSpheres)} stand-ins` : ''}<br>
    drawn: <b>${fmt(s.ninst)}</b> in view · ${fmt(s.l0 * 64)} residue beads (≤) · ${fmt(s.l1 * 16)} LOD1 · ${fmt(s.l2)} LOD2 · ${fmt(s.lip)} lipids · ${fmt(s.fre)} polymer beads<br>
    ${fps ? `${fps.toFixed(0)} fps · ` : ''}LOD bead ≥ ${fs.l0px.toFixed(1)} px · GPU pools ${R.gpuMemoryMB().toFixed(0)} MB · ${(T.bytesLoaded / 1048576).toFixed(0)} MB streamed${R.adapterInfo ? ' · ' + R.adapterInfo : ''}`;
  // scale bar
  const dpr = R.width / canvas.clientWidth;
  const pxPerNm = cam.pxPerNmAtUnit(R.height) / cam.dist / dpr;
  const target = 120 / pxPerNm;
  const p10 = Math.pow(10, Math.floor(Math.log10(target)));
  const nice = [1, 2, 5, 10].map(m => m * p10).reduce((a, b) => Math.abs(b - target) < Math.abs(a - target) ? b : a);
  ($('scalebar').querySelector('.bar') as HTMLElement).style.width = `${nice * pxPerNm}px`;
  ($('scalebar').querySelector('span') as HTMLElement).textContent = fmtNm(nice);
}

// ---------------------------------------------------------------- main loop
let gpuMs = 0, gpuBusy = false;
function renderNow(force = false) {
  const t0 = performance.now();
  R.frame(cam, fs, geo.G, T.meshDraws(), force || t0 - R.lastStatsT > 400);
  R.lastStatsT = performance.now();
  gpuBusy = true;
  R.device.queue.onSubmittedWorkDone().then(() => { gpuMs = performance.now() - t0; gpuBusy = false; });
}
let hudT = 0;
function tileTick() {
  T.update(cam, canvas.clientWidth / canvas.clientHeight, fs.cutOn, fs.plane, fs.slab, fs.organelle || fs.memColors, R.height);
  lastUpdate = performance.now();
  if (T.inflight || T.detInflight) dirty = true;
}
// streaming must not be throttled by requestAnimationFrame when frames are slow
setInterval(() => { if (T && performance.now() - lastUpdate > 150) tileTick(); }, 150);
function loop(now: number) {
  requestAnimationFrame(loop);
  const dt = lastFrameT ? now - lastFrameT : 16; lastFrameT = now;
  // keyboard flight
  const keys: Set<string> = (setupInput as any).keys;
  if (keys && keys.size) {
    const sp = cam.dist * 0.9 * dt / 1000;
    let mv: V3 = [0, 0, 0];
    if (keys.has('w')) mv = add(mv, scale(cam.fwd, sp)); if (keys.has('s')) mv = add(mv, scale(cam.fwd, -sp));
    if (keys.has('a')) mv = add(mv, scale(cam.right, -sp)); if (keys.has('d')) mv = add(mv, scale(cam.right, sp));
    if (keys.has('q')) mv = add(mv, scale(cam.up, -sp)); if (keys.has('e')) mv = add(mv, scale(cam.up, sp));
    if (keys.has('z')) cam.roll(-dt * 0.0015); if (keys.has('x')) cam.roll(dt * 0.0015);
    if (len(mv) > 0 || keys.has('z') || keys.has('x')) { cam.fly(mv); dirty = true; if (planeFollow) alignToView(); }
  }
  if (cam.step(now)) { dirty = true; if (planeFollow && !tour) alignToView(); }
  if (sweep) {
    const e = Math.min(1, (now - sweep.t0) / sweep.dur);
    const n = planeN(), d = sweep.d0 + (sweep.d1 - sweep.d0) * e;
    cam.target = add(cam.target, scale(n, d - fs.plane[3])); fs.plane[3] = d; syncPlaneUI();
    if (e >= 1) { sweep = null; $('sweepBtn').classList.remove('on'); }
    dirty = true;
  }
  tourStep(now);
  if (fs.motion > 0) dirty = true;          // illustrative motion: redraw every frame
  if (dyn.tick(dt)) dirty = true;           // tracer dynamics playing
  if (now - lastUpdate > 90) tileTick();
  if (dirty && !gpuBusy) {
    dirty = false;
    renderNow();
    frameTimes.push(dt); if (frameTimes.length > 30) frameTimes.shift();
    // adaptive quality while interacting
    if (quality === 'auto' && frameTimes.length >= 20) {
      const avg = frameTimes.reduce((a, b) => a + b) / frameTimes.length;
      if (avg > 45 && fs.l0px < 3.5) { fs.l0px = Math.min(3.5, fs.l0px * 1.15); frameTimes.length = 0; }
      else if (avg < 22 && fs.l0px > 0.9) { fs.l0px = Math.max(0.9, fs.l0px / 1.1); frameTimes.length = 0; }
    }
    setTimeout(() => { dirty = dirty || R.statsBusy; }, 0);
  } else if (!dirty) frameTimes.length = 0;
  if (now - hudT > 300) { hudT = now; updateHUD(); updateMinimap(now); if (selType >= 0 && T.inflight) renderSelType(); }
}

boot();
