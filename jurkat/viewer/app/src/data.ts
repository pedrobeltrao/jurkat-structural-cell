// Data access (fetch, or an embedded map for the single-file preview) and library preparation.
import { packColor } from './math';

declare global { interface Window { __YC_EMBED__?: Record<string, string>; } }

// Data location, configurable without rebuilding: ?data=<url> overrides config.json {"data_base": "<url>"} next to
// index.html, which overrides the default 'data/'. Relative URLs resolve against the page. The host must allow CORS
// (and, for whole-cell builds, HTTP Range requests; e.g. https://huggingface.co/datasets/<owner>/<name>/resolve/main/).
async function resolveDataBase(): Promise<string> {
  const withSlash = (u: string) => (u.endsWith('/') ? u : u + '/');
  try {
    const q = new URLSearchParams(location.search).get('data');
    if (q) return withSlash(new URL(q, document.baseURI).href);
  } catch { /* malformed parameter: ignore */ }
  if (!window.__YC_EMBED__) {
    try {
      const r = await fetch(new URL('config.json', document.baseURI).href, { cache: 'no-store' });
      if (r.ok) {
        const c = await r.json();
        if (c && typeof c.data_base === 'string' && c.data_base) return withSlash(new URL(c.data_base, document.baseURI).href);
      }
    } catch { /* no config.json: default */ }
  }
  return new URL('data/', document.baseURI).href;
}
export const DATA_BASE: string = await resolveDataBase();
// data/version.json changes on every deploy; it is read uncached and appended to every data URL so a browser never mixes
// cached files from two builds (type indices shift between builds)
const VERSION: Promise<any> = fetch(DATA_BASE + 'version.json', { cache: 'no-store' })
  .then(r => (r.ok ? r.json() : {})).catch(() => ({}));
export const DATA_VER: Promise<string> = VERSION.then(j => String(j?.v ?? ''));
/** optional files the build declares absent (version.json "absent", jenc_full.py): not requested, so no 404s */
export async function isAbsent(path: string): Promise<boolean> {
  const a = (await VERSION)?.absent;
  return Array.isArray(a) && a.includes(path);
}
export const verQuery = (v: string) => (v ? `?v=${encodeURIComponent(v)}` : '');

export function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64); const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export async function gunzipIfNeeded(buf: ArrayBuffer): Promise<ArrayBuffer> {
  const u8 = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  if (u8[0] === 0x1f && u8[1] === 0x8b) {
    const ds = new DecompressionStream('gzip');
    const stream = new Blob([buf]).stream().pipeThrough(ds);
    return await new Response(stream).arrayBuffer();
  }
  return buf;
}

// optional data/split.json {path: n_parts}: hosts with a per-file size limit get large files as path.000, path.001, ...
// {"_b64": 1}: every non-JSON file is stored as base64 text at path.txt (hosts that serve no binary types)
const SPLIT: Promise<Record<string, number>> = fetch(DATA_BASE + 'split.json', { cache: 'no-store' })
  .then(r => (r.ok ? r.json() : {})).catch(() => ({}));

/** fetch with retries on 429 / 503 (rate-limited hosts such as Hugging Face): Retry-After or 1, 2, 4, 8 s */
async function fetchRetry(url: string): Promise<Response> {
  for (let k = 0; ; k++) {
    const r = await fetch(url);
    if ((r.status !== 429 && r.status !== 503) || k >= 4) return r;
    const ra = Number(r.headers.get('retry-after'));
    await new Promise(res => setTimeout(res, ra > 0 ? Math.min(ra, 30) * 1000 : 1000 * 2 ** k));
  }
}
export async function fetchBin(path: string, onProgress?: (n: number, total: number) => void): Promise<ArrayBuffer> {
  const emb = window.__YC_EMBED__;
  if (emb && emb[path] !== undefined) return gunzipIfNeeded(b64ToBytes(emb[path]).buffer as ArrayBuffer);
  const cfg = await SPLIT; const np = cfg[path];
  const b64 = !!cfg._b64 && !path.endsWith('.json');
  const raw = async (p: string) => {
    const r = await fetchRetry(DATA_BASE + p + (b64 ? '.txt' : '') + verQuery(await DATA_VER));
    if (!r.ok) throw new Error(`HTTP ${r.status} for ${p}`);
    return b64 ? b64ToBytes(await r.text()).buffer as ArrayBuffer : await r.arrayBuffer();
  };
  if (b64 && !np) { const b = await raw(path); onProgress?.(1, 1); return gunzipIfNeeded(b); }
  if (np) {
    const parts: ArrayBuffer[] = []; let n = 0;
    for (let i = 0; i < np; i++) {
      const b = await raw(`${path}.${String(i).padStart(3, '0')}`); parts.push(b); n += b.byteLength; onProgress?.(i + 1, np);
    }
    const out = new Uint8Array(n); let o = 0;
    for (const b of parts) { out.set(new Uint8Array(b), o); o += b.byteLength; }
    return gunzipIfNeeded(out.buffer);
  }
  const r = await fetchRetry(DATA_BASE + path + verQuery(await DATA_VER));
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${path}`);
  if (!onProgress || !r.body) return gunzipIfNeeded(await r.arrayBuffer());
  const total = Number(r.headers.get('content-length') || 0);
  const reader = r.body.getReader(); const chunks: Uint8Array[] = []; let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); n += value.length; onProgress(n, total);
  }
  const out = new Uint8Array(n); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return gunzipIfNeeded(out.buffer);
}
export async function fetchJson(path: string): Promise<any> {
  const b = await fetchBin(path);
  return JSON.parse(new TextDecoder().decode(b));
}

export interface TypeRec {
  id: string; gene: string; genes: string; uniprot: string; name: string; src: string; pdb: string;
  oligomer: number; nb: number; nnt: number; bo: number; so: number; sn: number; br: number; rg: number;
  mass: number; plddt_mean: number | null; plddt_frac_kept: number | null; copies: number; regions: Record<string, number>;
  model?: string;   // AlphaFold DB entry of a predicted complex (src 'AF-multimer')
}

// Jurkat (jurkat/viewer/jenc.py REGION_NAMES): instance region codes, membrane (face_mem) order and compartment labels.
// Ids 0-17 are the v1-v4 regions (unchanged, so v4 data decode as before); model v5 adds 18 nucleolus and 19-30 the
// membrane-associated shells (lipid-anchored proteins on either side of their membrane, the cortex under the plasma
// membrane, the nuclear lamina); model v6 adds 31 nuclear speckle (the last id of the 5-bit region field of the instance
// and stand-in words) and maps its NPC-proximal shell (pack region sh_NPC) to 17 (nuclear pore).
export const REGION_NAMES = ['cytosol', 'nucleoplasm', 'mitochondrial matrix', 'ER lumen', 'Golgi lumen', 'endo-lysosomal lumen',
  'perinuclear space', 'plasma membrane', 'nuclear envelope', 'ER membrane', 'mito outer membrane', 'mito inner membrane',
  'Golgi membrane', 'endo-lysosomal membrane', 'chromatin', 'polysome ribosome', 'cytoskeleton', 'nuclear pore (NPC and pore-proximal)',
  'nucleolus', 'PM, cytosolic side (lipid-anchored)', 'PM, outer side (lipid/GPI-anchored)', 'cortex under the PM',
  'nuclear lamina', 'inner nuclear membrane, nucleoplasmic side (anchored)', 'ER, cytosolic side (lipid-anchored)',
  'ER, lumenal side (anchored)', 'Golgi, cytosolic side (lipid-anchored)', 'Golgi, lumenal side (anchored)',
  'endo-lysosome, cytosolic side (lipid-anchored)', 'endo-lysosome, lumenal side (anchored)',
  'mito outer membrane, cytosolic side (anchored)', 'nuclear speckle'];
export const MEM_NAMES = ['plasma membrane', 'nuclear envelope', 'ER membrane', 'mito outer membrane', 'mito inner membrane',
  'Golgi membrane', 'endo-lysosomal membrane'];
export const LABEL_NAMES = ['exterior', 'cytosol', 'perinuclear space', 'nucleoplasm', 'nucleolus', 'mito intermembrane space',
  'mito matrix', 'ER lumen', 'Golgi lumen', 'endo-lysosomal lumen', 'nuclear pore channel',
  'edge of the EM data (the imaged volume ends here; the cell continues)', 'nuclear speckle',
  'Golgi intercisternal cytosol'];
export const L_OUTV = 11, L_SPK = 12, L_GIC = 13;        // L_GIC: model v7 cisternal Golgi (cell3d GIC), filled like cytosol
// procedural lipid classes (worker.ts LCUM; index 5 is the sterol slot of the shader, unused: cholesterol is not drawn as a headgroup)
export const LIPID_CLASSES = ['PI', 'PE', 'PC', 'SM', 'PS', '(sterol, unused)', 'other lipids'];
// region indices of membrane-embedded instances (shaders.ts animInst uses the same range)
export const MEM_REGIONS = [7, 8, 9, 10, 11, 12, 13];
export const SOLUBLE_REGIONS = [0, 1, 2, 3, 4, 5, 6, 18, 31];
// v5 membrane-associated shells (not in the bilayer: animated like soluble proteins)
export const SHELL_REGIONS = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30];

// Goodsell-inspired palette (shared by shaders through a storage buffer)
// 0..31 instance regions, 32..42 lipids, 48..53 free-sphere kinds, 56..62 organelle-mode membranes (MEM order),
// 66..72 membranes-only view (colour of the enclosed compartment), 80..90 compartment-label fill, 91 traced cell surface,
// 92 the open face (edge of the EM data), 93 nuclear-speckle fill (label 12: 80 + 12 would be 92)
export const PAL: Record<number, string> = {
  0: '#9DB9DA', 1: '#B993C9', 2: '#E39A5B', 3: '#8DBF7A', 4: '#E6C653', 5: '#A57F5A', 6: '#C9A6D6',
  7: '#5B8FC7', 8: '#8E6BB0', 9: '#5FAE75', 10: '#C27C55', 11: '#B5653F', 12: '#C9A93B', 13: '#8A6440',
  14: '#9A72B8', 15: '#EE9BC0', 16: '#74C476', 17: '#7F98B5',
  18: '#8A4F9E', 19: '#7FA8DA', 20: '#3F6FA8', 21: '#D9A26B', 22: '#6A4C9C', 23: '#A88BC6', 24: '#82C493', 25: '#5E9A55',
  26: '#DDC466', 27: '#B0922C', 28: '#B8936B', 29: '#76543A', 30: '#D58C64', 31: '#C2408A',
  32: '#F3C643', 33: '#F0BC3C', 34: '#F6D15E', 35: '#EAB542', 36: '#EFA83A', 37: '#E7A93C', 38: '#F2CA55',
  40: '#F3C643', 41: '#F8E3A0', 42: '#E7A93C',
  48: '#B0437A', 49: '#E07636', 50: '#CDA866', 51: '#C3C27A', 52: '#C9558F', 53: '#7CC6C2',
  56: '#7EA7D3', 57: '#B993C9', 58: '#5FAE75', 59: '#D9946B', 60: '#B5653F', 61: '#E6C653', 62: '#A57F5A',
  66: '#A9C3E3', 67: '#B993C9', 68: '#A3C48C', 69: '#E8B48F', 70: '#D9946B', 71: '#E9D27A', 72: '#B89676',
  80: '#F6F1E7', 81: '#E4ECF5', 82: '#EBD9EA', 83: '#F1E7F1', 84: '#D6B9DE', 85: '#F6E6DA', 86: '#F2DCCB',
  87: '#E8F0DF', 88: '#F7EDC4', 89: '#EEE3D6', 90: '#ECE8F2', 91: '#D3DFEE', 92: '#8D939C', 93: '#F2C9DF',
};
export function paletteArray(): Float32Array<ArrayBuffer> {
  const a = new Float32Array(96 * 4);
  for (const [k, h] of Object.entries(PAL)) {
    const v = parseInt(h.slice(1), 16), i = Number(k) * 4;
    a[i] = ((v >> 16) & 255) / 255; a[i + 1] = ((v >> 8) & 255) / 255; a[i + 2] = (v & 255) / 255; a[i + 3] = 1;
  }
  return a;
}

// special colours by library type id (types.json 'id'); a trailing '*' matches a prefix. Resolved per type at load.
export const SPECIAL: Record<string, { c0: string; c1?: string; cls: string }> = {
  'RIBO*': { c0: '#EE9BC0', c1: '#C9558F', cls: 'ribosome' },
  'NUCLEOSOME': { c0: '#A98BCB', c1: '#7A4FA0', cls: 'nucleosome' },
  'H1-*': { c0: '#8E6CB0', c1: '#8E6CB0', cls: 'nucleosome' },
  'NPC': { c0: '#7F98B5', cls: 'NPC' },
  'MT_*': { c0: '#74C476', c1: '#74C476', cls: 'microtubule' },
  'F_*': { c0: '#E0893A', c1: '#E0893A', cls: 'actin' },
};
/** the SPECIAL entry of a type id (exact key first, then the longest matching prefix pattern) */
export function specialFor(id: string): { c0: string; c1?: string; cls: string } | undefined {
  if (SPECIAL[id] && !id.endsWith('*')) return SPECIAL[id];
  let best: string | null = null;
  for (const k of Object.keys(SPECIAL)) if (k.endsWith('*') && id.startsWith(k.slice(0, -1)) && (!best || k.length > best.length)) best = k;
  return best ? SPECIAL[best] : undefined;
}
/** true if a type id matches one of the patterns (exact ids, or prefixes ending in '*') */
export function idMatches(id: string, patterns: string[]): boolean {
  return patterns.some(p => (p.endsWith('*') ? id.startsWith(p.slice(0, -1)) : id === p));
}

/** colour of interaction module i: golden-angle hues, three lightness levels */
export function moduleColor(i: number): string {
  const h = (i * 137.508) % 360, s = 0.62, l = [0.58, 0.44, 0.70][i % 3];
  const f = (n: number) => { const k = (n + h / 30) % 12; return l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
  return '#' + [f(0), f(8), f(4)].map(v => Math.round(v * 255).toString(16).padStart(2, '0')).join('');
}

export interface Library {
  types: TypeRec[];
  beads: Float32Array<ArrayBuffer>;         // vec4 per bead: xyz, w = bits(kind | plddt << 8)
  st: Float32Array<ArrayBuffer>;            // sphere tree xyzr
  clusters: ArrayBuffer;       // per cluster: vec4f centre/radius + vec4u (offset, count, 0, 0)
  typeInfo: ArrayBuffer;       // 16 x u32 per type
  modes: Uint32Array<ArrayBuffer>;          // conformational motion modes (library/modes.bin; one zero word when absent)
  modeCap: number;             // nm, largest excursion of one bead along its path (0 = no cap)
  modules: { label: string; n: number }[];   // STRING interaction modules (library/organisation.json; empty when absent)
  moduleOf: Int32Array;        // per type its dominant module (-1: none)
  nClusters: number;
  byId: Map<string, number>;
}

export async function loadLibrary(progress: (msg: string, f: number) => void): Promise<Library> {
  progress('Loading structure index', 0.02);
  const tj = await fetchJson('library/types.json');
  const types: TypeRec[] = tj.types;
  const q = tj.bead_q as number;
  progress('Loading residue coordinates', 0.06);
  const raw = await fetchBin('library/beads.bin', (n, t) => progress('Loading residue coordinates', 0.06 + 0.5 * (t ? n / t : 0)));
  const st = new Float32Array(await fetchBin('library/spheretree.bin'));
  // optional: pool-ensemble motion modes per AlphaFold monomer type (exploration branch)
  let modes = new Uint32Array(4), modeIdx: Record<string, number[]> = {}, modeCap = 0;
  if (!(await isAbsent('library/modes.json'))) try {
    const mj = await fetchJson('library/modes.json');
    modes = new Uint32Array(await fetchBin('library/modes.bin')); modeIdx = mj.types; modeCap = mj.cap ?? 1.5;
  } catch { /* no modes in this build */ }
  // optional: STRING interaction module of each type (organisation layer of the packing)
  let modules: { label: string; n: number }[] = [], modOf: Record<string, number> = {};
  if (!(await isAbsent('library/organisation.json'))) try {
    const oj = await fetchJson('library/organisation.json');
    modules = oj.modules; modOf = oj.types;
  } catch { /* no organisation layer in this build */ }
  progress('Building level-of-detail clusters', 0.6);
  const nB = raw.byteLength / 8;
  const i16 = new Int16Array(raw), u8 = new Uint8Array(raw);
  const beads = new Float32Array(nB * 4), bu = new Uint32Array(beads.buffer);
  for (let i = 0; i < nB; i++) {
    beads[4 * i] = i16[4 * i] * q; beads[4 * i + 1] = i16[4 * i + 1] * q; beads[4 * i + 2] = i16[4 * i + 2] * q;
    bu[4 * i + 3] = u8[8 * i + 6] | (u8[8 * i + 7] << 8);
  }
  // 64-bead clusters (beads are Morton-ordered within each type)
  let nC = 0;
  for (const t of types) nC += Math.ceil(t.nb / 64);
  const clusters = new ArrayBuffer(nC * 32), cf = new Float32Array(clusters), cu = new Uint32Array(clusters);
  const typeInfo = new ArrayBuffer(types.length * 64), tu = new Uint32Array(typeInfo), tf = new Float32Array(typeInfo);
  let c = 0;
  types.forEach((t, ti) => {
    const c0 = c;
    for (let o = 0; o < t.nb; o += 64) {
      const n = Math.min(64, t.nb - o), b0 = t.bo + o;
      let mnx = 1e9, mny = 1e9, mnz = 1e9, mxx = -1e9, mxy = -1e9, mxz = -1e9;
      for (let k = 0; k < n; k++) {
        const j = 4 * (b0 + k), x = beads[j], y = beads[j + 1], z = beads[j + 2];
        if (x < mnx) mnx = x; if (x > mxx) mxx = x; if (y < mny) mny = y; if (y > mxy) mxy = y; if (z < mnz) mnz = z; if (z > mxz) mxz = z;
      }
      const cx = (mnx + mxx) / 2, cy = (mny + mxy) / 2, cz = (mnz + mxz) / 2;
      let r = 0;
      for (let k = 0; k < n; k++) {
        const j = 4 * (b0 + k);
        r = Math.max(r, Math.hypot(beads[j] - cx, beads[j + 1] - cy, beads[j + 2] - cz));
      }
      cf[8 * c] = cx; cf[8 * c + 1] = cy; cf[8 * c + 2] = cz; cf[8 * c + 3] = r + 0.85 + (modeIdx[ti]?.[3] ?? 0);
      cu[8 * c + 4] = b0; cu[8 * c + 5] = n;
      c++;
    }
    const sp = specialFor(t.id);
    const isPDB = t.src === 'PDB';
    tu[16 * ti + 0] = t.bo; tu[16 * ti + 1] = t.nb; tu[16 * ti + 2] = c0; tu[16 * ti + 3] = c - c0;
    tu[16 * ti + 4] = t.so; tu[16 * ti + 5] = t.sn;
    tf[16 * ti + 6] = t.br; tf[16 * ti + 7] = Math.sqrt(5 / 3) * t.rg;
    tu[16 * ti + 8] = sp ? packColor(sp.c0) : 0; tu[16 * ti + 9] = sp && sp.c1 ? packColor(sp.c1) : 0;
    tu[16 * ti + 10] = ((isPDB ? 1 : 0) | (Math.min(t.copies, 0xFFFFFF) << 8)) >>> 0;
    tf[16 * ti + 11] = isPDB ? 255 : (t.plddt_mean ?? 255);
    const mi = modeIdx[ti];
    if (mi) { tu[16 * ti + 12] = mi[0]; tu[16 * ti + 13] = mi[1]; tu[16 * ti + 14] = mi[2]; }
    tu[16 * ti + 15] = modOf[ti] !== undefined ? packColor(moduleColor(modOf[ti])) : 0;
  });
  const byId = new Map<string, number>();
  types.forEach((t, i) => byId.set(t.id, i));
  const moduleOf = new Int32Array(types.length).fill(-1);
  for (const [ti, m] of Object.entries(modOf)) moduleOf[Number(ti)] = m;
  return { types, beads, st, clusters, typeInfo, modes, modeCap, modules, moduleOf, nClusters: nC, byId };
}

export interface TileIndex {
  n: number; ix: Int32Array; iy: Int32Array; iz: Int32Array; nInst: Uint32Array; nGhost: Uint32Array;
  nMr: Uint32Array; nLk: Uint32Array; nVert: Uint32Array; nTri: Uint32Array; gz: Uint32Array;
  labelMask: Uint32Array; volGlucan: Float32Array; volMannan: Float32Array; key: Map<number, number>;
  wall: any; regionNames: string[];   // wall: yeast cell-wall parameters, null for Jurkat (no cell wall)
  // whole-cell build (jenc_full.py): tiles are gzip members of per-block shard files, read with HTTP Range requests
  wholeCell: boolean; off: Float64Array;
  shard: { block_tiles: number; path: string } | null;
  overview: { path: string; block_nm: number; columns: string[]; blocks: number[][] } | null;
  typesTilesU32: boolean;
}
// tile-state grid (fill shader: which tiles have molecules / stand-ins): TGRID tiles per axis, -64 .. 63 (+-12.8 um; the
// whole Jurkat cell spans about -58 .. 57, the 1.6 um sub-volume -4 .. 3). shaders.ts tileLoaded uses the same numbers.
export const TGRID = 128, TOFF = 64;
export const tkey = (ix: number, iy: number, iz: number) => ((ix + TOFF) * TGRID + (iy + TOFF)) * TGRID + (iz + TOFF);

export async function loadTileIndex(): Promise<TileIndex> {
  const j = await fetchJson('tiles/index.json');
  const cols: string[] = j.tile_columns; const T: number[][] = j.tiles; const n = T.length;
  const ci = (k: string) => cols.indexOf(k);
  const mk = <A extends Int32Array | Uint32Array | Float32Array>(C: new (n: number) => A, k: string) => {
    const a = new C(n), c = ci(k); if (c < 0) return a;    // optional column (e.g. yeast cell-wall volumes): zeros
    for (let i = 0; i < n; i++) a[i] = T[i][c]; return a;
  };
  const idx: TileIndex = {
    n, ix: mk(Int32Array, 'ix'), iy: mk(Int32Array, 'iy'), iz: mk(Int32Array, 'iz'),
    nInst: mk(Uint32Array, 'n_inst'), nGhost: mk(Uint32Array, 'n_ghost'), nMr: mk(Uint32Array, 'n_mrna_beads'),
    nLk: mk(Uint32Array, 'n_link_beads'), nVert: mk(Uint32Array, 'n_vert'), nTri: mk(Uint32Array, 'n_tri'),
    gz: mk(Uint32Array, 'gz_bytes'), labelMask: mk(Uint32Array, 'label_mask'),
    volGlucan: mk(Float32Array, 'vol_glucan_nm3'), volMannan: mk(Float32Array, 'vol_mannan_nm3'),
    key: new Map(), wall: j.wall ?? null, regionNames: j.region_names,
    wholeCell: !!j.whole_cell, off: new Float64Array(n), shard: j.shard ?? null, overview: j.overview ?? null,
    typesTilesU32: !!j.types_tiles_u32,
  };
  const co = ci('off');
  if (co >= 0) for (let i = 0; i < n; i++) idx.off[i] = T[i][co];
  for (let i = 0; i < n; i++) idx.key.set(tkey(idx.ix[i], idx.iy[i], idx.iz[i]), i);
  return idx;
}
