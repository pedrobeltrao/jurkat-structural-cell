// JavaScript port of the compartment geometry (same as the WGSL fill shader) for the minimap and picking.
// Jurkat T cell: six distance fields (fields.ts) and nuclear pores; label rule = cell3d.mem_sdfs + cell3d.label_from
// of the Jurkat pipeline (jurkat/yc/pipeline/cell3d.py).
import { F_CELL, F_NE, F_MITO, F_ER, F_GOLGI, F_ENDO, nFields, fieldSdf } from './fields';

// label codes (data.LABEL_NAMES order = cell3d.LABEL_NAMES)
export const L_EXT = 0, L_CYTO = 1, L_PERI = 2, L_NUCP = 3, L_NUCL = 4, L_IMS = 5, L_MATRIX = 6, L_ERL = 7, L_GOLGI = 8, L_ENDO = 9, L_PORE = 10;
// membrane codes (data.MEM_NAMES order = jenc.MEM_NAMES = face_mem)
export const M_PM = 0, M_NE = 1, M_ER = 2, M_OM = 3, M_IM = 4, M_GOLGI = 5, M_ENDO = 6, NMEM = 7;

const PORE_AXIAL = 150;     // cell3d.in_pore: |z| along the pore axis at most this
const POREGRID_N = 8;        // pore lookup grid for the fill shader (cells per axis)

/**
 * Complete a 'jurkat_fields' geometry.json (after loadFields set geo.F) with the members the renderer and the fill
 * shader expect from the yeast format: G (G[0..2] = half edge of the sub-volume cube: the fill shader intersects rays
 * with that cube), empty primitive lists (PT/PQ/PC), the pores near the cube and a uniform pore lookup grid.
 */
export function prepareGeometry(geo: any) {
  if (geo.kind !== 'jurkat_fields') return geo;
  const half = geo.cube_nm / 2;
  geo.half = half;
  geo.cellR = geo.R;                                  // radius of a sphere enclosing the whole cell (model frame centred on the cell)
  geo.G = new Array(23).fill(0);
  geo.G[0] = half; geo.G[1] = half; geo.G[2] = half;
  geo.G[11] = 1; geo.G[12] = 1; geo.G[13] = 1;         // (yeast nucleus semi-axes; unused, non-zero for safety)
  geo.PT = []; geo.PQ = []; geo.PC = [];
  const c = geo.consts, rc = c.PORE_R + 0.5 * c.PERI_W;
  const reach = Math.hypot(PORE_AXIAL, rc) + 2;        // a pore influences points at most this far from its centre
  // pores whose influence region meets the cube (the full list covers the whole nucleus)
  const all: number[] = geo.pores ?? [], P: number[] = [];
  for (let i = 0; i + 5 < all.length; i += 6) {
    if (Math.abs(all[i]) < half + reach && Math.abs(all[i + 1]) < half + reach && Math.abs(all[i + 2]) < half + reach) P.push(...all.slice(i, i + 6));
  }
  geo.poresAll = all.length / 6;
  geo.pores = P;
  // uniform grid over the cube (+ margin): per cell the pores whose influence box overlaps it
  // whole cell (jenc_full.py): 1,000 pores over a ~23 um cube need a finer grid to keep the per-cell lists short
  const n = geo.whole_cell ? 48 : POREGRID_N, ext = half + 50, h = 2 * ext / n, n3 = n ** 3;
  const lists: number[][] = Array.from({ length: n3 }, () => []);
  const nP = P.length / 6;
  const clampI = (v: number) => Math.min(n - 1, Math.max(0, Math.floor((v + ext) / h)));
  for (let k = 0; k < nP; k++) {
    const x = P[6 * k], y = P[6 * k + 1], z = P[6 * k + 2];
    for (let i = clampI(x - reach); i <= clampI(x + reach); i++)
      for (let j = clampI(y - reach); j <= clampI(y + reach); j++)
        for (let l = clampI(z - reach); l <= clampI(z + reach); l++) lists[(i * n + j) * n + l].push(k);
  }
  const pore_off = new Array(n3 + 1).fill(0), pore_idx: number[] = [];
  for (let i = 0; i < n3; i++) { pore_off[i] = pore_idx.length; pore_idx.push(...lists[i]); }
  pore_off[n3] = pore_idx.length;
  geo.grid = { n, ext, prim_off: new Array(n3 + 1).fill(0), prim_idx: [], pore_off, pore_idx };
  return geo;
}

export interface MemSdf { d: number[]; matrix: number; hp: number }

export class Geometry {
  G: number[]; pores: number[]; c: any; grid: any; n3: number;
  F: Float32Array | null;
  constructor(geo: any) {
    this.F = geo.F ?? null;
    this.G = geo.G; this.pores = geo.pores; this.c = geo.consts; this.grid = geo.grid;
    this.n3 = geo.grid.n ** 3;
  }
  cell(x: number, y: number, z: number) {
    const n = this.grid.n, ext = this.grid.ext, h = 2 * ext / n;
    const f = (v: number) => Math.min(n - 1, Math.max(0, Math.floor((v + ext) / h)));
    return (f(x) * n + f(y)) * n + f(z);
  }
  field(k: number, x: number, y: number, z: number): number {
    if (nFields(this.F) <= k) return 1e9;                  // field missing: far outside
    return fieldSdf(this.F!, x, y, z, k);
  }
  /** cell3d.in_pore: index of a pore whose axis passes within rc of p (|axial| <= 150 nm), or -1; also its rho */
  inPore(x: number, y: number, z: number, rc: number): { k: number; rho: number; z: number } {
    const g = this.grid, cell = this.cell(x, y, z), P = this.pores;
    for (let q = g.pore_off[cell]; q < g.pore_off[cell + 1]; q++) {
      const o = 6 * g.pore_idx[q];
      const vx = x - P[o], vy = y - P[o + 1], vz = z - P[o + 2];
      const zz = vx * P[o + 3] + vy * P[o + 4] + vz * P[o + 5];
      if (zz < -PORE_AXIAL || zz > PORE_AXIAL) continue;
      const rx = vx - zz * P[o + 3], ry = vy - zz * P[o + 4], rz = vz - zz * P[o + 5];
      if (rx * rx + ry * ry + rz * rz < rc * rc) return { k: g.pore_idx[q], rho: Math.hypot(rx, ry, rz), z: zz };
    }
    return { k: -1, rho: 0, z: 0 };
  }
  /** cell3d.mem_sdfs: signed distance to each membrane (negative = lumenal / non-cytoplasmic side) */
  memSdfs(x: number, y: number, z: number): MemSdf {
    const PERI = this.c.PERI_W, IMS = this.c.IMS_W, rc = this.c.PORE_R + 0.5 * PERI;
    const d = new Array(NMEM).fill(1e9);
    d[M_PM] = -this.field(F_CELL, x, y, z);                 // outer depth
    const hp = this.field(F_NE, x, y, z);
    let dne = Math.abs(hp) - 0.5 * PERI;
    if (Math.abs(hp) < rc + PERI) {
      const p = this.inPore(x, y, z, rc);
      if (p.k >= 0) dne = Math.sqrt((p.rho - rc) ** 2 + hp * hp) - 0.5 * PERI;   // pore-membrane torus
    }
    d[M_NE] = dne;
    const om = this.field(F_MITO, x, y, z);
    d[M_OM] = om;
    const matrix = om + IMS;
    d[M_IM] = -matrix;
    d[M_ER] = this.field(F_ER, x, y, z);
    d[M_GOLGI] = this.field(F_GOLGI, x, y, z);
    d[M_ENDO] = this.field(F_ENDO, x, y, z);
    return { d, matrix, hp };
  }
  /** cell3d.label_from */
  labelFrom(x: number, y: number, z: number, m: MemSdf): number {
    const PERI = this.c.PERI_W, rc = this.c.PORE_R + 0.5 * PERI;
    if (m.d[M_PM] < 0) return L_EXT;
    if (m.d[M_NE] < 0) return L_PERI;
    const hp = m.hp;
    if (hp < -0.5 * PERI) {
      if (hp > -PERI - 8 && this.inPore(x, y, z, rc).k >= 0) return L_PORE;
      return L_NUCP;
    }
    if (hp < 0) return L_PORE;                               // pore channel between the midplane and the ONM
    if (m.d[M_OM] < 0) return m.matrix < 0 ? L_MATRIX : L_IMS;
    if (m.d[M_ER] < 0) return L_ERL;
    if (m.d[M_GOLGI] < 0) return L_GOLGI;
    if (m.d[M_ENDO] < 0) return L_ENDO;
    return L_CYTO;
  }
  label(x: number, y: number, z: number): number {
    return this.labelFrom(x, y, z, this.memSdfs(x, y, z));
  }
}
