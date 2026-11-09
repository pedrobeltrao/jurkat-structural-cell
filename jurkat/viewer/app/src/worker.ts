/// <reference lib="webworker" />
// Tile decoder + procedural near-camera detail (lipids; the yeast cell-wall polymers are removed for Jurkat).
// All procedural content is deterministic (seeded by tile and element index), so regenerating a region
// after the cut plane moves reproduces exactly the same molecules.
import { outerDepth } from './fields';

const TILE = 200, Q_ORG = 80, QSTEP = 360 / 65535, MR_Q = 0.005, LK_Q = 0.01;
const INV_SQRT2 = 0.70710678;

let base = '';
let embedded: Record<string, Uint8Array> = {};
let bo: Uint32Array, nbArr: Uint32Array, brArr: Float32Array;
let beadXYZ: Float32Array;   // library beads (nm), 3 per bead
let ready: Promise<void> | null = null;
let G: number[] = [];
let SDF: Float32Array | null = null;   // real-shape distance fields (fields.ts); null: spherical cell

interface Decoded {
  lo: [number, number, number];
  inst: Uint32Array; ghost: Uint32Array;
  verts: Float32Array; normals: Float32Array; tris: Uint32Array; fmem: Uint8Array; vmem: Uint8Array;
}
let ver = '';
const cache = new Map<number, Decoded>();
const CACHE_MAX = 48;

let split: Record<string, number> = {};     // data/split.json, as data.ts fetchBin: parts path.NNN, base64 text at path.txt
async function getRaw(path: string): Promise<ArrayBuffer> {
  const b64 = !!split._b64 && !path.endsWith('.json');
  const r = await fetch(base + path + (b64 ? '.txt' : '') + (ver ? `?v=${encodeURIComponent(ver)}` : ''));
  if (!r.ok) throw new Error(`HTTP ${r.status} ${path}`);
  if (!b64) return r.arrayBuffer();
  const bin = atob(await r.text()); const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u.buffer;
}
async function getBin(path: string): Promise<ArrayBuffer> {
  if (embedded[path]) return gunzip(embedded[path].slice().buffer);
  const np = split[path];
  if (!np) return gunzip(await getRaw(path));
  const parts: ArrayBuffer[] = []; let n = 0;
  for (let i = 0; i < np; i++) { const b = await getRaw(`${path}.${String(i).padStart(3, '0')}`); parts.push(b); n += b.byteLength; }
  const out = new Uint8Array(n); let o = 0;
  for (const b of parts) { out.set(new Uint8Array(b), o); o += b.byteLength; }
  return gunzip(out.buffer);
}
// whole-cell builds (jenc_full.py): a tile is the gzip member at bytes [off, off + n) of its block shard. Fetched with an
// HTTP Range request; a server that ignores Range (status 200, e.g. python -m http.server) or a failed ranged request
// (CORS) falls back to the whole shard, kept in a small cache so its other tiles are sliced locally.
const shardCache = new Map<string, Promise<ArrayBuffer>>();
const SHARD_CACHE_MAX = 3;
let rangeOk = true;
function wholeShard(path: string): Promise<ArrayBuffer> {
  let p = shardCache.get(path);
  if (!p) {
    p = fetch(base + path + (ver ? `?v=${encodeURIComponent(ver)}` : '')).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status} ${path}`);
      return r.arrayBuffer();
    });
    p.catch(() => shardCache.delete(path));
    shardCache.set(path, p);
    while (shardCache.size > SHARD_CACHE_MAX) shardCache.delete(shardCache.keys().next().value!);
  }
  return p;
}
async function getTileBytes(m: any): Promise<ArrayBuffer> {
  if (!m.shard) return getBin(`tiles/${m.ix}_${m.iy}_${m.iz}.bin.gz`);
  const { path, off, n } = m.shard as { path: string; off: number; n: number };
  const cached = shardCache.get(path);
  if (cached) return gunzip((await cached).slice(off, off + n));
  if (rangeOk) {
    let r: Response | null = null;
    try {
      r = await fetch(base + path + (ver ? `?v=${encodeURIComponent(ver)}` : ''), { headers: { Range: `bytes=${off}-${off + n - 1}` } });
    } catch (e) {                                  // network / CORS failure of the ranged request
      console.warn('ranged fetch failed, using whole shards', e);
      rangeOk = false;
    }
    if (r) {
      if (r.status === 206) return gunzip(await r.arrayBuffer());
      if (!r.ok) throw new Error(`HTTP ${r.status} ${path}`);
      const whole = r.arrayBuffer();               // Range ignored: this response is the whole shard
      shardCache.set(path, whole);
      while (shardCache.size > SHARD_CACHE_MAX) shardCache.delete(shardCache.keys().next().value!);
      return gunzip((await whole).slice(off, off + n));
    }
  }
  return gunzip((await wholeShard(path)).slice(off, off + n));
}
async function gunzip(buf: ArrayBuffer): Promise<ArrayBuffer> {
  const u = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  if (u[0] === 0x1f && u[1] === 0x8b) {
    return await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  }
  return buf;
}

async function init(msg: any) {
  base = msg.base; G = msg.G; SDF = msg.F ?? null;
  try { const r = await fetch(base + 'version.json', { cache: 'no-store' }); if (r.ok) ver = String((await r.json()).v ?? ''); } catch { ver = ''; }
  try { const r = await fetch(base + 'split.json', { cache: 'no-store' }); if (r.ok) split = await r.json(); } catch { split = {}; }
  if (msg.embedded) embedded = msg.embedded;
  const tj = JSON.parse(new TextDecoder().decode(await getBin('library/types.json')));
  const T = tj.types, n = T.length, q = tj.bead_q;
  bo = new Uint32Array(n); nbArr = new Uint32Array(n); brArr = new Float32Array(n);
  for (let i = 0; i < n; i++) { bo[i] = T[i].bo; nbArr[i] = T[i].nb; brArr[i] = T[i].br; }
  const raw = await getBin('library/beads.bin');
  const i16 = new Int16Array(raw), nB = raw.byteLength / 8;
  beadXYZ = new Float32Array(nB * 3);
  for (let i = 0; i < nB; i++) { beadXYZ[3 * i] = i16[4 * i] * q; beadXYZ[3 * i + 1] = i16[4 * i + 1] * q; beadXYZ[3 * i + 2] = i16[4 * i + 2] * q; }
}

// ------------------------------------------------------------------ tile decoding
const pad4 = (x: number) => (x + 3) & ~3;

function decodePolylines(buf: ArrayBuffer, off: number, n: number, nb: number, lo: number[], q: number, kind: number, out: Float32Array, outU: Uint32Array, o0: number, kinds?: Uint8Array) {
  const lens = new Uint32Array(buf, off, n); off += 4 * n;
  const st = new Int32Array(buf, off, 3 * n); off += 12 * n;
  const d = new Int8Array(buf, off, 3 * (nb - n)); off += 3 * (nb - n);
  off = pad4(off);
  let k = 0, o = o0;
  for (let i = 0; i < n; i++) {
    let x = st[3 * i], y = st[3 * i + 1], z = st[3 * i + 2];
    const L = lens[i];
    for (let j = 0; j < L; j++) {
      if (j > 0) { x += d[3 * k]; y += d[3 * k + 1]; z += d[3 * k + 2]; k++; }
      out[4 * o] = lo[0] + x * q; out[4 * o + 1] = lo[1] + y * q; out[4 * o + 2] = lo[2] + z * q;
      outU[4 * o + 3] = ((kinds ? kinds[i] : kind) | ((i & 0x7FFF) << 4)) >>> 0;
      o++;
    }
  }
  return { off, o };
}

function decodeTile(buf: ArrayBuffer) {
  const h = new Uint32Array(buf, 0, 16);
  if (h[0] !== 0x31544359) throw new Error('bad tile magic');
  const t = new Int32Array(buf, 4, 3);
  const lo: [number, number, number] = [t[0] * TILE, t[1] * TILE, t[2] * TILE];
  const [nI, nG, nM, nMB, nL, nLB, nV, nT, fl, nX, nXB] = [h[4], h[5], h[6], h[7], h[8], h[9], h[10], h[11], h[12], h[13], h[14]];
  let off = 64;
  const inst = new Uint32Array(buf.slice(off, off + 16 * nI)); off += 16 * nI;
  const ghost = new Uint32Array(buf.slice(off, off + 16 * nG)); off += 16 * nG;
  const free = new Float32Array((nMB + nLB + nXB) * 4), freeU = new Uint32Array(free.buffer);
  let o = 0;
  if (nM) { const r = decodePolylines(buf, off, nM, nMB, lo, MR_Q, 0, free, freeU, o); off = r.off; o = r.o; }
  if (nL) { const r = decodePolylines(buf, off, nL, nLB, lo, LK_Q, 1, free, freeU, o); off = r.off; o = r.o; }
  const verts = new Float32Array(nV * 3);
  if (nV) {
    const v = new Uint16Array(buf, off, 3 * nV);
    for (let i = 0; i < nV; i++) for (let a = 0; a < 3; a++) verts[3 * i + a] = lo[a] - Q_ORG + v[3 * i + a] * QSTEP;
    off += pad4(6 * nV);
  }
  let tris = new Uint32Array(nT * 3);
  if (nT) {
    if (fl & 1) tris = new Uint32Array(buf.slice(off, off + 12 * nT));
    else tris = Uint32Array.from(new Uint16Array(buf, off, 3 * nT));
    off += pad4((fl & 1 ? 4 : 2) * 3 * nT);
  }
  const fmem = nT ? new Uint8Array(buf.slice(off, off + nT)) : new Uint8Array(0);
  if (nT) off += pad4(nT);
  if (nX) {   // extra polylines: kind per chain (4 pre-rRNA, 5 disordered protein)
    const kinds = new Uint8Array(buf.slice(off, off + nX)); off += pad4(nX);
    decodePolylines(buf, off, nX, nXB, lo, MR_Q, 0, free, freeU, o, kinds);
  }
  // area-weighted vertex normals (face normals point to the positive-SDF / cytoplasmic side)
  const normals = new Float32Array(nV * 3), vmem = new Uint8Array(nV);
  for (let f = 0; f < nT; f++) {
    const a = tris[3 * f], b = tris[3 * f + 1], c = tris[3 * f + 2];
    const ux = verts[3 * b] - verts[3 * a], uy = verts[3 * b + 1] - verts[3 * a + 1], uz = verts[3 * b + 2] - verts[3 * a + 2];
    const vx = verts[3 * c] - verts[3 * a], vy = verts[3 * c + 1] - verts[3 * a + 1], vz = verts[3 * c + 2] - verts[3 * a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const k of [a, b, c]) { normals[3 * k] += nx; normals[3 * k + 1] += ny; normals[3 * k + 2] += nz; vmem[k] = fmem[f]; }
  }
  for (let i = 0; i < nV; i++) {
    const l = Math.hypot(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]) || 1;
    normals[3 * i] /= l; normals[3 * i + 1] /= l; normals[3 * i + 2] /= l;
  }
  return { lo, inst, ghost, free, verts, normals, tris, fmem, vmem, nI, nG };
}

// ------------------------------------------------------------------ instance decoding (mirrors the WGSL)
const i16f = (x: number) => (((x << 16) >> 16) / 32767) * INV_SQRT2;
function instPose(w: Uint32Array, i: number, lo: number[], P: Float64Array) {
  const w0 = w[4 * i], w1 = w[4 * i + 1], w2 = w[4 * i + 2], w3 = w[4 * i + 3];
  P[0] = lo[0] - Q_ORG + (w1 & 0xFFFF) * QSTEP; P[1] = lo[1] - Q_ORG + (w1 >>> 16) * QSTEP; P[2] = lo[2] - Q_ORG + (w2 & 0xFFFF) * QSTEP;
  const a = i16f(w2 >>> 16), b = i16f(w3 & 0xFFFF), c = i16f(w3 >>> 16);
  const big = Math.sqrt(Math.max(0, 1 - a * a - b * b - c * c));
  const qi = (w0 >>> 14) & 3;
  const q = qi === 0 ? [big, a, b, c] : qi === 1 ? [a, big, b, c] : qi === 2 ? [a, b, big, c] : [a, b, c, big];
  const [qw, qx, qy, qz] = q;
  // rotation matrix (row-major) from (w,x,y,z)
  P[3] = qw * qw + qx * qx - qy * qy - qz * qz; P[4] = 2 * (qx * qy - qw * qz); P[5] = 2 * (qx * qz + qw * qy);
  P[6] = 2 * (qx * qy + qw * qz); P[7] = qw * qw - qx * qx + qy * qy - qz * qz; P[8] = 2 * (qy * qz - qw * qx);
  P[9] = 2 * (qx * qz - qw * qy); P[10] = 2 * (qy * qz + qw * qx); P[11] = qw * qw - qx * qx - qy * qy + qz * qz;
  return { type: w0 & 0x3FFF, region: (w0 >>> 16) & 31, flags: w0 >>> 21 };
}

// ------------------------------------------------------------------ spatial hash of protein beads
class BeadHash {
  h: number; tbl: number; mask: number; start: Uint32Array; pts: Float32Array; inst: Int32Array;
  constructor(xyz: Float32Array, owner: Int32Array, n: number, h: number) {
    this.h = h;
    let tbl = 1; while (tbl < n * 2) tbl <<= 1; tbl = Math.max(tbl, 1024);
    this.tbl = tbl; this.mask = tbl - 1;
    const key = new Uint32Array(n), cnt = new Uint32Array(tbl + 1);
    for (let i = 0; i < n; i++) {
      key[i] = this.k(Math.floor(xyz[3 * i] / h), Math.floor(xyz[3 * i + 1] / h), Math.floor(xyz[3 * i + 2] / h));
      cnt[key[i] + 1]++;
    }
    for (let i = 0; i < tbl; i++) cnt[i + 1] += cnt[i];
    this.start = cnt.slice();
    this.pts = new Float32Array(n * 3); this.inst = new Int32Array(n);
    const fill = cnt.slice(0, tbl);
    for (let i = 0; i < n; i++) {
      const j = fill[key[i]]++;
      this.pts[3 * j] = xyz[3 * i]; this.pts[3 * j + 1] = xyz[3 * i + 1]; this.pts[3 * j + 2] = xyz[3 * i + 2];
      this.inst[j] = owner[i];
    }
  }
  k(x: number, y: number, z: number) { return (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) & this.mask; }
  // true if any bead within r of (x,y,z) (r <= h); optionally ignore beads of instance 'skip'
  near(x: number, y: number, z: number, r: number, skip = -1): boolean {
    const h = this.h, cx = Math.floor(x / h), cy = Math.floor(y / h), cz = Math.floor(z / h), r2 = r * r;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const key = this.k(cx + dx, cy + dy, cz + dz);
      for (let j = this.start[key], e = this.start[key + 1]; j < e; j++) {
        const ax = this.pts[3 * j] - x, ay = this.pts[3 * j + 1] - y, az = this.pts[3 * j + 2] - z;
        if (ax * ax + ay * ay + az * az < r2 && this.inst[j] !== skip) return true;
      }
    }
    return false;
  }
}

function collectBeads(list: { w: Uint32Array; i: number; lo: number[]; id: number }[]) {
  let n = 0;
  const P = new Float64Array(12);
  for (const e of list) n += nbArr[e.w[4 * e.i] & 0x3FFF];
  const xyz = new Float32Array(n * 3), owner = new Int32Array(n);
  let k = 0;
  for (const e of list) {
    const { type } = instPose(e.w, e.i, e.lo, P);
    const b0 = bo[type], nb = nbArr[type];
    for (let j = 0; j < nb; j++) {
      const bx = beadXYZ[3 * (b0 + j)], by = beadXYZ[3 * (b0 + j) + 1], bz = beadXYZ[3 * (b0 + j) + 2];
      xyz[3 * k] = P[0] + P[3] * bx + P[4] * by + P[5] * bz;
      xyz[3 * k + 1] = P[1] + P[6] * bx + P[7] * by + P[8] * bz;
      xyz[3 * k + 2] = P[2] + P[9] * bx + P[10] * by + P[11] * bz;
      owner[k] = e.id; k++;
    }
  }
  return { xyz, owner, n };
}

// ------------------------------------------------------------------ RNG
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const hash3 = (a: number, b: number, c: number) => {
  let x = Math.imul(a ^ 0x9E3779B9, 0x85EBCA6B) ^ Math.imul(b + 0x632BE5AB, 0xC2B2AE35) ^ Math.imul(c + 0x27D4EB2F, 0x165667B1);
  x ^= x >>> 16; x = Math.imul(x, 0x7feb352d); x ^= x >>> 15; x = Math.imul(x, 0x846ca68b); x ^= x >>> 16;
  return x >>> 0;
};
function gauss(r: () => number) { return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(6.2831853 * r()); }

function octEncode(nx: number, ny: number, nz: number) {
  const s = Math.abs(nx) + Math.abs(ny) + Math.abs(nz) || 1;
  let x = nx / s, y = ny / s;
  if (nz < 0) { const ox = x; x = (1 - Math.abs(y)) * (ox >= 0 ? 1 : -1); y = (1 - Math.abs(ox)) * (y >= 0 ? 1 : -1); }
  const ex = Math.max(0, Math.min(255, Math.round((x + 1) * 127.5))), ey = Math.max(0, Math.min(255, Math.round((y + 1) * 127.5)));
  return ex | (ey << 8);
}

// ------------------------------------------------------------------ detail generation
// lipid class weights (mol %), mammalian default: PI, PE, PC, SM, PS, (sterol slot: cholesterol is not drawn as a
// headgroup), other. Indices match data.LIPID_CLASSES and the lipid palette 32 + class.
const LCUM = (() => { const w = [5, 20, 45, 10, 8, 0, 12]; const s = w.reduce((a, b) => a + b); let c = 0; return w.map(x => (c += x / s)); })();
const LIPID_AREA = 0.65;   // nm^2 per lipid per leaflet
const MASK_R = 0.95;

interface Range { n: number[]; d: number; lo: number; hi: number; mode: 'slab' | 'shell' }
function sOf(R: Range, x: number, y: number, z: number) { return R.n[0] * x + R.n[1] * y + R.n[2] * z - R.d; }

function genDetail(tileIdx: number, D: Decoded, R: Range, want: { lipids: boolean; wall: boolean }) {
  const lipids: number[] = [];
  const free: number[] = [];
  const stats = { lipCand: 0, lipMasked: 0 };
  const P = new Float64Array(12);
  const inRange = (x: number, y: number, z: number, m: number) => {
    if (R.mode === 'shell') return outerDepth(SDF, G, x, y, z) < 70 + m;
    const s = sOf(R, x, y, z); return s >= R.lo - m && s <= R.hi + m;
  };

  // ---- lipids on the tile's membrane triangles
  if (want.lipids && D.tris.length && R.mode === 'slab') {
    const list: { w: Uint32Array; i: number; lo: number[]; id: number }[] = [];
    const nI = D.inst.length / 4, nG = D.ghost.length / 4;
    for (let i = 0; i < nI; i++) if ((D.inst[4 * i] >>> 21) & 1) list.push({ w: D.inst, i, lo: D.lo, id: i });
    // ghosts carry their own tile origin implicitly: positions are encoded relative to *this* tile's origin
    for (let i = 0; i < nG; i++) list.push({ w: D.ghost, i, lo: D.lo, id: nI + i });
    const B = collectBeads(list);
    const hash = new BeadHash(B.xyz, B.owner, B.n, 1.0);
    const V = D.verts, N = D.normals, T = D.tris;
    const nT = T.length / 3;
    for (let f = 0; f < nT; f++) {
      const ia = T[3 * f], ib = T[3 * f + 1], ic = T[3 * f + 2];
      const ax = V[3 * ia], ay = V[3 * ia + 1], az = V[3 * ia + 2];
      const bx = V[3 * ib], by = V[3 * ib + 1], bz = V[3 * ib + 2];
      const cx = V[3 * ic], cy = V[3 * ic + 1], cz = V[3 * ic + 2];
      const s0 = sOf(R, ax, ay, az), s1 = sOf(R, bx, by, bz), s2 = sOf(R, cx, cy, cz);
      if (Math.max(s0, s1, s2) < R.lo - 3 || Math.min(s0, s1, s2) > R.hi + 3) continue;
      const ux = bx - ax, uy = by - ay, uz = bz - az, vx = cx - ax, vy = cy - ay, vz = cz - az;
      const area = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
      const mem = D.fmem[f];
      const leaflets = [-1, 1];          // bilayers only (no lipid-droplet monolayer in the Jurkat model)
      for (const sg of leaflets) {
        const r = rng(hash3(tileIdx, f, sg + 7));
        const nexp = area / LIPID_AREA;
        const Nn = Math.floor(nexp) + (r() < nexp - Math.floor(nexp) ? 1 : 0);
        if (!Nn) continue;
        const k = Math.ceil(Math.sqrt(Nn)), k2 = k * k;
        // choose Nn of the k^2 sub-triangles (stratified, blue-noise-like sampling)
        const cells = new Uint16Array(k2); for (let i = 0; i < k2; i++) cells[i] = i;
        for (let i = 0; i < Nn; i++) { const j = i + Math.floor(r() * (k2 - i)); const t = cells[i]; cells[i] = cells[j]; cells[j] = t; }
        for (let m = 0; m < Nn; m++) {
          const c = cells[m], row = Math.floor(Math.sqrt(c)), col = c - row * row;
          const jj = col >> 1, up = (col & 1) === 0;
          // lattice points L(r,s) -> barycentric (alpha, beta) = ((r-s)/k, s/k)
          let p0r, p0s, p1r, p1s, p2r, p2s;
          if (up) { p0r = row; p0s = jj; p1r = row + 1; p1s = jj; p2r = row + 1; p2s = jj + 1; }
          else { p0r = row; p0s = jj; p1r = row + 1; p1s = jj + 1; p2r = row; p2s = jj + 1; }
          let uu = r(), vv = r(); if (uu + vv > 1) { uu = 1 - uu; vv = 1 - vv; }
          const rr = p0r + uu * (p1r - p0r) + vv * (p2r - p0r), ss = p0s + uu * (p1s - p0s) + vv * (p2s - p0s);
          const al = (rr - ss) / k, be = ss / k, ga = 1 - al - be;
          const px = ax + al * ux + be * vx, py = ay + al * uy + be * vy, pz = az + al * uz + be * vz;
          let nx = ga * N[3 * ia] + al * N[3 * ib] + be * N[3 * ic];
          let ny = ga * N[3 * ia + 1] + al * N[3 * ib + 1] + be * N[3 * ic + 1];
          let nz = ga * N[3 * ia + 2] + al * N[3 * ib + 2] + be * N[3 * ic + 2];
          const nl = Math.hypot(nx, ny, nz) || 1; nx /= nl; ny /= nl; nz /= nl;
          const u = r();
          let cls = 0; while (cls < 6 && u > LCUM[cls]) cls++;
          stats.lipCand++;
          // record point: bilayer midsurface
          const qx = px, qy = py, qz = pz;
          const s = sOf(R, qx, qy, qz);
          if (s < R.lo - 2.5 || s > R.hi + 2.5) continue;
          const offs = cls === 5 ? [1.55, 0.75] : [1.95, 1.25, 0.55];
          let hit = false;
          for (const o of offs) {
            if (hash.near(qx + nx * sg * o, qy + ny * sg * o, qz + nz * sg * o, MASK_R)) { hit = true; break; }
          }
          if (hit) { stats.lipMasked++; continue; }
          const w = (octEncode(nx, ny, nz) | (cls << 16) | ((sg > 0 ? 1 : 0) << 19) | ((mem & 15) << 20)) >>> 0;
          lipids.push(qx, qy, qz, w);
        }
      }
    }
  }

  const L = new Float32Array(lipids.length), LU = new Uint32Array(L.buffer);
  for (let i = 0; i < lipids.length; i += 4) { L[i] = lipids[i]; L[i + 1] = lipids[i + 1]; L[i + 2] = lipids[i + 2]; LU[i + 3] = lipids[i + 3]; }
  const F = new Float32Array(free.length), FU = new Uint32Array(F.buffer);
  for (let i = 0; i < free.length; i += 4) { F[i] = free[i]; F[i + 1] = free[i + 1]; F[i + 2] = free[i + 2]; FU[i + 3] = free[i + 3]; }
  return { lipids: L, free: F, stats };
}

// ------------------------------------------------------------------ message loop
self.onmessage = async (ev: MessageEvent) => {
  const m = ev.data;
  try {
    if (m.cmd === 'init') {
      ready = init(m);
      await ready;
      (self as any).postMessage({ cmd: 'ready' });
      return;
    }
    await ready;
    if (m.cmd === 'load') {
      const buf = await getTileBytes(m);
      const T = decodeTile(buf);
      const nV = T.verts.length / 3;
      const vb = new Float32Array(nV * 8), vbu = new Uint32Array(vb.buffer);
      for (let i = 0; i < nV; i++) {
        vb[8 * i] = T.verts[3 * i]; vb[8 * i + 1] = T.verts[3 * i + 1]; vb[8 * i + 2] = T.verts[3 * i + 2]; vbu[8 * i + 3] = T.vmem[i];
        vb[8 * i + 4] = T.normals[3 * i]; vb[8 * i + 5] = T.normals[3 * i + 1]; vb[8 * i + 6] = T.normals[3 * i + 2];
      }
      cache.set(m.idx, { lo: T.lo, inst: T.inst, ghost: T.ghost, verts: T.verts, normals: T.normals, tris: T.tris, fmem: T.fmem, vmem: T.vmem });
      while (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value!);
      const inst = T.inst.slice();
      const ib = T.tris.slice();
      (self as any).postMessage({ cmd: 'loaded', idx: m.idx, inst, free: T.free, vb, ib, lo: T.lo },
        [inst.buffer, T.free.buffer, vb.buffer, ib.buffer]);
    } else if (m.cmd === 'detail') {
      let D = cache.get(m.idx);
      if (!D) {
        const T = decodeTile(await getTileBytes(m));
        D = { lo: T.lo, inst: T.inst, ghost: T.ghost, verts: T.verts, normals: T.normals, tris: T.tris, fmem: T.fmem, vmem: T.vmem };
        cache.set(m.idx, D);
      } else { cache.delete(m.idx); cache.set(m.idx, D); }
      const t0 = performance.now();
      const out = genDetail(m.idx, D, m.range, m.want);
      (self as any).postMessage({ cmd: 'detail', idx: m.idx, key: m.key, lipids: out.lipids, free: out.free, stats: out.stats, ms: performance.now() - t0 },
        [out.lipids.buffer, out.free.buffer]);
    }
  } catch (e: any) {
    (self as any).postMessage({ cmd: 'error', idx: m.idx, key: m.key, req: m.cmd, error: String(e && e.stack || e) });
  }
};
