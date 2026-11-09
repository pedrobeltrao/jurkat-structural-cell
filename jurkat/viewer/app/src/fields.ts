// Signed-distance fields of the real cell shapes (pipeline: cellshape.py, sampled as in cell3d.field_sdf).
// One Float32Array: [number of fields, 12 header values per field (grid origin xyz, spacing, bricks along x/y/z,
// voxels per brick, clamp inside, clamp outside, index offset, data offset), then the index and data sections].
// Index per brick: -1 inside, -2 outside, else the brick's slot; a slot holds (B+1)^3 node values.
// Without fields the cell is the analytic sphere of radius G[0].
// Jurkat (geometry.json kind 'jurkat_fields', jenc.web_fields): six signed distances in nm, negative inside:
// 0 cell (plasma-membrane midsurface; outer depth = -field), 1 ne_mid (perinuclear midsurface, negative toward the
// nucleoplasm), 2 mito (outer membrane), 3 er, 4 golgi, 5 endo. Same binary layout as yeast, 6 fields instead of 3.
export const F_CELL = 0, F_NE = 1, F_MITO = 2, F_ER = 3, F_GOLGI = 4, F_ENDO = 5;

export function nFields(F: Float32Array | null): number { return F && F.length ? F[0] : 0; }

export function fieldSdf(F: Float32Array, x: number, y: number, z: number, k: number): number {
  const o = 1 + 12 * k, h = F[o + 3], B = F[o + 7];
  const fx = (x - F[o]) / h, fy = (y - F[o + 1]) / h, fz = (z - F[o + 2]) / h;
  const nbx = F[o + 4], nby = F[o + 5], nbz = F[o + 6];
  if (fx < 0 || fy < 0 || fz < 0 || fx >= nbx * B || fy >= nby * B || fz >= nbz * B) return F[o + 9];
  const ix = Math.floor(fx), iy = Math.floor(fy), iz = Math.floor(fz);
  const bx = Math.floor(ix / B), by = Math.floor(iy / B), bz = Math.floor(iz / B);
  const bi = F[F[o + 10] + (bx * nby + by) * nbz + bz];
  if (bi === -1) return -F[o + 8];
  if (bi === -2) return F[o + 9];
  const n1 = B + 1, sx = n1 * n1, sy = n1;
  const tx = fx - ix, ty = fy - iy, tz = fz - iz;
  const b = F[o + 11] + bi * n1 * n1 * n1 + ((ix - bx * B) * n1 + (iy - by * B)) * n1 + (iz - bz * B);
  const c00 = F[b] + (F[b + sx] - F[b]) * tx, c01 = F[b + 1] + (F[b + sx + 1] - F[b + 1]) * tx;
  const c10 = F[b + sy] + (F[b + sx + sy] - F[b + sy]) * tx, c11 = F[b + sy + 1] + (F[b + sx + sy + 1] - F[b + sy + 1]) * tx;
  const c0 = c00 + (c10 - c00) * ty, c1 = c01 + (c11 - c01) * ty;
  return c0 + (c1 - c0) * tz;
}

/** depth below the outer cell surface (negative outside) */
export function outerDepth(F: Float32Array | null, G: number[], x: number, y: number, z: number): number {
  return nFields(F) > F_CELL ? -fieldSdf(F!, x, y, z, F_CELL) : G[0] - Math.hypot(x, y, z);
}

/** outward normal of the cell surface near (x, y, z) */
export function outerNormal(F: Float32Array | null, G: number[], x: number, y: number, z: number, e = 2): [number, number, number] {
  if (nFields(F) <= F_CELL) { const r = Math.hypot(x, y, z) || 1; return [x / r, y / r, z / r]; }
  const d = (a: number, b: number, c: number) => outerDepth(F, G, a, b, c);
  const gx = d(x - e, y, z) - d(x + e, y, z), gy = d(x, y - e, z) - d(x, y + e, z), gz = d(x, y, z - e) - d(x, y, z + e);
  const l = Math.hypot(gx, gy, gz);
  return l > 1e-9 ? [gx / l, gy / l, gz / l] : [0, 0, 0];
}

/** the fields described in geometry.json ("fields"), from their binary file; null when the build has none */
export async function loadFields(geo: any, fetchBin: (p: string) => Promise<ArrayBuffer>): Promise<Float32Array | null> {
  const fd = geo.fields;
  if (!fd) return null;
  const buf = await fetchBin(fd.file);
  const n = fd.head.length;
  let total = 1 + 12 * n;
  for (let k = 0; k < n; k++) total += fd.n_index[k] + fd.n_data[k];
  const F = new Float32Array(total);
  F[0] = n;
  let off = 0;
  for (let k = 0; k < n; k++) {
    F.set(fd.head[k], 1 + 12 * k);
    const idx = new Int32Array(buf, off, fd.n_index[k]); off += 4 * fd.n_index[k];
    F.set(idx, fd.head[k][10]);
    const dat = new Int16Array(buf, off, fd.n_data[k]); off += 2 * fd.n_data[k]; off = Math.ceil(off / 4) * 4;
    const d0 = fd.head[k][11];
    for (let i = 0; i < dat.length; i++) F[d0 + i] = dat[i] * fd.q;
  }
  return F;
}
