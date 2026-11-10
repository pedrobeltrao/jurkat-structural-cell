// WGSL shaders for the whole-cell viewer.
// Conventions: world units = nm. All rendering is camera-relative (eye at origin) for float precision.
// Reversed-Z infinite perspective (depth 1 = near, 0 = infinitely far). Cut plane s(p) = dot(n,p) - d;
// content with s >= 0 is kept (camera looks along +n in section views).

export const PAGE = 1024;      // instances per pool page
export const SPAGE = 4096;     // lipids / free spheres per pool page

const COMMON = /* wgsl */ `
const PAGE: u32 = ${PAGE}u;
const SPAGE: u32 = ${SPAGE}u;
const INV_SQRT2: f32 = 0.70710678;
// draw-list items (list01 .y): bead / sphere-tree offset in the low OFF_BITS bits, (count - 1) above (count <= 64 beads
// per L0 cluster, <= 16 spheres per L1 item: 6 bits). Jurkat: 26-bit offsets (67 M) instead of the yeast 22 bits (4.2 M).
const OFF_BITS: u32 = 26u;
const OFF_MASK: u32 = 0x3FFFFFFu;
struct U {
  proj: mat4x4f,
  view: mat4x4f,          // rotation only (camera-relative)
  invViewProj: mat4x4f,   // camera-relative clip -> world-relative
  eye: vec4f,             // world eye, w = px per nm at unit view depth
  plane: vec4f,           // world n.xyz, d
  planeV: vec4f,          // view-space n.xyz, sEye = dot(n,eye)-d
  cut: vec4f,             // x enabled, y slab thickness, z fog range, w time
  lod: vec4f,             // x L0 bead px, y L1 instance px, z min px for free spheres/lipids, w protein bead radius
  vp: vec4f,              // width, height, 1/w, 1/h
  frustum: array<vec4f, 6>,  // camera-relative planes, inside: dot(n,p)+w >= 0
  modes: vec4u,           // x colorMode, y regionMask, z hidden free-sphere kinds, w highlightActive
  sel: vec4u,             // x selected id, y selected type+1, z shaded(1), w organelle mode
  misc: vec4f,            // x QSTEP, y nucleic bead radius, z outline strength, w dpr
  cell: vec4f,            // x half edge of the sub-volume cube (Jurkat; yeast: R outer), y, z unused, w near
  caps: vec4u,            // x capL0 (offset of L1 items in list01), y motion amount x 1000, z motion cap nm x 1000,
                          // w membranes-only view (whole-cell surfaces, palette 66 + membrane, no cut)
};
@group(0) @binding(0) var<uniform> u: U;

fn sPlaneW(p: vec3f) -> f32 { return dot(u.plane.xyz, p + u.eye.xyz) - u.plane.w; }   // p camera-relative

fn hash32(x0: u32) -> u32 {
  var x = x0; x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u; return x;
}
fn unpackColor(c: u32) -> vec3f {
  return vec3f(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u)) / 255.0;
}
fn qrot(q: vec4f, v: vec3f) -> vec3f {   // q = (x,y,z,w)
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
fn sphereInFrustum(c: vec3f, r: f32) -> bool {
  for (var k = 0; k < 6; k++) { if (dot(u.frustum[k].xyz, c) + u.frustum[k].w < -r) { return false; } }
  return true;
}
fn i16f(x: u32) -> f32 { return f32(bitcast<i32>(x << 16u) >> 16u) / 32767.0 * INV_SQRT2; }
`;

const INST = /* wgsl */ `
struct TypeInfo {
  beadOff: u32, beadCount: u32, clusterOff: u32, clusterCount: u32,
  stOff: u32, stCount: u32, br: f32, req: f32,
  color0: u32, color1: u32, flags: u32, plddt: f32,
  modeOff: u32, coefOff: u32, nConf: u32,             // pool-ensemble motion modes (nConf 0: none)
  modColor: u32,                                       // packed colour of the STRING interaction module (0: none)
};
struct Inst { t: u32, region: u32, flags: u32, p: vec3f, q: vec4f, seed: u32 };
@group(0) @binding(1) var<storage, read> inst: array<vec4u>;
@group(0) @binding(2) var<storage, read> pageOrg: array<vec4f>;       // xyz origin (camera-relative is computed), w = bitcast count
@group(0) @binding(3) var<storage, read> types: array<TypeInfo>;
@group(0) @binding(4) var<storage, read> typeFlags: array<u32>;       // bit0 highlight, bit1 hidden, bit2 own class, bit3 no motion

fn decodeInst(i: u32) -> Inst {
  let w = inst[i];
  let org = pageOrg[i / PAGE].xyz;
  var o: Inst;
  o.t = w.x & 0x3FFFu;
  let qi = (w.x >> 14u) & 3u;
  o.region = (w.x >> 16u) & 31u;
  o.flags = w.x >> 21u;
  o.p = org + vec3f(f32(w.y & 0xFFFFu), f32(w.y >> 16u), f32(w.z & 0xFFFFu)) * u.misc.x - u.eye.xyz;
  let a = i16f(w.z >> 16u); let b = i16f(w.w & 0xFFFFu); let c = i16f(w.w >> 16u);
  let big = sqrt(max(0.0, 1.0 - a * a - b * b - c * c));
  // stored order (w,x,y,z) with component qi replaced by 'big'
  var q4: vec4f;
  if (qi == 0u) { q4 = vec4f(big, a, b, c); }
  else if (qi == 1u) { q4 = vec4f(a, big, b, c); }
  else if (qi == 2u) { q4 = vec4f(a, b, big, c); }
  else { q4 = vec4f(a, b, c, big); }
  o.q = vec4f(q4.y, q4.z, q4.w, q4.x);    // -> (x,y,z,w)
  o.seed = hash32(w.y ^ hash32(w.z ^ hash32(w.w)));   // stable per instance (tile-relative pose bits)
  return o;
}

// ---- illustrative motion (u.caps.y = amount x 1000; time u.cut.w in s). Not physics: smooth pseudo-random
// rigid-body wobble per instance plus flexing of low-confidence regions, slowed to about 1 Hz.
fn rnd(s: u32, k: u32) -> f32 { return f32(hash32(s ^ (k * 0x9E3779B9u)) & 0xFFFFu) / 65535.0; }
fn wob(s: u32, k: u32, t: f32) -> f32 {          // smooth signal in [-1, 1]
  return 0.6 * sin(t * (0.5 + rnd(s, k)) + 6.2832 * rnd(s, k + 101u))
       + 0.4 * sin(t * (1.1 + 1.3 * rnd(s, k + 211u)) + 6.2832 * rnd(s, k + 307u));
}
fn qmulq(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}
fn animInst(I: Inst, br: f32) -> Inst {
  if (u.caps.y == 0u || (typeFlags[I.t] & 8u) != 0u) { return I; }
  var o = I;
  let a = f32(u.caps.y) / 1000.0; let t = u.cut.w; let s = I.seed;
  let sz = max(br, 2.0);
  var dp = a * 0.35 * sqrt(2.0 / sz) * vec3f(wob(s, 1u, t), wob(s, 2u, t), wob(s, 3u, t));   // nm; larger = slower
  var ax = normalize(vec3f(wob(s, 4u, 0.3 * t), wob(s, 5u, 0.3 * t), wob(s, 6u, 0.3 * t)) + vec3f(1e-3));
  if (I.region >= 7u && I.region <= 13u) {        // membrane protein (Jurkat regions mem_PM .. mem_ENDO): stay in the bilayer
    let nrm = qrot(I.q, vec3f(0.0, 0.0, 1.0));
    dp -= nrm * dot(dp, nrm); ax = nrm;
  }
  let h = 0.5 * a * 0.35 / sz * wob(s, 7u, 0.8 * t);  // half-angle: the rim moves about 0.35 nm
  o.p = I.p + dp;
  o.q = qmulq(vec4f(ax * sin(h), cos(h)), I.q);
  return o;
}
// per-bead flexing in the molecule frame, spatially coherent (~5 nm wavelength); amplitude from pLDDT
fn flexBead(b: vec3f, plddt: f32, s: u32) -> vec3f {
  if (u.caps.y == 0u) { return b; }
  var amp = 0.04;
  if (plddt < 254.5) { amp = mix(0.45, 0.04, clamp((plddt - 50.0) / 40.0, 0.0, 1.0)); }
  let t = u.cut.w * 0.9; let k = 1.25;
  let d = vec3f(sin(dot(b, vec3f(k, 0.4, -0.3)) + t * (0.8 + rnd(s, 11u)) + 6.2832 * rnd(s, 12u)),
                sin(dot(b, vec3f(-0.3, k, 0.5)) + t * (0.7 + rnd(s, 13u)) + 6.2832 * rnd(s, 14u)),
                sin(dot(b, vec3f(0.5, -0.2, k)) + t * (0.9 + rnd(s, 15u)) + 6.2832 * rnd(s, 16u)));
  return b + (f32(u.caps.y) / 1000.0) * amp * d;
}
`;

// ---------------------------------------------------------------- culling / LOD compute
export const CULL = COMMON + INST + /* wgsl */ `
struct Counters { l0: atomic<u32>, l1: atomic<u32>, l2: atomic<u32>, lip: atomic<u32>, fre: atomic<u32>, ninst: atomic<u32>, p0: u32, p1: u32 };
struct CullParams { numPages: u32, strideX: u32, capL0: u32, capL1: u32, capL2: u32, capLip: u32, capFree: u32, numSPages: u32 };
struct Cluster { c: vec4f, info: vec4u };   // centre xyz, radius; bead offset, count
@group(1) @binding(0) var<storage, read_write> counters: Counters;
@group(1) @binding(1) var<storage, read_write> list01: array<vec2u>;   // [0,capL0) L0 items, [capL0, capL0+capL1) L1 items
@group(1) @binding(2) var<storage, read_write> listL2: array<u32>;
@group(1) @binding(3) var<uniform> cp: CullParams;
@group(1) @binding(4) var<storage, read> clusters: array<Cluster>;

fn visibleClass(t: TypeInfo, region: u32, tf: u32) -> bool {
  if ((tf & 2u) != 0u) { return false; }
  // bit2: molecule classes with their own Show switch (ribosomes, cytoskeleton, ...) ignore the compartment filter
  if ((tf & 4u) == 0u && ((u.modes.y >> region) & 1u) == 0u) { return false; }
  return true;
}

@compute @workgroup_size(64)
fn cullInstances(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * cp.strideX;
  let page = i / PAGE;
  if (page >= cp.numPages) { return; }
  let cnt = bitcast<u32>(pageOrg[page].w);
  if ((i % PAGE) >= cnt) { return; }
  let I = decodeInst(i);
  let T = types[I.t];
  if (!visibleClass(T, I.region, typeFlags[I.t])) { return; }
  let br = T.br;
  if (u.cut.x > 0.5) {
    let s = sPlaneW(I.p);
    if (s < -br || s - br > u.cut.y) { return; }
  }
  if (!sphereInFrustum(I.p, br)) { return; }
  atomicAdd(&counters.ninst, 1u);
  let dist = max(length(I.p), 0.01);
  let pxPerNm = u.eye.w / dist;
  let beadPx = 2.0 * u.lod.w * pxPerNm;
  if (beadPx >= u.lod.x && T.beadCount > 0u) {
    for (var c = 0u; c < T.clusterCount; c++) {
      let CL = clusters[T.clusterOff + c];
      let cs = CL.c;
      let cc = I.p + qrot(I.q, cs.xyz);
      let r = cs.w;
      if (u.cut.x > 0.5) {
        let s = sPlaneW(cc);
        if (s < -r || s - r > u.cut.y) { continue; }
      }
      if (!sphereInFrustum(cc, r)) { continue; }
      let ci = CL.info;
      let k = atomicAdd(&counters.l0, 1u);
      if (k < cp.capL0) { list01[k] = vec2u(i, ci.x | ((ci.y - 1u) << OFF_BITS)); }
    }
  } else if (br * pxPerNm >= u.lod.y && T.stCount > 0u) {
    let n = T.stCount;
    var o = 0u;
    loop {
      if (o >= n) { break; }
      let m = min(16u, n - o);
      let k = atomicAdd(&counters.l1, 1u);
      if (k < cp.capL1) { list01[cp.capL0 + k] = vec2u(i, (T.stOff + o) | ((m - 1u) << OFF_BITS)); }
      o += 16u;
    }
  } else {
    let k = atomicAdd(&counters.l2, 1u);
    if (k < cp.capL2) { listL2[k] = i; }
  }
}
`;

// lipids and free spheres culling (separate bind group layout)
export const CULL_SPHERES = COMMON + /* wgsl */ `
struct Counters { l0: atomic<u32>, l1: atomic<u32>, l2: atomic<u32>, lip: atomic<u32>, fre: atomic<u32>, ninst: atomic<u32>, p0: u32, p1: u32 };
struct SParams { numPages: u32, strideX: u32, cap: u32, isLipid: u32 };
@group(1) @binding(0) var<storage, read_write> counters: Counters;
@group(1) @binding(1) var<storage, read> spool: array<vec4f>;
@group(1) @binding(2) var<storage, read> spages: array<vec4u>;      // x count, y tile slot, z visible flag
@group(1) @binding(3) var<storage, read_write> slist: array<u32>;
@group(1) @binding(4) var<uniform> sp: SParams;

fn freeRadius(kind: u32, w: u32) -> f32 {
  if (kind == 7u) { return f32((w >> 9u) & 1023u) * 0.1; }   // whole-cell overview stand-in: radius in the record
  if (kind == 0u) { return 0.6; }   // mRNA nucleotide
  if (kind == 1u) { return 1.0; }   // linker DNA (2 bp per bead)
  if (kind == 5u) { return 0.45; }  // disordered protein residue (FG repeats)
  return 0.6;                        // mannan, pre-rRNA
}

@compute @workgroup_size(64)
fn cullSpheres(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * sp.strideX;
  let page = i / SPAGE;
  if (page >= sp.numPages) { return; }
  let pg = spages[page];
  if (pg.z == 0u || (i % SPAGE) >= pg.x) { return; }
  let rec = spool[i];
  let p = rec.xyz - u.eye.xyz;
  var r = 2.6;
  if (sp.isLipid == 0u) {
    let kind = bitcast<u32>(rec.w) & 15u;
    if (kind == 15u) { return; }                 // stand-in hidden: its tile's molecules are loaded (tiles.ts Overview)
    if (((u.modes.z >> kind) & 1u) != 0u) { return; }
    r = freeRadius(kind, bitcast<u32>(rec.w));
  }
  if (u.cut.x > 0.5) {
    let s = sPlaneW(p);
    if (s < -r || s - r > u.cut.y) { return; }
  }
  if (!sphereInFrustum(p, r)) { return; }
  let px = r * u.eye.w / max(length(p), 0.01);
  if (px < u.lod.z) { return; }
  if (sp.isLipid == 1u) {
    let k = atomicAdd(&counters.lip, 1u);
    if (k < sp.cap) { slist[k] = i; }
  } else {
    let k = atomicAdd(&counters.fre, 1u);
    if (k < sp.cap) { slist[k] = i; }
  }
}
`;

export const FINALIZE = /* wgsl */ `
struct Counters { l0: u32, l1: u32, l2: u32, lip: u32, fre: u32, ninst: u32, p0: u32, p1: u32 };
struct Caps { capL0: u32, capL1: u32, capL2: u32, capLip: u32, capFree: u32, a: u32, b: u32, c: u32 };
@group(0) @binding(0) var<storage, read> counters: Counters;
@group(0) @binding(1) var<storage, read_write> args: array<u32>;
@group(0) @binding(2) var<uniform> caps: Caps;
@compute @workgroup_size(1)
fn main() {
  // drawIndirect args: vertexCount, instanceCount, firstVertex, firstInstance
  args[0] = 384u; args[1] = min(counters.l0, caps.capL0); args[2] = 0u; args[3] = 0u;
  args[4] = 96u;  args[5] = min(counters.l1, caps.capL1); args[6] = 0u; args[7] = 0u;
  args[8] = 6u;   args[9] = min(counters.l2, caps.capL2); args[10] = 0u; args[11] = 0u;
  args[12] = 18u; args[13] = min(counters.lip, caps.capLip); args[14] = 0u; args[15] = 0u;
  args[16] = 6u;  args[17] = min(counters.fre, caps.capFree); args[18] = 0u; args[19] = 0u;
}
`;

// ---------------------------------------------------------------- sphere impostor drawing
const IMPOSTOR = /* wgsl */ `
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) vpos: vec3f,
  @location(1) @interpolate(flat) center: vec3f,
  @location(2) @interpolate(flat) radius: f32,
  @location(3) @interpolate(flat) color: vec3f,
  @location(4) @interpolate(flat) id: u32,
};
struct FOut {
  @location(0) color: vec4f,
  @location(1) id: u32,
  @builtin(frag_depth) depth: f32,
};
const ID_INST: u32 = 0x40000000u;
const ID_LIPID: u32 = 0x80000000u;
const ID_FREE: u32 = 0xC0000000u;

fn emitQuad(cw: vec3f, r: f32, corner: u32, color: vec3f, id: u32) -> VOut {
  var o: VOut;
  let cv = (u.view * vec4f(cw, 1.0)).xyz;
  let d = length(cv);
  o.center = cv; o.radius = r; o.color = color; o.id = id;
  if (r <= 0.0 || d <= r * 1.02 || cv.z > -u.cell.w) {
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0); o.vpos = vec3f(0.0); return o;
  }
  let k = d / sqrt(d * d - r * r);
  var off = vec2f(-1.0, -1.0);
  if (corner == 1u || corner == 4u) { off = vec2f(1.0, -1.0); }
  else if (corner == 2u || corner == 3u) { off = vec2f(-1.0, 1.0); }
  else if (corner == 5u) { off = vec2f(1.0, 1.0); }
  // billboard perpendicular to the view ray through the centre
  let fwd = cv / d;
  var up0 = vec3f(0.0, 1.0, 0.0);
  if (abs(fwd.y) > 0.99) { up0 = vec3f(1.0, 0.0, 0.0); }
  let right = normalize(cross(fwd, up0));
  let up = cross(right, fwd);
  let vp = cv + (right * off.x + up * off.y) * r * k * 1.05;
  o.vpos = vp;
  o.pos = u.proj * vec4f(vp, 1.0);
  return o;
}

fn shadeFrag(in: VOut) -> FOut {
  var o: FOut;
  let rd = normalize(in.vpos);
  let b = dot(rd, in.center);
  let c = dot(in.center, in.center) - in.radius * in.radius;
  var h = b * b - c;
  if (h < 0.0) { discard; }
  h = sqrt(h);
  var t = b - h;
  let t1 = b + h;
  var nrm = (rd * t - in.center) / in.radius;
  var cap = false;
  if (u.cut.x > 0.5) {
    let dn = dot(u.planeV.xyz, rd);
    let s0 = u.planeV.w + t * dn;
    if (s0 < 0.0) {
      if (dn <= 0.0) { discard; }
      let tp = -u.planeV.w / dn;
      if (tp > t1) { discard; }
      t = tp; nrm = -u.planeV.xyz; cap = true;
    }
    if (u.planeV.w + t * dn > u.cut.y) { discard; }
  }
  if (t < u.cell.w) { discard; }
  let p = rd * t;
  let clip = u.proj * vec4f(p, 1.0);
  o.depth = clip.z / clip.w;
  var shade = 1.0;
  // spheres a few pixels wide show only their bright, front-facing centre: fade to the sphere's mean shade there,
  // so residue beads (L0) are not lighter than the larger spheres of the coarser levels at the LOD boundary
  let rpx = in.radius * u.eye.w / max(length(in.center), 1e-3);
  let sharp = smoothstep(0.75, 3.0, rpx);
  if (u.sel.z == 1u) {
    let l = normalize(vec3f(0.35, 0.45, 1.0));
    shade = 0.45 + 0.55 * max(dot(nrm, l), 0.0) + 0.12 * pow(max(nrm.z, 0.0), 8.0);
    shade = mix(0.80, shade, sharp);       // mean over the projected disc: n.l ~ 2/3 l.z, pow(n.z, 8) ~ 1/5
  } else {
    shade = mix(0.977, 0.93 + 0.07 * max(nrm.z, 0.0), sharp);
  }
  if (cap) { shade = 0.9; }
  o.color = vec4f(in.color * shade, 1.0);
  o.id = in.id;
  return o;
}
`;

const COLOR = /* wgsl */ `
@group(0) @binding(5) var<storage, read> palette: array<vec4f>;   // 0..31 region colours, 32.. special

fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3f {
  let k = vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0);
  let p = abs(fract(vec3f(h) + k) * 6.0 - 3.0);
  return v * mix(vec3f(1.0), clamp(p - 1.0, vec3f(0.0), vec3f(1.0)), s);
}
fn viridis(t0: f32) -> vec3f {
  let t = clamp(t0, 0.0, 1.0);
  let c0 = vec3f(0.2777, 0.0054, 0.3340); let c1 = vec3f(0.1050, 1.4046, 1.3846);
  let c2 = vec3f(-0.3308, 0.2148, 0.0950); let c3 = vec3f(-4.6342, -5.7991, -19.3324);
  let c4 = vec3f(6.2282, 14.1799, 56.6906); let c5 = vec3f(4.7763, -13.7451, -65.3530);
  let c6 = vec3f(-5.4355, 4.6459, 26.3124);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}
fn plddtColor(p: f32) -> vec3f {
  if (p > 90.0) { return vec3f(0.0, 0.325, 0.839); }
  if (p > 70.0) { return vec3f(0.396, 0.796, 0.953); }
  if (p > 50.0) { return vec3f(1.0, 0.859, 0.075); }
  return vec3f(1.0, 0.490, 0.271);
}
// kind: 0 protein bead, 1 nucleic-acid bead, 2 coarse (sphere tree / single sphere)
fn instColor(I: Inst, T: TypeInfo, kind: u32, plddt: f32, id: u32) -> vec3f {
  let mode = u.modes.x;
  let hj = f32(hash32(I.t * 2654435761u + 17u) & 1023u) / 1023.0;
  var c: vec3f;
  if (mode == 0u) {          // Goodsell compartment palette with per-type lightness jitter
    if (T.color0 != 0u) {
      c = unpackColor(T.color0);
      if (kind == 1u && T.color1 != 0u) { c = unpackColor(T.color1); }
    } else { c = palette[I.region].xyz; }
    c = c * (0.93 + 0.14 * hj);
  } else if (mode == 1u) {   // structure source
    if ((T.flags & 1u) != 0u) { c = vec3f(0.30, 0.55, 0.85); } else { c = vec3f(0.96, 0.66, 0.30); }
    if (kind == 1u) { c = c * 0.8; }
  } else if (mode == 2u) {   // pLDDT
    var p = plddt;
    if (kind == 2u) { p = T.plddt; }
    if (p > 100.5 || p < 0.0) { c = vec3f(0.72, 0.72, 0.72); } else { c = plddtColor(p); }
  } else if (mode == 3u) {   // abundance (log10 copies, 1 .. 1e6)
    c = viridis(log(max(f32(T.flags >> 8u), 1.0)) / log(10.0) / 6.0);
  } else if (mode == 4u) {   // random per type
    c = hsv2rgb(hj, 0.55, 0.92);
  } else if (mode == 6u) {   // STRING interaction module
    if (T.modColor != 0u) { c = unpackColor(T.modColor); } else { c = vec3f(0.82, 0.82, 0.80); }
    if (kind == 1u) { c = c * 0.8; }
  } else {                    // compartment only
    c = palette[I.region].xyz;
  }
  if (u.modes.w == 1u && (typeFlags[I.t] & 1u) == 0u) {
    let g = dot(c, vec3f(0.3, 0.59, 0.11));
    c = mix(vec3f(g), vec3f(0.93), 0.72);
  }
  if (u.sel.x == id && id != 0u) { c = mix(c, vec3f(1.0, 0.92, 0.2), 0.55); }
  return c;
}
`;

const DI_HEAD = COMMON + INST + IMPOSTOR + COLOR;
export const DRAW_L0 = DI_HEAD + /* wgsl */ `
@group(1) @binding(0) var<storage, read> beads: array<vec4f>;     // xyz, w = bitcast(kind | plddt<<8)
@group(1) @binding(1) var<storage, read> list01: array<vec2u>;
@group(1) @binding(2) var<storage, read> modes: array<u32>;      // per bead 4 x (int8 xyz), then f32 conformer paths
const N_MODES: u32 = 4u;
fn i8x(w: u32, sh: u32) -> f32 { return f32(bitcast<i32>(w << (24u - sh)) >> 24u); }
fn coefAt(T: TypeInfo, k: u32, j: u32) -> f32 { return bitcast<f32>(modes[T.coefOff + (k % T.nConf) * N_MODES + j]); }
// displacement of bead 'local' along the protein's pool-ensemble path: Catmull-Rom through the sampled conformers'
// mode coefficients, about 2 s per conformer, each instance at its own phase
fn modeDisp(T: TypeInfo, local: u32, s: u32) -> vec3f {
  let n = T.nConf;
  let ph = u.cut.w * 0.5 + rnd(s, 21u) * f32(n);
  let k = u32(floor(ph)) % n; let f = fract(ph);
  let f2 = f * f; let f3 = f2 * f;
  var d = vec3f(0.0);
  for (var j = 0u; j < N_MODES; j++) {
    let p0 = coefAt(T, k + n - 1u, j); let p1 = coefAt(T, k, j); let p2 = coefAt(T, k + 1u, j); let p3 = coefAt(T, k + 2u, j);
    let c = 0.5 * (2.0 * p1 + (p2 - p0) * f + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f2 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * f3);
    let w = modes[T.modeOff + local * N_MODES + j];
    d += c * vec3f(i8x(w, 0u), i8x(w, 8u), i8x(w, 16u));
  }
  let cap = f32(u.caps.z) / 1000.0;     // nm, largest excursion of one bead (modes.json "cap"; 0 = none)
  let L = length(d);
  if (cap > 0.0 && L > cap) { d *= cap / L; }
  return d * f32(u.caps.y) / 1000.0;
}
@vertex
fn vsL0(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let item = list01[ii];
  let j = vi / 6u; let corner = vi % 6u;
  let cnt = (item.y >> OFF_BITS) + 1u;
  if (j >= cnt) { var o: VOut; o.pos = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  let I0 = decodeInst(item.x); let T = types[I0.t];
  let I = animInst(I0, T.br);
  let b = beads[(item.y & OFF_MASK) + j];
  let bw = bitcast<u32>(b.w);
  let kind = bw & 255u;
  var r = u.lod.w;
  if (kind == 1u) { r = u.misc.y; }
  let id = ID_INST | item.x;
  let pl = f32((bw >> 8u) & 255u);
  let col = instColor(I, T, kind, pl, id);
  var bp = b.xyz;
  if (u.caps.y > 0u && T.nConf > 0u) { bp = b.xyz + modeDisp(T, (item.y & OFF_MASK) + j - T.beadOff, I.seed); }
  else if ((typeFlags[I.t] & 8u) == 0u) { bp = flexBead(b.xyz, select(255.0, pl, kind == 0u), I.seed); }
  return emitQuad(I.p + qrot(I.q, bp), r, corner, col, id);
}
@fragment
fn fsMain(in: VOut) -> FOut { return shadeFrag(in); }
`;
export const DRAW_L1 = DI_HEAD + /* wgsl */ `
@group(1) @binding(0) var<storage, read> stSpheres: array<vec4f>; // xyz r
@group(1) @binding(1) var<storage, read> list01: array<vec2u>;
@vertex
fn vsL1(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let item = list01[u.caps.x + ii];
  let j = vi / 6u; let corner = vi % 6u;
  let cnt = (item.y >> OFF_BITS) + 1u;
  if (j >= cnt) { var o: VOut; o.pos = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  let I0 = decodeInst(item.x); let T = types[I0.t];
  let I = animInst(I0, T.br);
  let s = stSpheres[(item.y & OFF_MASK) + j];
  let id = ID_INST | item.x;
  // sphere-tree spheres are padded for collision; shrink slightly for display
  return emitQuad(I.p + qrot(I.q, s.xyz), s.w * 0.9, corner, instColor(I, T, 2u, -1.0, id), id);
}
@fragment
fn fsMain(in: VOut) -> FOut { return shadeFrag(in); }
`;
export const DRAW_L2 = DI_HEAD + /* wgsl */ `
@group(1) @binding(0) var<storage, read> listL2: array<u32>;
@vertex
fn vsL2(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let i = listL2[ii];
  let I0 = decodeInst(i); let T = types[I0.t];
  let I = animInst(I0, T.br);
  let id = ID_INST | i;
  return emitQuad(I.p, T.req, vi, instColor(I, T, 2u, -1.0, id), id);
}
@fragment
fn fsMain(in: VOut) -> FOut { return shadeFrag(in); }
`;

export const DRAW_SPHERES = COMMON + IMPOSTOR + /* wgsl */ `
@group(1) @binding(0) var<storage, read> spool: array<vec4f>;
@group(1) @binding(1) var<storage, read> spages: array<vec4u>;
@group(1) @binding(2) var<storage, read> slist: array<u32>;
@group(0) @binding(5) var<storage, read> palette: array<vec4f>;

fn octDecode(e: vec2f) -> vec3f {
  var v = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  if (v.z < 0.0) { let xy = (1.0 - abs(v.yx)) * select(vec2f(-1.0), vec2f(1.0), v.xy >= vec2f(0.0)); v = vec3f(xy, v.z); }
  return normalize(v);
}

// lipid record: xyz = midsurface point, w = bits: oct normal (8+8), class (3) <<16, leaflet sign <<19, membrane (4) <<20
@vertex
fn vsLipid(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let i = slist[ii];
  let rec = spool[i];
  let w = bitcast<u32>(rec.w);
  let n0 = octDecode(vec2f(f32(w & 255u), f32((w >> 8u) & 255u)) / 127.5 - 1.0);
  let cls = (w >> 16u) & 7u;
  let sgn = select(-1.0, 1.0, ((w >> 19u) & 1u) == 1u);
  let mem = (w >> 20u) & 15u;
  let j = vi / 6u; let corner = vi % 6u;
  var off = 1.95; var r = 0.55;
  var col = palette[40u].xyz;                  // head
  if (cls == 5u) {                              // ergosterol: two beads, rigid ring system
    col = palette[42u].xyz;
    if (j == 0u) { off = 1.55; r = 0.45; } else if (j == 1u) { off = 0.75; r = 0.45; } else { r = 0.0; }
  } else {
    col = palette[32u + cls].xyz;
    if (j == 1u) { off = 1.25; r = 0.45; col = palette[41u].xyz; }
    else if (j == 2u) { off = 0.55; r = 0.45; col = palette[41u].xyz; }
  }
  let slot = spages[i / SPAGE].y;
  let id = ID_LIPID | (slot << 5u) | (mem << 1u) | select(0u, 1u, sgn > 0.0);
  let p = rec.xyz - u.eye.xyz + n0 * sgn * off;
  return emitQuad(p, r, corner, col, id);
}

// free sphere record: xyz, w = bits: kind (4) | chain (15) << 4
@vertex
fn vsFree(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let i = slist[ii];
  let rec = spool[i];
  let w = bitcast<u32>(rec.w);
  let kind = w & 15u;
  let chain = (w >> 4u) & 0x7FFFu;
  var r = 0.6;
  if (kind == 1u) { r = 1.0; } else if (kind == 2u) { r = 0.75; } else if (kind == 5u) { r = 0.45; }
  let hj = f32(hash32(chain * 747796405u + kind) & 255u) / 255.0;
  var col = palette[48u + min(kind, 6u)].xyz * (0.94 + 0.12 * hj);
  if (kind == 7u) {                              // overview stand-in: radius and instance region in the record
    r = f32((w >> 9u) & 1023u) * 0.1;
    let h2 = f32(hash32(bitcast<u32>(rec.x) ^ hash32(bitcast<u32>(rec.z))) & 255u) / 255.0;
    col = palette[(w >> 4u) & 31u].xyz * (0.9 + 0.2 * h2);
  }
  let slot = spages[i / SPAGE].y;
  let id = ID_FREE | ((slot & 0xFFFu) << 18u) | ((kind & 7u) << 15u) | chain;
  var p = rec.xyz;
  if (u.caps.y > 0u && kind != 7u) {     // illustrative motion: FG repeats and mRNA most, wall polysaccharides and DNA little
    var amps = array<f32, 6>(0.25, 0.08, 0.06, 0.25, 0.25, 0.5);
    let amp = amps[min(kind, 5u)] * f32(u.caps.y) / 1000.0;
    let t = u.cut.w; let ph = 6.2832 * f32(hash32(chain * 2654435761u + kind) & 1023u) / 1023.0;
    p += amp * vec3f(sin(dot(p, vec3f(0.9, 0.3, -0.2)) + 1.1 * t + ph), sin(dot(p, vec3f(-0.2, 0.9, 0.4)) + 0.9 * t + 2.0 * ph),
                     sin(dot(p, vec3f(0.3, -0.3, 0.9)) + 1.3 * t + 3.0 * ph));
  }
  return emitQuad(p - u.eye.xyz, r, vi, col, id);
}

@fragment
fn fsMain(in: VOut) -> FOut { return shadeFrag(in); }
`;

// ---------------------------------------------------------------- tracer dynamics overlay (dynamics.ts)
// sphere record (2 x vec4): world xyz, radius; rgb, id bits
export const DRAW_TRACERS = COMMON + IMPOSTOR + /* wgsl */ `
@group(1) @binding(0) var<storage, read> tsph: array<vec4f>;
@vertex
fn vsTracer(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let a = tsph[2u * ii]; let b = tsph[2u * ii + 1u];
  return emitQuad(a.xyz - u.eye.xyz, a.w, vi, b.xyz, bitcast<u32>(b.w));
}
@fragment
fn fsMain(in: VOut) -> FOut { return shadeFrag(in); }
`;
// screen-space line quads; segment record (3 x vec4): a xyz, width px; b xyz, clip to slab (1); rgb
export const DRAW_LINES = COMMON + /* wgsl */ `
@group(1) @binding(0) var<storage, read> tseg: array<vec4f>;
struct LOut { @builtin(position) pos: vec4f, @location(0) color: vec3f, @location(1) pv: vec3f, @location(2) @interpolate(flat) clip: f32 };
struct LF { @location(0) color: vec4f, @location(1) id: u32 };
@vertex
fn vsLine(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> LOut {
  let A = tseg[3u * ii]; let B = tseg[3u * ii + 1u];
  var o: LOut; o.color = tseg[3u * ii + 2u].xyz; o.clip = B.w;
  var a = (u.view * vec4f(A.xyz - u.eye.xyz, 1.0)).xyz;
  var b = (u.view * vec4f(B.xyz - u.eye.xyz, 1.0)).xyz;
  let zn = -u.cell.w * 1.01;
  if (a.z > zn && b.z > zn) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  if (a.z > zn) { a = mix(b, a, (zn - b.z) / (a.z - b.z)); }
  if (b.z > zn) { b = mix(a, b, (zn - a.z) / (b.z - a.z)); }
  let pa = u.proj * vec4f(a, 1.0); let pb = u.proj * vec4f(b, 1.0);
  var dir = (pb.xy / pb.w - pa.xy / pa.w) * u.vp.xy;
  if (dot(dir, dir) < 1e-8) { dir = vec2f(1.0, 0.0); }
  let nrm = normalize(vec2f(-dir.y, dir.x)) * A.w * u.misc.w * u.vp.zw;   // half width in NDC
  let c = vi % 6u;
  let useB = c == 1u || c == 4u || c == 5u;
  let side = select(-1.0, 1.0, c == 2u || c == 3u || c == 5u);
  let p = select(pa, pb, useB);
  o.pv = select(a, b, useB);
  o.pos = vec4f(p.xy + nrm * side * p.w, p.z, p.w);
  return o;
}
@fragment
fn fsLine(in: LOut) -> LF {
  if (in.clip > 0.5 && u.cut.x > 0.5) {
    let s = u.planeV.w + dot(u.planeV.xyz, in.pv);
    if (s < 0.0 || s > u.cut.y) { discard; }
  }
  var o: LF; o.color = vec4f(in.color, 1.0); o.id = 0u;
  return o;
}
`;

// ---------------------------------------------------------------- membrane meshes (per tile, or organelle overview)
export const DRAW_MESH = COMMON + /* wgsl */ `
@group(0) @binding(5) var<storage, read> palette: array<vec4f>;
struct MP { slot: u32, organelle: u32, offset0: f32, offset1: f32 };   // organelle: 0 tile, 1 organelle mode, 2 under lipids, 3 membranes only
@group(1) @binding(0) var<uniform> mp: MP;
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) prel: vec3f,
  @location(1) nrm: vec3f,
  @location(2) @interpolate(flat) mem: u32,
  @location(3) @interpolate(flat) leaf: u32,
};
@vertex
fn vsMesh(@location(0) p: vec4f, @location(1) n: vec4f, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  let mem = bitcast<u32>(p.w);
  let off = select(mp.offset0, mp.offset1, ii == 1u);    // two leaflets (Jurkat: no lipid-droplet monolayer)
  let pr = p.xyz + n.xyz * off - u.eye.xyz;
  o.prel = pr; o.nrm = n.xyz; o.mem = mem; o.leaf = ii;
  o.pos = u.proj * (u.view * vec4f(pr, 1.0));
  return o;
}
struct FOut { @location(0) color: vec4f, @location(1) id: u32 };
@fragment
fn fsMesh(in: VOut, @builtin(front_facing) ff: bool) -> FOut {
  var o: FOut;
  let outside = !ff;      // membrane triangles are wound inward (tiles3d marching cubes): the outside is the back face
  // membranes only: the plasma membrane is seen from inside (its outside dropped) so the organelles show through it
  if (mp.organelle == 3u && in.mem == 0u && outside) { discard; }
  if (u.cut.x > 0.5) {
    let s = sPlaneW(in.prel);
    if (s < 0.0 || (mp.organelle != 1u && s > u.cut.y)) { discard; }
  }
  let vn = normalize((u.view * vec4f(in.nrm, 0.0)).xyz);
  var c = palette[41u].xyz;
  if (mp.organelle == 1u) { c = palette[56u + in.mem].xyz; } else if (mp.organelle == 3u) { c = palette[66u + in.mem].xyz; }
  let l = abs(vn.z);
  var shade = 0.82 + 0.18 * l;
  if (mp.organelle == 1u || mp.organelle == 3u) { shade = 0.55 + 0.45 * l; if (!outside) { shade *= 0.8; } }
  o.color = vec4f(c * shade, 1.0);
  o.id = 0x80000000u | (mp.slot << 5u) | (in.mem << 1u) | in.leaf;
  return o;
}
`;

// ---------------------------------------------------------------- exact compartment fill (distance fields)
// Jurkat sub-volume: the six distance fields of jenc.py (fields.ts) and the label rule of cell3d.label_from.
// Rays are clipped to the sub-volume cube |x|,|y|,|z| <= u.cell.x (the faces of the cube behave like cut faces).
export const FILL = COMMON + /* wgsl */ `
@group(0) @binding(5) var<storage, read> palette: array<vec4f>;
struct Prim { q0: vec4f, q1: vec4f, tc: vec4u };      // (yeast primitives; empty for Jurkat)
// G0.w: open face on (model v5), G1.x: z of the open face, G1.y: offset of the cristae header in fields (0 = none)
struct Geo { G0: vec4f, G1: vec4f, G2: vec4f, consts: vec4f, grid: vec4f,       // consts: PORE_R, PERI_W, IMS_W, nPores
             nax: vec4f, nr0: vec4f, nr1: vec4f, nr2: vec4f };
@group(1) @binding(0) var<storage, read> prims: array<Prim>;
@group(1) @binding(1) var<storage, read> pores: array<vec4f>;      // per pore: centre (perinuclear midsurface), NE normal
@group(1) @binding(2) var<storage, read> gridOff: array<u32>;   // prim list offsets (n^3+1) then pore list offsets
@group(1) @binding(3) var<storage, read> gridIdx: array<u32>;
@group(1) @binding(4) var<uniform> geo: Geo;
@group(1) @binding(5) var<storage, read> tileState: array<u32>;   // 128^3 (data.TGRID), bit0 molecules loaded, bit1 stand-ins
@group(1) @binding(6) var<storage, read> fields: array<f32>;       // distance fields (fields.ts layout)

// membranes (data.MEM_NAMES / jenc face_mem order): 0 PM, 1 NE, 2 ER, 3 MITO_OM, 4 MITO_IM, 5 GOLGI, 6 ENDO
const NMEM: u32 = 7u;
// fields: 0 cell, 1 ne_mid, 2 mito, 3 er, 4 golgi, 5 endo, 6 nucleolus (v5) (all negative inside)
struct Mem { d: array<f32, 7>, matrix: f32, hp: f32, cell: u32 };
// distance field k (cell3d.field_sdf): trilinear inside a stored brick, +-clamp elsewhere
fn fieldSdf(p: vec3f, k: u32) -> f32 {
  if (fields[0] < f32(k) + 0.5) { return 1e9; }
  let o = 1u + 12u * k; let h = fields[o + 3u]; let B = u32(fields[o + 7u]);
  let f = (p - vec3f(fields[o], fields[o + 1u], fields[o + 2u])) / h;
  let nb = vec3u(u32(fields[o + 4u]), u32(fields[o + 5u]), u32(fields[o + 6u]));
  if (any(f < vec3f(0.0)) || any(f >= vec3f(nb * B))) { return fields[o + 9u]; }
  let ii = vec3u(floor(f)); let bb = ii / B;
  let bi = i32(fields[u32(fields[o + 10u]) + (bb.x * nb.y + bb.y) * nb.z + bb.z]);
  if (bi == -1) { return -fields[o + 8u]; }
  if (bi == -2) { return fields[o + 9u]; }
  let n1 = B + 1u; let sx = n1 * n1; let sy = n1;
  let t = f - vec3f(ii); let l = ii - bb * B;
  let b = u32(fields[o + 11u]) + u32(bi) * n1 * n1 * n1 + (l.x * n1 + l.y) * n1 + l.z;
  let c00 = mix(fields[b], fields[b + sx], t.x); let c01 = mix(fields[b + 1u], fields[b + sx + 1u], t.x);
  let c10 = mix(fields[b + sy], fields[b + sx + sy], t.x); let c11 = mix(fields[b + sy + 1u], fields[b + sx + sy + 1u], t.x);
  return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}
// depth below the plasma membrane (negative outside): cell3d.outer_depth = -field(cell)
fn outerDepth(p: vec3f) -> f32 {
  if (fields[0] > 0.5) { return -fieldSdf(p, 0u); }
  return -1e9;
}
// height above the open face (model v5: the EM volume truncates the cell at z = G1.x); 1e9 without one
fn faceDist(p: vec3f) -> f32 {
  if (geo.G0.w > 0.5) { return p.z - geo.G1.x; }
  return 1e9;
}
// depth inside the modelled domain (cell3d out[M_OUTER]): the plasma membrane or the open face, whichever is nearer
fn domainDepth(p: vec3f) -> f32 { return min(outerDepth(p), faceDist(p)); }
// cell3d.crista_lumen: signed distance to the intracristal lumen (negative inside); b = boundary matrix SDF.
// header at o = G1.y (fields.ts CR_HDR): n, origin, h, dims, B, bricks, params/index/data offsets, CR_W, CR_GAP, CR_JW
fn cristaLumen(p: vec3f, b: f32) -> f32 {
  let o = u32(geo.G1.y);
  let c = vec3i(floor((p - vec3f(fields[o + 1u], fields[o + 2u], fields[o + 3u])) / fields[o + 4u]));
  let dims = vec3i(i32(fields[o + 5u]), i32(fields[o + 6u]), i32(fields[o + 7u]));
  if (any(c < vec3i(0)) || any(c >= dims)) { return 1e9; }
  let B = i32(fields[o + 8u]); let nb = vec3i(i32(fields[o + 9u]), i32(fields[o + 10u]), i32(fields[o + 11u]));
  let bb = c / B; let l = c - bb * B;
  let slot = i32(fields[u32(fields[o + 13u]) + u32((bb.x * nb.y + bb.y) * nb.z + bb.z)]);
  if (slot < 0) { return 1e9; }
  let k = i32(fields[u32(fields[o + 14u]) + u32(slot * B * B * B + (l.x * B + l.y) * B + l.z)]);
  if (k < 0) { return 1e9; }
  let q = u32(fields[o + 12u]) + 12u * u32(k);
  if (fields[q + 11u] == 0.0) { return 1e9; }
  let S = fields[q + 4u];
  let t = dot(vec3f(fields[q], fields[q + 1u], fields[q + 2u]), p) - fields[q + 3u];
  let uu = t - S * floor(t / S + 0.5);
  let slab = abs(uu) - fields[o + 15u];
  let lam = max(slab, b + fields[o + 16u]);
  let w = abs(dot(vec3f(fields[q + 5u], fields[q + 6u], fields[q + 7u]), p - vec3f(fields[q + 8u], fields[q + 9u], fields[q + 10u]))) - fields[o + 17u];
  return min(lam, max(slab, w));
}
// pore whose axis (through its centre along the NE normal) passes within rc of p: returns rho, or -1 (cell3d.in_pore)
fn poreRho(p: vec3f, cell: u32, rc: f32) -> f32 {
  let n3 = u32(geo.grid.x * geo.grid.x * geo.grid.x);
  let po0 = gridOff[n3 + 1u + cell]; let po1 = gridOff[n3 + 1u + cell + 1u];
  for (var k = po0; k < po1; k++) {
    let i = gridIdx[k];
    let v = p - pores[2u * i].xyz; let nn = pores[2u * i + 1u].xyz;
    let z = dot(v, nn);
    if (abs(z) > 150.0) { continue; }
    let rho = length(v - z * nn);
    if (rho < rc) { return rho; }
  }
  return -1.0;
}
fn gridCell(p: vec3f) -> u32 {
  let n = u32(geo.grid.x); let ext = geo.grid.y; let h = 2.0 * ext / f32(n);
  let g = clamp(vec3i(floor((p + vec3f(ext)) / h)), vec3i(0), vec3i(i32(n) - 1));
  return (u32(g.x) * n + u32(g.y)) * n + u32(g.z);
}
// cell3d.mem_sdfs (negative = lumenal / non-cytoplasmic side)
fn memSdfs(p: vec3f) -> Mem {
  var m: Mem;
  let PERI = geo.consts.y; let PORE_R = geo.consts.x; let IMS = geo.consts.z;
  m.d[0] = outerDepth(p);                       // PM
  let hp = fieldSdf(p, 1u);                     // 0 at the perinuclear midsurface, negative toward the nucleoplasm
  m.hp = hp;
  var d = abs(hp) - 0.5 * PERI;
  let rc = PORE_R + 0.5 * PERI;
  let cell = gridCell(p); m.cell = cell;
  if (abs(hp) < rc + PERI) {
    let rho = poreRho(p, cell, rc);
    if (rho >= 0.0) { d = sqrt((rho - rc) * (rho - rc) + hp * hp) - 0.5 * PERI; }   // pore-membrane torus
  }
  m.d[1] = d;                                   // NE (ONM and INM)
  let om = fieldSdf(p, 2u);
  m.d[3] = om;                                  // mito outer membrane
  m.matrix = om + IMS;
  if (geo.G1.y > 0.5 && m.matrix < fields[u32(geo.G1.y) + 15u]) {     // procedural cristae (model v4+)
    m.matrix = max(m.matrix, -cristaLumen(p, m.matrix));
  }
  m.d[4] = -m.matrix;                           // mito inner membrane
  m.d[2] = fieldSdf(p, 3u);                     // ER
  m.d[5] = fieldSdf(p, 4u);                     // Golgi
  m.d[6] = fieldSdf(p, 5u);                     // endo-lysosomal
  return m;
}
// labels (cell3d.label_from): 0 EXT, 1 CYTO, 2 PERI, 3 NUCP, 4 NUCL, 5 IMS, 6 MATRIX, 7 ERL, 8 GOLGI, 9 ENDO, 10 PORE,
// 11 OUTV (below the open face: outside the imaged EM volume)
fn labelOf(p: vec3f, m: Mem) -> u32 {
  let PERI = geo.consts.y; let PORE_R = geo.consts.x;
  if (faceDist(p) < 0.0) { return 11u; }
  if (m.d[0] < 0.0) { return 0u; }
  if (m.d[1] < 0.0) { return 2u; }
  let hp = m.hp;
  if (hp < -0.5 * PERI) {
    if (hp > -PERI - 8.0 && poreRho(p, m.cell, PORE_R + 0.5 * PERI) >= 0.0) { return 10u; }
    if (fieldSdf(p, 6u) < 0.0) { return 4u; }   // nucleolus (1e9 when the build has no nucleolus field)
    return 3u;
  }
  if (hp < 0.0) { return 10u; }                 // pore channel between the midplane and the ONM
  if (m.d[3] < 0.0) { if (m.matrix < 0.0) { return 6u; } return 5u; }
  if (m.d[2] < 0.0) { return 7u; }
  if (m.d[5] < 0.0) { return 8u; }
  if (m.d[6] < 0.0) { return 9u; }
  return 1u;
}
// bit0: molecules loaded, bit1: overview stand-ins shown (whole-cell mode); either pushes the cut-face backdrop back
fn tileLoaded(p: vec3f) -> bool {
  let g = vec3i(floor(p / 200.0)) + vec3i(64);
  if (any(g < vec3i(0)) || any(g > vec3i(127))) { return false; }
  return (tileState[(u32(g.x) * 128u + u32(g.y)) * 128u + u32(g.z)] & 3u) != 0u;
}

struct VO { @builtin(position) pos: vec4f, @location(0) ndc: vec2f };
@vertex
fn vsFill(@builtin(vertex_index) vi: u32) -> VO {
  var o: VO;
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
  o.pos = vec4f(xy, 0.0, 1.0); o.ndc = xy;
  return o;
}
struct FOut { @location(0) color: vec4f, @location(1) id: u32, @builtin(frag_depth) depth: f32 };

fn shadeLabel(p: vec3f, o: ptr<function, FOut>, bands: bool) {
  let m = memSdfs(p);
  let lab = labelOf(p, m);
  var c = palette[80u + lab].xyz;
  var id = 0x100u + lab;
  if (lab == 11u) { (*o).color = vec4f(palette[80u].xyz, 1.0); (*o).id = id; return; }   // outside the EM volume: not filled
  if (bands) {
    for (var k = 0u; k < NMEM; k++) {
      if (abs(m.d[k]) < 2.5) { c = palette[41u].xyz; id = 0x200u + k; }
    }
    // open face: a distinct grey line where the section meets the edge of the EM data (no membrane there), at least
    // 4 nm and about 2 px wide at any zoom (u.eye.w / distance = px per nm)
    let wf = max(4.0, 2.0 * length(p - u.eye.xyz) / u.eye.w);
    if (faceDist(p) < wf && lab != 0u) { c = palette[92u].xyz; id = 0x100u + 11u; }
  }
  (*o).color = vec4f(c, 1.0);
  (*o).id = id;
}

fn fillDepth(pv0: vec3f) -> f32 {
  let clip = u.proj * vec4f((u.view * vec4f(pv0, 1.0)).xyz, 1.0);
  return clamp(clip.z / clip.w, 0.0, 1.0);
}

@fragment
fn fsFill(in: VO) -> FOut {
  var o: FOut;
  let pw = u.invViewProj * vec4f(in.ndc, 1.0, 1.0);    // near plane point (camera-relative)
  let rd = normalize(pw.xyz / pw.w);
  let eye = u.eye.xyz;
  let H = u.cell.x;
  o.color = vec4f(palette[80u].xyz, 1.0); o.id = 0u; o.depth = 0.0;
  if (u.caps.w == 1u) { return o; }                   // membranes-only view (meshes only)
  // ray vs the sub-volume cube [-H, H]^3 (slab method)
  let rs = select(rd, vec3f(1e-9), abs(rd) < vec3f(1e-9));
  let ta = (vec3f(-H) - eye) / rs; let tb = (vec3f(H) - eye) / rs;
  let tmn = min(ta, tb); let tmx = max(ta, tb);
  let tn = max(max(tmn.x, tmn.y), tmn.z); let t1 = min(min(tmx.x, tmx.y), tmx.z);
  if (t1 <= max(tn, 0.0)) { return o; }               // the ray misses the cube (or it lies behind the camera)
  var t0 = max(tn, 0.0);
  var onFace = tn > 0.0;                              // the ray enters through a cube face (camera outside the cube)
  var onCut = false;
  var tEnd = t1;          // the trace ends here (where the ray leaves the cube or enters the removed half)
  if (u.cut.x > 0.5) {
    let dn = dot(u.plane.xyz, rd);
    let sE = u.planeV.w;
    if (abs(dn) < 1e-6) { if (sE < 0.0) { return o; } }
    else {
      let tp = -sE / dn;
      if (dn > 0.0) { if (tp > t1) { return o; } if (tp > t0) { t0 = tp; onCut = true; onFace = false; } }
      else { if (tp < t0) { return o; } tEnd = min(t1, tp); }
    }
    if (u.sel.w == 1u) {
      // organelle mode: cut face only (meshes provide the rest)
      if (!onCut) { o.color = vec4f(palette[80u].xyz, 1.0); return o; }
    }
  }
  if (u.sel.w == 1u && !onCut) { return o; }          // organelle mode without a cut: the meshes show everything
  var t = t0;
  var p = eye + rd * t;
  if ((onCut || onFace) && domainDepth(p) >= 0.0) {
    // a cut face or a face of the sub-volume cube inside the cell: compartment colours with membrane bands
    let inSlab = u.cut.x < 0.5 || dot(u.plane.xyz, p) - u.plane.w < u.cut.y;
    if (u.sel.w == 0u && tileLoaded(p) && inSlab) {
      // molecules are drawn here: show the compartment backdrop behind them
      if (onCut) { t = t + u.cut.y / max(dot(u.plane.xyz, rd), 0.05); }
      else { t = t + 40.0; }
      if (t >= tEnd) { return o; }
      p = eye + rd * t;
    }
    shadeLabel(p, &o, true);
    o.depth = fillDepth(rd * t);
    return o;
  }
  // exterior part of the cube: the plasma membrane, found by sphere tracing the (clamped) distance field; model v5:
  // the domain (cell above the open face), so the flat face where the EM volume truncates the cell is hit as well
  if (!onCut && !onFace && domainDepth(eye) > 0.0) { return o; }   // camera inside the cell and the cube, no cut
  var hit = false;
  for (var k = 0; k < 160; k++) {
    let d = -domainDepth(p);
    if (d < 0.25) { hit = true; break; }
    t = t + max(d, 0.5);
    if (t >= tEnd) { return o; }
    p = eye + rd * t;
  }
  if (!hit) { return o; }
  let ps = p;
  // one surface colour, lit like the molecules by the smoothed surface normal
  let e = 10.0;
  let g = vec3f(domainDepth(ps + vec3f(e, 0.0, 0.0)) - domainDepth(ps - vec3f(e, 0.0, 0.0)),
                domainDepth(ps + vec3f(0.0, e, 0.0)) - domainDepth(ps - vec3f(0.0, e, 0.0)),
                domainDepth(ps + vec3f(0.0, 0.0, e)) - domainDepth(ps - vec3f(0.0, 0.0, e)));
  var nn = -rd;
  if (dot(g, g) > 1e-6) { nn = -normalize(g); }
  let vn = (u.view * vec4f(nn, 0.0)).xyz;
  var shade = 0.93 + 0.07 * max(vn.z, 0.0);
  if (u.sel.z == 1u) {
    let l = normalize(vec3f(0.35, 0.45, 1.0));
    shade = 0.5 + 0.5 * max(dot(vn, l), 0.0) + 0.06 * pow(max(vn.z, 0.0), 8.0);
  }
  o.color = vec4f(palette[91u].xyz * shade, 1.0);
  o.id = 0x200u;            // plasma membrane
  if (faceDist(ps) < outerDepth(ps) + 0.5) {          // the open face (edge of the EM data, no membrane): flat grey
    o.color = vec4f(palette[92u].xyz * (0.9 + 0.1 * max(vn.z, 0.0)), 1.0);
    o.id = 0x100u + 11u;
  }
  o.depth = fillDepth(rd * t);
  return o;
}
`;

// ---------------------------------------------------------------- pick (id + depth of one pixel)
// WebGPU only copies whole depth subresources to buffers, so a single texel is read here instead.
export const PICK = /* wgsl */ `
@group(0) @binding(0) var idTex: texture_2d<u32>;
@group(0) @binding(1) var depthTex: texture_depth_2d;
@group(0) @binding(2) var<uniform> px: vec4u;
@group(0) @binding(3) var<storage, read_write> outp: array<u32, 4>;
@compute @workgroup_size(1)
fn main() {
  let p = vec2i(px.xy);
  outp[0] = textureLoad(idTex, p, 0).x;
  outp[1] = bitcast<u32>(textureLoad(depthTex, p, 0));
}
`;

// ---------------------------------------------------------------- composite (outlines, depth cueing)
export const COMPOSITE = COMMON + /* wgsl */ `
@group(1) @binding(0) var colorTex: texture_2d<f32>;
@group(1) @binding(1) var idTex: texture_2d<u32>;
@group(1) @binding(2) var depthTex: texture_depth_2d;
struct VO { @builtin(position) pos: vec4f };
@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> VO {
  var o: VO;
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
  o.pos = vec4f(xy, 0.0, 1.0);
  return o;
}
fn viewPos(px: vec2i, d: f32) -> vec3f {
  let ndc = vec2f((f32(px.x) + 0.5) * u.vp.z * 2.0 - 1.0, 1.0 - (f32(px.y) + 0.5) * u.vp.w * 2.0);
  let w = u.invViewProj * vec4f(ndc, max(d, 1e-7), 1.0);
  return w.xyz / w.w;
}
@fragment
fn fsComposite(in: VO) -> @location(0) vec4f {
  let px = vec2i(in.pos.xy);
  let dims = vec2i(textureDimensions(colorTex));
  var c = textureLoad(colorTex, px, 0).xyz;
  let id = textureLoad(idTex, px, 0).x;
  let d = textureLoad(depthTex, px, 0);
  let cat = id >> 30u;
  if (d <= 0.0) { return vec4f(c, 1.0); }
  let prel = viewPos(px, d);
  let dist = length(prel);
  let pxPerNm = u.eye.w / max(dist, 1e-3);
  // depth cue behind the cut face (Goodsell-style darkening with depth)
  if (u.cut.x > 0.5 && cat != 0u && u.sel.w == 0u) {
    let s = sPlaneW(prel);
    let f = smoothstep(0.0, u.cut.z, s);
    c = mix(c, c * vec3f(0.55, 0.56, 0.62), f * 0.8);
  }
  // outlines at molecule (id) boundaries; strength fades when molecules are only a few pixels wide
  let rad = i32(max(1.0, round(u.misc.w)));
  var edge = 0.0;
  var selEdge = 0.0;
  let offs = array<vec2i, 4>(vec2i(rad, 0), vec2i(-rad, 0), vec2i(0, rad), vec2i(0, -rad));
  for (var k = 0; k < 4; k++) {
    let q = clamp(px + offs[k], vec2i(0), dims - vec2i(1));
    let id2 = textureLoad(idTex, q, 0).x;
    if (id2 != id) {
      let d2 = textureLoad(depthTex, q, 0);
      if (cat == 0u) { if ((id2 >> 30u) == 0u) { edge = 1.0; } }
      else if (d2 <= d * 1.0005 || (id2 >> 30u) == 0u) { edge = 1.0; }
      if (u.sel.x != 0u && (id2 == u.sel.x || id == u.sel.x)) { selEdge = 1.0; }
    }
  }
  var strength = u.misc.z;
  if (cat == 1u) { strength *= smoothstep(0.6, 2.5, pxPerNm); }
  else if (cat == 2u) { strength *= 0.55 * smoothstep(1.5, 6.0, pxPerNm); }
  else if (cat == 3u) { strength *= 0.7 * smoothstep(1.5, 5.0, pxPerNm); }
  else { strength *= 0.35; }
  c = mix(c, c * 0.18, edge * strength);
  if (selEdge > 0.0) { c = vec3f(1.0, 0.85, 0.1); }
  return vec4f(c, 1.0);
}
`;
