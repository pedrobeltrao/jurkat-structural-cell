import { V3, add, sub, scale, cross, dot, norm, rotAxis, lerp3, easeInOut } from './math';

// Free-rotation orbit camera (a cell has no natural "up"). World units are nm.
export class Camera {
  target: V3 = [0, 0, 0];
  dist = 3000;          // Jurkat: starts on the 1.6 um sub-volume
  fwd: V3 = [0, 0, -1];
  up: V3 = [0, 1, 0];
  fovy = 35 * Math.PI / 180;
  minDist = 1.5;
  maxDist = 25000;      // the whole-cell organelle surfaces extend to about +-7 um
  anim: null | { t0: number; dur: number; a: CamState; b: CamState; onDone?: () => void; onStep?: (t: number) => void } = null;

  get right(): V3 { return norm(cross(this.fwd, this.up)); }
  get eye(): V3 { return sub(this.target, scale(this.fwd, this.dist)); }

  orthonormalize() {
    this.fwd = norm(this.fwd);
    const r = norm(cross(this.fwd, this.up));
    this.up = norm(cross(r, this.fwd));
  }
  orbit(dx: number, dy: number) {
    // dx, dy in radians; camera orbits the target
    const r = this.right;
    this.fwd = rotAxis(this.fwd, this.up, -dx);
    this.fwd = rotAxis(this.fwd, r, -dy);
    this.up = rotAxis(this.up, r, -dy);
    this.orthonormalize();
  }
  roll(a: number) { this.up = rotAxis(this.up, this.fwd, a); this.orthonormalize(); }
  pan(dxPx: number, dyPx: number, pxPerNmAtTarget: number) {
    const s = 1 / pxPerNmAtTarget;
    this.target = add(this.target, add(scale(this.right, -dxPx * s), scale(this.up, dyPx * s)));
  }
  zoom(f: number) { this.dist = Math.min(this.maxDist, Math.max(this.minDist, this.dist * f)); }
  fly(v: V3) { this.target = add(this.target, v); }

  pxPerNmAtUnit(heightPx: number) { return heightPx / (2 * Math.tan(this.fovy / 2)); }
  distForWidth(widthNm: number, aspect: number) {
    return widthNm / (2 * Math.tan(this.fovy / 2) * Math.min(aspect, 1.0));
  }
  near() { return Math.min(5, Math.max(0.02, this.dist * 0.004)); }

  viewRot(): Float32Array {
    const r = this.right, u = this.up, f = this.fwd, m = new Float32Array(16);
    m[0] = r[0]; m[4] = r[1]; m[8] = r[2];
    m[1] = u[0]; m[5] = u[1]; m[9] = u[2];
    m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2];
    m[15] = 1;
    return m;
  }
  proj(aspect: number): Float32Array {
    const f = 1 / Math.tan(this.fovy / 2), m = new Float32Array(16);
    m[0] = f / aspect; m[5] = f; m[11] = -1; m[14] = this.near();
    return m;
  }
  // camera-relative frustum planes (inside: dot(n,p)+w >= 0); last plane is a dummy "always inside"
  frustum(aspect: number): Float32Array {
    const ty = Math.tan(this.fovy / 2), tx = ty * aspect;
    const r = this.right, u = this.up, f = this.fwd;
    const P = (n: V3, w = 0) => { const k = norm(n); return [k[0], k[1], k[2], w]; };
    const out = [
      ...P(add(r, scale(f, tx))), ...P(add(scale(r, -1), scale(f, tx))),
      ...P(add(u, scale(f, ty))), ...P(add(scale(u, -1), scale(f, ty))),
      f[0], f[1], f[2], -this.near(),
      0, 0, 0, 1,
    ];
    return new Float32Array(out);
  }
  // world-space ray through a pixel (u,v in [0,1], v downwards)
  ray(u: number, v: number, aspect: number): V3 {
    const ty = Math.tan(this.fovy / 2), tx = ty * aspect;
    const x = (u * 2 - 1) * tx, y = (1 - v * 2) * ty;
    return norm(add(this.fwd, add(scale(this.right, x), scale(this.up, y))));
  }

  state(): CamState { return { target: [...this.target] as V3, dist: this.dist, fwd: [...this.fwd] as V3, up: [...this.up] as V3 }; }
  set(s: CamState) { this.target = [...s.target] as V3; this.dist = s.dist; this.fwd = [...s.fwd] as V3; this.up = [...s.up] as V3; this.orthonormalize(); }

  flyTo(b: CamState, dur = 1800, onDone?: () => void, onStep?: (t: number) => void) {
    this.anim = { t0: performance.now(), dur, a: this.state(), b, onDone, onStep };
  }
  step(now: number): boolean {
    if (!this.anim) return false;
    const A = this.anim;
    let t = Math.min(1, (now - A.t0) / A.dur);
    const e = easeInOut(t);
    // zoom out in the middle of long flights (log-distance arc)
    const la = Math.log(A.a.dist), lb = Math.log(A.b.dist);
    const span = Math.hypot(...sub(A.b.target, A.a.target));
    const peak = Math.log(Math.max(A.a.dist, A.b.dist, span * 1.2));
    const arc = Math.max(0, peak - Math.max(la, lb));
    const ld = la + (lb - la) * e + arc * Math.sin(Math.PI * e);
    this.target = lerp3(A.a.target, A.b.target, e);
    this.dist = Math.exp(ld);
    this.fwd = slerpDir(A.a.fwd, A.b.fwd, e);
    // transport 'up' then blend towards target up
    let upA = A.a.up;
    this.up = norm(lerp3(upA, A.b.up, e));
    if (Math.abs(dot(this.up, this.fwd)) > 0.98) this.up = norm(cross(cross(this.fwd, A.b.up), this.fwd));
    this.orthonormalize();
    A.onStep?.(e);
    if (t >= 1) { this.anim = null; A.onDone?.(); }
    return true;
  }
}

export interface CamState { target: V3; dist: number; fwd: V3; up: V3 }

function slerpDir(a: V3, b: V3, t: number): V3 {
  const d = Math.max(-1, Math.min(1, dot(a, b)));
  const om = Math.acos(d);
  if (om < 1e-4) return norm(lerp3(a, b, t));
  if (om > Math.PI - 1e-3) {   // antipodal: rotate around any perpendicular
    const p = norm(cross(a, Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
    return rotAxis(a, p, Math.PI * t);
  }
  const s = Math.sin(om);
  return norm(add(scale(a, Math.sin((1 - t) * om) / s), scale(b, Math.sin(t * om) / s)));
}
