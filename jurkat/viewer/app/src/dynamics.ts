// Tracer dynamics overlay: Brownian-dynamics trajectories of inert probes in a frozen crowd box (jurkat/validate/dynamics.py).
// data/dynamics.json is fetched on first use only; each tracer file is uint16 (frames, n, 3), position = box_lo + q / quant * box_nm.
import { fetchJson, fetchBin, isAbsent } from './data';
import { Renderer, ID_FREE } from './renderer';
import { V3 } from './math';

interface DynTracer { name: string; radius_nm: number; file: string; n: number; frames: number; D_over_D0?: number; q?: Uint16Array }
interface DynSet { region: string; box_lo: V3; phi?: number; tracers: DynTracer[] }
interface DynData { frame_ns: number; box_nm: number; quant: number; sets: DynSet[] }

const COLS: V3[] = [[0.22, 0.9, 0.16], [0.93, 0.16, 0.8], [0.15, 0.7, 1.0], [1.0, 0.72, 0.08]];   // GFP green, IgG magenta, ...
const BG: V3 = [0.965, 0.945, 0.906];
const BOX_COL: V3 = [0.95, 0.5, 0.0];
const SPEEDS = [5, 15, 30, 60, 150];     // frames per second
const hex = (c: V3) => '#' + c.map(x => Math.round(x * 255).toString(16).padStart(2, '0')).join('');
const fmtT = (ns: number) => ns < 1000 ? `${ns.toFixed(0)} ns` : `${(ns / 1000).toFixed(ns < 1e4 ? 2 : 1)} µs`;
const $ = (id: string) => document.getElementById(id) as HTMLElement;
/** picking id: free-sphere category, kind 6, chain = tracer index * 4096 + particle */
export const ID_TRACER = (ID_FREE | (6 << 15)) >>> 0;

export class Dynamics {
  data: DynData | null = null; loading: Promise<DynData | null> | null = null;
  set = 0; f = 0; playing = false; visible = false; speed = 30; which = -1; trail = 50; token = 0;
  el = $('dynPanel');
  onRegion: (region: string) => void = (r) => { this.show(r); };   // region chosen in the panel (main: apply its bookmark)

  constructor(public R: Renderer, public onChange: () => void, public toast: (m: string) => void) {}

  load() {
    return (this.loading ??= isAbsent('dynamics.json').then(a => (a ? null : fetchJson('dynamics.json'))).then((j: DynData) => (this.data = j?.sets?.length ? j : null)).catch(() => null));
  }
  get cur() { return this.data!.sets[this.set]; }
  nFrames(S = this.cur) { return Math.min(...S.tracers.map(t => (t.q ? Math.floor(t.q.length / (3 * t.n)) : t.frames))); }

  async show(region?: string) {
    const tok = ++this.token;
    const D = await this.load();
    if (tok !== this.token) return;
    if (!D) { this.toast('No dynamics data in this build'); return; }
    let s = D.sets.findIndex(x => x.region === region);
    if (s < 0) s = Math.min(this.set, D.sets.length - 1);
    try {
      await Promise.all(D.sets[s].tracers.filter(t => !t.q).map(async t => { t.q = new Uint16Array(await fetchBin(t.file)); }));
    } catch (e: any) { if (tok === this.token) this.toast(`Could not load the tracer trajectories: ${e?.message || e}`); return; }
    if (tok !== this.token) return;
    if (s !== this.set || !this.visible) { this.f = 0; if (s !== this.set) this.which = -1; }
    this.set = s; this.visible = true; this.playing = true;
    this.renderPanel(); this.upload(); this.onChange();
  }
  hide() {
    this.token++;
    if (!this.visible) return;
    this.visible = false; this.playing = false; this.el.style.display = 'none';
    this.R.setTracers(new Float32Array(0), 0, new Float32Array(0), 0); this.onChange();
  }
  toggle(region?: string) { if (this.visible) this.hide(); else this.show(region); }
  setPlaying(on: boolean) { if (!this.visible) return; this.playing = on; this.syncPanel(); this.onChange(); }

  /** advance while playing; true when the overlay changed (caller sets dirty) */
  tick(dt: number) {
    if (!this.visible || !this.playing) return false;
    this.f += Math.min(dt, 100) / 1000 * this.speed;
    if (this.f > this.nFrames() - 1) this.f = 0;
    this.upload(); this.syncPanel();
    return true;
  }

  describe(chain: number) {
    const t = this.data?.sets[this.set]?.tracers[chain >> 12];
    if (!t) return { title: 'Tracer', sub: '' };
    return { title: `${t.name} tracer`, sub: `inert probe, r = ${t.radius_nm} nm, diffusing in the frozen ${this.cur.region} crowd${t.D_over_D0 != null ? ` (D/D₀ ${t.D_over_D0.toFixed(2)})` : ''}` };
  }

  upload() {
    const D = this.data!, S = this.cur, k = D.box_nm / D.quant, lo = S.box_lo, nF = this.nFrames();
    const f0 = Math.min(Math.floor(this.f), nF - 1), a = this.f - f0, f1 = Math.min(f0 + 1, nF - 1);
    const show = S.tracers.map((t, j) => [t, j] as const).filter(([t, j]) => t.q && (this.which < 0 || this.which === j));
    const L = Math.min(this.trail, f0);
    const nP = show.reduce((s, [t]) => s + t.n, 0);
    const sph = new Float32Array(nP * 8), su = new Uint32Array(sph.buffer);
    const seg = new Float32Array((12 + nP * (L + 1)) * 12);
    let nS = 0, nG = 0;
    const line = (ax: number, ay: number, az: number, bx: number, by: number, bz: number, w: number, clip: number, c: V3) => {
      seg.set([ax, ay, az, w, bx, by, bz, clip, c[0], c[1], c[2], 0], 12 * nG++);
    };
    for (const [t, j] of show) {
      const q = t.q!, n = t.n, col = COLS[j % COLS.length];
      const P = (fr: number, i: number, c: number) => lo[c] + q[(fr * n + i) * 3 + c] * k;
      for (let i = 0; i < n; i++) {
        const x = P(f0, i, 0) * (1 - a) + P(f1, i, 0) * a, y = P(f0, i, 1) * (1 - a) + P(f1, i, 1) * a, z = P(f0, i, 2) * (1 - a) + P(f1, i, 2) * a;
        sph.set([x, y, z, t.radius_nm, col[0], col[1], col[2]], 8 * nS);
        su[8 * nS + 7] = (ID_TRACER | (j << 12) | i) >>> 0; nS++;
        let px = x, py = y, pz = z;
        for (let fr = f0; fr >= f0 - L; fr--) {
          const qx = P(fr, i, 0), qy = P(fr, i, 1), qz = P(fr, i, 2);
          const e = Math.min(1, (this.f - fr) / Math.max(1, this.trail)) * 0.8;
          if (px !== qx || py !== qy || pz !== qz) line(px, py, pz, qx, qy, qz, 1.6, 1, col.map((v, m) => v + (BG[m] - v) * e) as V3);
          px = qx; py = qy; pz = qz;
        }
      }
    }
    const B = D.box_nm;
    for (let c = 0; c < 8; c++) for (const b of [1, 2, 4]) if (!(c & b)) {
      const p = (m: number) => [0, 1, 2].map(ax => lo[ax] + ((m >> ax) & 1) * B);
      const [u, v] = [p(c), p(c | b)];
      line(u[0], u[1], u[2], v[0], v[1], v[2], 1.5, 0, BOX_COL);
    }
    this.R.setTracers(sph, nS, seg, nG);
  }

  renderPanel() {
    const D = this.data!, S = this.cur, el = this.el, nF = this.nFrames();
    el.innerHTML = `<button class="close" id="dynClose" title="Hide (G)">×</button>
      <div class="dh">Tracer dynamics <span>· ${S.region}, ${D.box_nm} nm box${S.phi != null ? `, crowding φ ${S.phi.toFixed(2)}` : ''}</span></div>
      <div class="row"><button class="small" id="dynPlay" title="Play / pause (K)"></button><input type="range" id="dynFrame" min="0" max="${nF - 1}" step="1"><span class="val" id="dynT"></span></div>
      <div class="row"><label>Speed</label><select id="dynSpeed">${SPEEDS.map(v => `<option value="${v}">${v} frames/s · ${fmtT(v * D.frame_ns)} per s</option>`).join('')}</select></div>
      <div class="row"><label>Tracers</label><div class="seg" id="dynWhich" style="flex:1">${S.tracers.map((t, j) => `<button data-w="${j}">${t.name}</button>`).join('')}<button data-w="-1">${S.tracers.length > 2 ? 'All' : 'Both'}</button></div></div>
      ${D.sets.length > 1 ? `<div class="row"><label>Region</label><select id="dynSet">${D.sets.map((s, i) => `<option value="${i}">${s.region}</option>`).join('')}</select></div>` : ''}
      <div class="row"><label>Trails</label><select id="dynTrail"><option value="0">Off</option><option value="10">10 frames (${fmtT(10 * D.frame_ns)})</option><option value="50">50 frames (${fmtT(50 * D.frame_ns)})</option><option value="200">200 frames (${fmtT(200 * D.frame_ns)})</option></select></div>
      <div class="dl">${S.tracers.map((t, j) => `<div><span class="sw" style="background:${hex(COLS[j % COLS.length])}"></span>${t.name}</div><div>r ${t.radius_nm} nm</div><div>${t.n} tracers</div><div>D/D₀ ${t.D_over_D0 != null ? t.D_over_D0.toFixed(2) : '–'}</div>`).join('')}</div>
      <div class="sub">Brownian dynamics among frozen crowders, ${D.frame_ns} ns per frame. D/D₀: long-time diffusion relative to water. Spheres at the probe radius; orange: simulation box.</div>`;
    el.style.display = 'block';
    $('dynClose').onclick = () => this.hide();
    $('dynPlay').onclick = () => this.setPlaying(!this.playing);
    const fr = $('dynFrame') as HTMLInputElement;
    fr.oninput = () => { this.f = Number(fr.value); this.playing = false; this.upload(); this.syncPanel(); this.onChange(); };
    fr.onchange = () => fr.blur();
    const sp = $('dynSpeed') as HTMLSelectElement; sp.value = String(this.speed);
    sp.onchange = () => { this.speed = Number(sp.value); sp.blur(); };
    el.querySelectorAll<HTMLElement>('#dynWhich button').forEach(b => b.onclick = () => { this.which = Number(b.dataset.w); this.upload(); this.syncPanel(); this.onChange(); });
    const tr = $('dynTrail') as HTMLSelectElement; tr.value = String(this.trail);
    tr.onchange = () => { this.trail = Number(tr.value); tr.blur(); this.upload(); this.onChange(); };
    const ss = document.getElementById('dynSet') as HTMLSelectElement | null;
    if (ss) { ss.value = String(this.set); ss.onchange = () => { ss.blur(); this.onRegion(D.sets[Number(ss.value)].region); }; }
    this.syncPanel();
  }
  syncPanel() {
    if (!this.visible) return;
    const D = this.data!, f = Math.floor(this.f);
    $('dynPlay').textContent = this.playing ? '❚❚' : '▶';
    ($('dynFrame') as HTMLInputElement).value = String(f);
    $('dynT').textContent = `${f} · ${fmtT(this.f * D.frame_ns)}`;
    $('dynT').title = `frame ${f} of ${this.nFrames() - 1} (${D.frame_ns} ns per frame)`;
    this.el.querySelectorAll<HTMLElement>('#dynWhich button').forEach(b => b.classList.toggle('on', Number(b.dataset.w) === this.which));
  }
}
