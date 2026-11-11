"""Tier 1 of the microscopy-driven revision: annotation-level localisation comparison for every model gene.

Steps (conda_envs/vcell/bin/python jurkat/validate/loc_compare.py <step>):
  fetch      HPA XML for every model gene with HPA subcellular data (8 threads, polite pacing, resumable cache under
             data/reference/hpa_subcellular/xml/<GENE>_<ENSG>.xml.gz; failures -> logs/hpa_xml_fail.tsv). ~1 h.
  celllines  parse the ICC/IF cell lines per gene (regex on <cellLine ...>NAME</cellLine> inside <data> entries)
             -> data/derived/validation/hpa_celllines.tsv (+ per-gene image URLs, per-line locations)
  jurkat     download every HPA JURKAT image (composite + blue/red/green/yellow channels)
             -> data/reference/hpa_subcellular/jurkat_images/  (HPA = CC BY-SA 4.0: never into the public repo)
  model      per-gene model fractions: intended (inventory comp_weights x copies, complexes expanded, regions as
             jwc.step_counts / jpack_test SOL/TMR) and placed (pack v4 tiles job_* arrays + pre-pass)
             -> data/derived/validation/cache/loc_model.parquet
  compare    mapping table, per-source scores, cause classes, rules -> loc_compare.tsv, loc_mapping.tsv, loc_rules.md
  holdout    held-out test set (HPA JURKAT annotations, OpenCell, HPA md5 test half) -> loc_holdout_baseline.json
  report     loc_rules.md (clusters, top-50 single-protein disagreements, rule impact, hold-out baseline)
  all        model compare holdout report
Long runs: setsid nohup ... > logs/loc_compare_<step>.log 2>&1 &
"""
import os, sys, re, json, glob, gzip, time, threading
import numpy as np, pandas as pd

V = (__import__("os").environ.get("VCELL_ROOT", ".") + ""); D = f"{V}/data/derived"; W = f"{D}/wc"
OUT = f"{D}/validation"; CACHE = f"{OUT}/cache"; REF = f"{V}/data/reference"
HX = f"{REF}/hpa_subcellular/xml"; JI = f"{REF}/hpa_subcellular/jurkat_images"
NPROC = 8
# model version (v5: nucleolus region, shells, TM routing, open face). LOC_VERSION=v4 reproduces the v4 run (archived tiles
# must then be at wc/tiles). Outputs of v5 carry a _v5 suffix; v4 files keep their original names.
VER = os.environ.get("LOC_VERSION", "v5")
SFX = "" if VER == "v4" else f"_{VER}"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------- model gene list
def model_genes():
    """genes with copies > 0 in the model: ingredient gene tokens (histone families expanded to inventory members)"""
    I = pd.read_csv(os.environ.get("VAL_ING", f"{D}/ingredients_v0.tsv"), sep="\t")
    toks = {g for s in I.loc[I.copies > 0, "genes"].astype(str) for g in s.split(";")}
    V0 = pd.read_csv(f"{D}/inventory_v0.tsv", sep="\t")
    fam = {f: list(V0.gene[(V0.copies > 0) & V0.gene.astype(str).str.match(rf"^{f}(C|-)\d")]) for f in ("H2A", "H2B", "H3", "H4")}
    genes = set()
    for t in toks:
        genes |= set(fam[t]) if t in fam else {t}
    return sorted(genes), fam


def hpa_table():
    return pd.read_csv(f"{REF}/hpa_subcellular/subcellular_location.tsv", sep="\t").drop_duplicates("Gene name").set_index("Gene name")


# ----------------------------------------------------------------- step: fetch HPA XML
def fetch():
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(HX, exist_ok=True)
    genes, _ = model_genes(); hpa = hpa_table()
    todo = [(g, hpa.at[g, "Gene"]) for g in genes if g in hpa.index]
    todo = [(g, e) for g, e in todo if not os.path.exists(f"{HX}/{g}_{e}.xml.gz")]
    log(f"{len(genes)} model genes, {sum(g in hpa.index for g in genes)} with HPA subcellular data, {len(todo)} to fetch")
    lock = threading.Lock(); done = [0]; fails = []

    def one(ge):
        g, e = ge; f = f"{HX}/{g}_{e}.xml.gz"
        for k in range(4):
            try:
                req = urllib.request.Request(f"https://www.proteinatlas.org/{e}.xml",
                                             headers={"User-Agent": "virtualcell-jurkat-validation (academic; ETH Zurich)"})
                data = urllib.request.urlopen(req, timeout=120).read()
                tmp = f + ".part"; gzip.open(tmp, "wb").write(data); os.replace(tmp, f)
                break
            except Exception as ex:
                if k == 3:
                    with lock: fails.append((g, e, repr(ex)))
                time.sleep(5 * (k + 1))
        time.sleep(0.4)                                         # polite pacing per thread
        with lock:
            done[0] += 1
            if done[0] % 250 == 0: log(f"{done[0]}/{len(todo)} fetched, {len(fails)} failed")

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, todo))
    pd.DataFrame(fails, columns=["gene", "ensg", "error"]).to_csv(f"{V}/logs/hpa_xml_fail.tsv", sep="\t", index=False)
    log(f"done: {done[0]} attempted, {len(fails)} failed")


# ----------------------------------------------------------------- step: HPA ICC/IF cell lines per gene
RE_ICC = re.compile(r'<cellExpression source="HPA" technology="ICC/IF">(.*?)</cellExpression>', re.S)
RE_SUB = re.compile(r'<subAssay type="([^"]*)"[^>]*>(.*?)</subAssay>', re.S)
RE_DATA = re.compile(r"<data>(.*?)</data>", re.S)
RE_CL = re.compile(r'<cellLine organ="([^"]*)" cellosaurusID="([^"]*)">([^<]+)</cellLine>')
RE_LOC = re.compile(r"<location[^>]*>([^<]+)</location>")
RE_URL = re.compile(r"<imageUrl>([^<]+)</imageUrl>")
RE_VER = re.compile(r'<verification type="(?:validation|reliability)"[^>]*>([^<]+)</verification>')
RE_AB = re.compile(r'<antibody id="([^"]+)"')


def _parse_xml(f):
    g, e = os.path.basename(f)[:-7].rsplit("_", 1)
    txt = gzip.open(f).read().decode("utf-8", "replace")
    rows = []
    parts = re.split(r"(?=<antibody id=)", txt)
    for part in parts[1:]:
        ab = RE_AB.match(part).group(1)
        for icc in RE_ICC.findall(part):
            for typ, sub in RE_SUB.findall(icc):
                if typ != "human": continue
                ver = RE_VER.search(sub); ver = ver.group(1) if ver else ""
                for d in RE_DATA.findall(sub):
                    m = RE_CL.search(d)
                    if not m: continue
                    rows.append(dict(gene=g, ensg=e, antibody=ab, cell_line=m.group(3), organ=m.group(1), cellosaurus=m.group(2),
                                     ab_reliability=ver, locations=";".join(RE_LOC.findall(d)),
                                     image_urls=";".join(u for u in RE_URL.findall(d) if "blue_red_green" in u)))
    return rows


def celllines():
    from concurrent.futures import ProcessPoolExecutor
    genes, _ = model_genes(); gs = set(genes)
    fs = [f for f in sorted(glob.glob(f"{HX}/*.xml.gz")) if os.path.basename(f).rsplit("_", 1)[0] in gs]
    with ProcessPoolExecutor(NPROC) as ex:
        rows = [r for rr in ex.map(_parse_xml, fs, chunksize=64) for r in rr]
    C = pd.DataFrame(rows)
    C["n_images"] = C.image_urls.str.count(";") + (C.image_urls != "").astype(int)
    C.to_csv(f"{OUT}/hpa_celllines.tsv", sep="\t", index=False)
    log(f"{len(fs)} XML parsed; {C.gene.nunique()} genes with ICC/IF cell-line data; {len(C)} (gene, antibody, line) rows")
    cov = C.groupby("cell_line").agg(genes=("gene", "nunique"), antibodies=("antibody", "nunique"), images=("n_images", "sum"))
    cov = cov.sort_values("genes", ascending=False)
    cov.to_csv(f"{OUT}/hpa_cellline_coverage.tsv", sep="\t")
    log("cell-line coverage (genes):\n" + cov.to_string())
    j = C[C.cell_line.str.upper() == "JURKAT"]
    log(f"JURKAT: {j.gene.nunique()} genes, {j.antibody.nunique()} antibodies, {j.n_images.sum()} images")


def jurkat_images():
    """download all HPA JURKAT field images: composite + per-channel (blue nucleus, red microtubules, green target, yellow ER)"""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(JI, exist_ok=True)
    C = pd.read_csv(f"{OUT}/hpa_celllines.tsv", sep="\t", keep_default_na=False)
    J = C[C.cell_line.str.upper() == "JURKAT"]
    todo, man = [], []
    for r in J.itertuples():
        for u in [x for x in r.image_urls.split(";") if x]:
            base = u[: -len("blue_red_green.jpg")]
            for ch in ("blue_red_green", "blue", "red", "green", "yellow"):
                uu = f"{base}{ch}.jpg"; fn = f"{JI}/{r.gene}_{r.antibody}_{os.path.basename(uu)}"
                man.append(dict(gene=r.gene, antibody=r.antibody, locations=r.locations, ab_reliability=r.ab_reliability,
                                channel=ch, url=uu, file=os.path.basename(fn)))
                if not os.path.exists(fn): todo.append((uu, fn))
    log(f"{J.gene.nunique()} JURKAT genes, {len(man)} image files, {len(todo)} to download")
    fails = []

    def one(a):
        u, fn = a
        for k in range(3):
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "virtualcell-jurkat-validation (academic; ETH Zurich)"})
                data = urllib.request.urlopen(req, timeout=120).read(); open(fn + ".part", "wb").write(data); os.replace(fn + ".part", fn)
                break
            except Exception as ex:
                if k == 2: fails.append((u, repr(ex)))
                time.sleep(3 * (k + 1))
        time.sleep(0.3)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, todo))
    M = pd.DataFrame(man); M["present"] = [os.path.exists(f"{JI}/{f}") for f in M.file]
    M.to_csv(f"{JI}/manifest.tsv", sep="\t", index=False)
    open(f"{JI}/LICENSE_NOTE.txt", "w").write("Images from the Human Protein Atlas (proteinatlas.org), CC BY-SA 4.0. "
                                              "Analysis only; do not redistribute in the public repository.\n")
    log(f"downloaded; {len(fails)} failures; {M.present.sum()} of {len(M)} files present")


# ----------------------------------------------------------------- model regions and comparison compartments
RN = ["cytosol", "nucleoplasm", "matrix", "ER_lumen", "golgi_lumen", "endo_lumen", "perinuclear",
      "PM", "NE", "ER", "MITO_OM", "MITO_IM", "GOLGI", "ENDO"]                 # jpack_test RNAMES (tile job_region), v4
SHELLS = ["sh_PM_cyto", "sh_PM_exo", "sh_PM_cortex", "sh_NE_lamina", "sh_NE_nuc", "sh_ER_cyto", "sh_ER_lumen", "sh_GOLGI_cyto",
          "sh_GOLGI_lumen", "sh_ENDO_cyto", "sh_ENDO_lumen", "sh_OM_cyto"]
if VER == "v5":
    RN = RN[:7] + ["nucleolus"] + RN[7:] + SHELLS                         # v5 jpack_test.RNAMES
elif VER == "v7":                                                         # v7 (JWC_REGIONS=v7): + virtual volume regions, + sh_NPC_fg
    RN = (RN[:7] + ["nucleolus", "speckle", "mt_lattice", "mt_plus", "pcm", "vim_cage"] + RN[7:] + SHELLS
          + ["sh_NPC", "sh_PM_periph", "sh_ENDO_periph", "sh_GOLGI_periph", "sh_NPC_fg"])
elif VER != "v4":                                                         # v6: + speckle (volume) and the NPC-proximal shell
    RN = RN[:7] + ["nucleolus", "speckle"] + RN[7:] + SHELLS + ["sh_NPC", "sh_PM_periph", "sh_ENDO_periph", "sh_GOLGI_periph"]
COMPS = ["PM", "cytosol", "nucleus", "NE", "ER", "mito", "golgi", "endolysosomal"]
REG2COMP = {"cytosol": "cytosol", "nucleoplasm": "nucleus", "matrix": "mito", "ER_lumen": "ER", "golgi_lumen": "golgi",
            "endo_lumen": "endolysosomal", "perinuclear": "NE", "PM": "PM", "NE": "NE", "ER": "ER", "MITO_OM": "mito",
            "MITO_IM": "mito", "GOLGI": "golgi", "ENDO": "endolysosomal",
            # v5: nucleolus is nuclear; shells count as their membrane at light-microscopy resolution (cortex as PM, lamina as NE)
            "nucleolus": "nucleus", "sh_PM_cyto": "PM", "sh_PM_exo": "PM", "sh_PM_cortex": "PM", "sh_NE_lamina": "NE", "sh_NE_nuc": "NE",
            "sh_ER_cyto": "ER", "sh_ER_lumen": "ER", "sh_GOLGI_cyto": "golgi", "sh_GOLGI_lumen": "golgi", "sh_ENDO_cyto": "endolysosomal",
            "sh_ENDO_lumen": "endolysosomal", "sh_OM_cyto": "mito",
            # v6: nuclear speckles are nuclear; the NPC-proximal shell (both faces of the NE, near pores) reads as nuclear rim
            "speckle": "nucleus", "sh_NPC": "NE", "sh_PM_periph": "PM", "sh_ENDO_periph": "endolysosomal", "sh_GOLGI_periph": "golgi",
            # v7: virtual cytosol regions (MT bands, PCM, vimentin cage) read as cytosol; the NPC channel as nuclear rim
            "mt_lattice": "cytosol", "mt_plus": "cytosol", "pcm": "cytosol", "vim_cage": "cytosol", "sh_NPC_fg": "NE"}
SOLUBLE = {"cytosol", "nucleoplasm", "matrix", "ER_lumen", "golgi_lumen", "endo_lumen", "perinuclear", "nucleolus", "speckle",
           "mt_lattice", "mt_plus", "pcm", "vim_cage"}
# jpack_test.SOL / TMR (copied, not imported: analysis must not depend on pipeline import side effects)
SOL = {"cytosol": "cytosol", "cytoskeleton": "cytosol", "peroxisome": "cytosol", "lipid_droplet": "cytosol", "unassigned": "cytosol",
       "plasma_membrane": "cytosol", "nucleus": "nucleoplasm", "nuclear_envelope": "nucleoplasm", "mitochondrion": "matrix",
       "er": "ER_lumen", "secreted": "ER_lumen", "golgi": "golgi_lumen", "endolysosomal": "endo_lumen",
       "nucleolus": "nucleolus", "pm_anchor_cyto": "sh_PM_cyto", "pm_anchor_exo": "sh_PM_exo", "cortex": "sh_PM_cortex",
       "lamina": "sh_NE_lamina", "ne_anchor_nuc": "sh_NE_nuc", "er_anchor_cyto": "sh_ER_cyto", "er_anchor_lumen": "sh_ER_lumen",
       "golgi_anchor_cyto": "sh_GOLGI_cyto", "golgi_anchor_lumen": "sh_GOLGI_lumen", "endo_anchor_cyto": "sh_ENDO_cyto",
       "endo_anchor_lumen": "sh_ENDO_lumen", "om_anchor_cyto": "sh_OM_cyto", "speckle": "speckle", "npc": "sh_NPC",
       "pm_periph_cyto": "sh_PM_periph", "endo_periph_cyto": "sh_ENDO_periph", "golgi_periph_cyto": "sh_GOLGI_periph",
       "mt_lattice": "mt_lattice", "mt_plus": "mt_plus", "pcm": "pcm", "vim_cage": "vim_cage", "npc_fg": "sh_NPC_fg"}   # v7 keys
TMR = {"plasma_membrane": "PM", "nucleus": "NE", "nuclear_envelope": "NE", "mitochondrion": "MITO_IM", "er": "ER", "golgi": "GOLGI",
       "endolysosomal": "ENDO", "cytosol": "ER", "cytoskeleton": "ER", "peroxisome": "ER", "lipid_droplet": "ER", "unassigned": "ER", "secreted": "ER",
       "nucleolus": "NE", "pm_anchor_cyto": "PM", "pm_anchor_exo": "PM", "cortex": "PM", "lamina": "NE", "ne_anchor_nuc": "NE",
       "er_anchor_cyto": "ER", "er_anchor_lumen": "ER", "golgi_anchor_cyto": "GOLGI", "golgi_anchor_lumen": "GOLGI",
       "endo_anchor_cyto": "ENDO", "endo_anchor_lumen": "ENDO", "om_anchor_cyto": "MITO_IM", "speckle": "NE", "npc": "NE",
       "pm_periph_cyto": "PM", "endo_periph_cyto": "ENDO", "golgi_periph_cyto": "GOLGI",
       "mt_lattice": "ER", "mt_plus": "ER", "pcm": "ER", "vim_cage": "ER", "npc_fg": "NE"}
PM_REGS = {"PM", "sh_PM_cyto", "sh_PM_exo", "sh_PM_cortex", "sh_PM_periph"}
PRE_ING = {"NUCLEOSOME": "NUCLEOSOME", "NPC": "NPC", "RIBO80S": "RIBO80S", "MT_TUBA1B-TUBB4B": "TUBA1B-TUBB4B", "F_ACTB": "ACTB"}


def pre_region(lib):
    """pre-pass region. Accounting fix (v6 validation, applied to v5 too): F-actin subunits are all cortical (jcytoskeleton:
    8-150 nm below the PM), so they count as the cortex shell (-> PM at light resolution, as the packed cortex); MT tubulin
    stays cytosol (HPA 'Microtubules' maps to cytosol)."""
    if VER != "v4" and lib == "F_ACTB": return "sh_PM_cortex"
    if VER not in ("v4", "v5", "v6"):                           # v7: lamina lattice, chromatin functional layer (v8 chromatin)
        if lib in ("LMNB1", "LMNB2", "LMNA"): return "sh_NE_lamina"
        if lib in ("CPX-2387", "CTCF", "CPX-5989", "CPX-5991"): return "nucleoplasm"
    return "nucleoplasm" if (lib == "NUCLEOSOME" or lib.startswith("H1-")) else "NE" if lib == "NPC" else "cytosol"


_LIBS = None


def _tile_sums(files):
    acc = {}
    for f in files:
        z = np.load(f)
        jl, jr, jn, jp = z["job_lib"], z["job_region"], z["job_n"], z["job_placed"]
        k = jl.astype(np.int64) * 64 + jr
        u, inv = np.unique(k, return_inverse=True)
        n = np.bincount(inv, jn); p = np.bincount(inv, jp)
        for a, b, c in zip(u, n, p):
            x = acc.get(a); acc[a] = (x[0] + b, x[1] + c) if x else (b, c)
    return acc


def _memb_one(i):
    sys.path.insert(0, f"{V}/jurkat/yc/pipeline")
    from config import LIB
    f = f"{LIB}/{i}.npz"
    return ("membrane" in np.load(f).files) if os.path.exists(f) else False


def pack_arrays():
    """cached: per (library, region) requested/placed instances over all v4 tiles; membrane flag per ingredient"""
    import multiprocessing as mp
    cf = f"{CACHE}/loc_pack_{VER}.npz"
    if os.path.exists(cf):
        z = np.load(cf, allow_pickle=True); return z["req"], z["pla"], dict(zip(z["ids"], z["memb"]))
    os.makedirs(CACHE, exist_ok=True)
    I = pd.read_csv(os.environ.get("VAL_ING", f"{D}/ingredients_v0.tsv"), sep="\t"); libs = json.load(open(f"{W}/libs.json"))["libs"]
    fs = sorted(glob.glob(f"{W}/tiles/*.npz")); chunks = [fs[i::64] for i in range(64)]
    req = np.zeros((len(libs), len(RN))); pla = np.zeros((len(libs), len(RN)))
    with mp.get_context("fork").Pool(7) as pool:
        ids = list(I.ing_id); memb = pool.map(_memb_one, ids, chunksize=200)
        log(f"membrane flags read for {len(ids)} ingredients")
        for k, acc in enumerate(pool.imap_unordered(_tile_sums, chunks)):
            for key, (n, p) in acc.items():
                req[key // 64, key % 64] += n; pla[key // 64, key % 64] += p
            if k % 16 == 0: log(f"tile chunk {k + 1}/64")
    np.savez(cf, req=req, pla=pla, ids=np.array(ids), memb=np.array(memb))
    log(f"tiles read: requested {req.sum():,.0f}, placed {pla.sum():,.0f}")
    return req, pla, dict(zip(ids, memb))


def model():
    """per-gene intended and placed copies per (source comp_weights key, model region)"""
    I = pd.read_csv(os.environ.get("VAL_ING", f"{D}/ingredients_v0.tsv"), sep="\t")
    V0 = pd.read_csv(f"{D}/inventory_v0.tsv", sep="\t").set_index("gene")
    L = json.load(open(f"{W}/libs.json")); libs = L["libs"]; consumed = L["consumed"]
    f_vol, f_pm = L.get("f_vol", 1.0), L.get("f_pm", 1.0)          # v5 open-face scaling (jwc.step_counts)
    req, pla, memb = pack_arrays()
    lidx = {l: i for i, l in enumerate(libs)}
    PL = json.load(open(f"{W}/prepass_libs.json"))
    pc = np.bincount(np.load(f"{W}/prepass_types.npy", mmap_mode="r"), minlength=len(PL))
    pre = {}                                                    # ing_id -> {region: placed copies}
    for l, c in zip(PL, pc):
        ing = PRE_ING.get(l, l); pre.setdefault(ing, {}); pre[ing][pre_region(l)] = pre[ing].get(pre_region(l), 0) + float(c)
    genes, fam = model_genes()
    famw = {f: (V0.loc[m, "copies"] / V0.loc[m, "copies"].sum()) for f, m in fam.items()}
    rows = []
    for r in I.itertuples():
        if r.copies <= 0: continue
        cw = json.loads(r.comp_weights) if isinstance(r.comp_weights, str) else {}
        if not cw: cw = {"unassigned": 1.0}
        cons = float(consumed.get(r.ing_id, 0)); rest = max(r.copies - cons, 0.0)
        if VER not in ("v4", "v5", "v6") and r.ing_id in ("LMNB1", "LMNB2", "LMNA") and cons > 0 and rest > 0 and "lamina" in cw:
            # jwc consumed_key (v7): the lattice consumption comes out of the lamina key only
            C_ = float(r.copies); cw = {k: (C_ * v / rest if k != "lamina" else max(C_ * v - cons, 0.0) / rest) for k, v in cw.items()}
        tm = bool(r.transmembrane) and memb.get(r.ing_id, False)
        ent = {}                                                # (src comp, region) -> [intended, placed]
        li = lidx.get(r.ing_id)
        for comp, wt in cw.items():
            if tm:
                reg = TMR.get(comp, "ER")
                if reg == "MITO_IM" and r.mito_om_frac >= 0.5: reg = "MITO_OM"
            else:
                reg = SOL.get(comp, "cytosol")
                if reg in ("ER_lumen", "golgi_lumen", "endo_lumen") and r.signal_frac < 0.5: reg = "cytosol"
            ri = RN.index(reg)
            f = (pla[li, ri] / req[li, ri]) if (li is not None and req[li, ri] > 0) else 0.0
            fs = f_pm if reg in PM_REGS else f_vol
            x = ent.setdefault((comp, reg), [0.0, 0.0]); x[0] += rest * wt * fs; x[1] += rest * wt * f * fs
        for reg, c in pre.get(r.ing_id, {}).items():
            x = ent.setdefault(("prepass", reg), [0.0, 0.0]); x[0] += c; x[1] += c
        gs = str(r.genes).split(";"); st = [float(s) for s in str(r.stoich).split(";")]
        for g, s in zip(gs, st):
            members = famw[g].items() if g in famw else [(g, 1.0)]
            for gg, fw in members:
                for (comp, reg), (a, b) in ent.items():
                    rows.append((gg, r.ing_id, r.kind, s * fw, comp, reg, a * s * fw, b * s * fw, bool(r.transmembrane), tm,
                                 float(r.mass_da), float(r.signal_frac)))
    M = pd.DataFrame(rows, columns=["gene", "ing_id", "kind", "stoich", "src_comp", "region", "intended", "placed", "ing_tm",
                                    "ing_memlib", "ing_mass_da", "signal_frac"])
    M.to_parquet(f"{CACHE}/loc_model{SFX}.parquet")
    log(f"model table: {M.gene.nunique()} genes, {M.ing_id.nunique()} ingredients, intended {M.intended.sum():,.0f} "
        f"gene copies, placed-equivalent {M.placed.sum():,.0f}")


# ----------------------------------------------------------------- source vocabularies -> comparison compartments
# special (non-model) targets: "secretory" = secreted/extracellular (supports ER/golgi/endolysosomal/PM, never "missing");
# "anymem" = UniProt/GO generic "Membrane" (supports any membrane compartment). Sperm-only HPA terms are excluded.
SPERM = {"principal piece", "mid piece", "end piece", "acrosome", "calyx", "equatorial segment", "connecting piece",
         "flagellar centriole", "perinuclear theca", "annulus"}
HPA_MAP = {   # term -> (comp, flag)
    "nucleoplasm": ("nucleus", ""), "nuclear bodies": ("nucleus", ""), "nuclear speckles": ("nucleus", ""),
    "nucleoli": ("nucleus", "no_nucleolus"), "nucleoli fibrillar center": ("nucleus", "no_nucleolus"),
    "nucleoli rim": ("nucleus", "no_nucleolus"), "mitotic chromosome": ("nucleus", "mitotic"), "kinetochore": ("nucleus", "mitotic"),
    "nuclear membrane": ("NE", ""),
    "cytosol": ("cytosol", ""), "cytoplasmic bodies": ("cytosol", ""), "rods & rings": ("cytosol", ""), "aggresome": ("cytosol", ""),
    "actin filaments": ("cytosol", "cortex_not_modelled" if VER == "v4" else ""),   # v5+: + PM support (cortex), see sources() "focal adhesion sites": ("PM", "adherent_cells"),
    "microtubules": ("cytosol", ""), "microtubule ends": ("cytosol", ""), "intermediate filaments": ("cytosol", ""),
    "centrosome": ("cytosol", ""), "centriolar satellite": ("cytosol", ""), "basal body": ("cytosol", "cilium"),
    "primary cilium": ("cytosol", "cilium"), "primary cilium tip": ("cytosol", "cilium"),
    "primary cilium transition zone": ("cytosol", "cilium"), "mitotic spindle": ("cytosol", "mitotic"),
    "midbody": ("cytosol", "mitotic"), "midbody ring": ("cytosol", "mitotic"), "cytokinetic bridge": ("cytosol", "mitotic"),
    "cleavage furrow": ("cytosol", "mitotic"),
    "peroxisomes": ("cytosol", "no_peroxisome"), "lipid droplets": ("cytosol", "no_lipid_droplet"),
    "mitochondria": ("mito", ""), "endoplasmic reticulum": ("ER", ""), "golgi apparatus": ("golgi", ""),
    "plasma membrane": ("PM", ""), "cell junctions": ("PM", ""),
    "vesicles": ("endolysosomal", ""), "endosomes": ("endolysosomal", ""), "lysosomes": ("endolysosomal", ""),
}
OC_MAP = {"cytoplasmic": "cytosol", "nucleoplasm": "nucleus", "nucleolus_gc": "nucleus", "nucleolus_fc_dfc": "nucleus",
          "chromatin": "nucleus", "nuclear_punctae": "nucleus", "nuclear_membrane": "NE", "membrane": "PM",
          "cell_contact": "PM", "focal_adhesions": "PM", "er": "ER", "golgi": "golgi", "mitochondria": "mito",
          "vesicles": "endolysosomal", "centrosome": "cytosol", "cytoskeleton": "cytosol", "big_aggregates": "cytosol"}
UP_RULES = [  # ordered regex on the lowercased location head (UniProt statement or GO-CC term name)
    (r"nucle(us|ar) (inner |outer )?membrane|nucle(us|ar) envelope|nuclear pore|nuclear lamina", "NE", ""),
    (r"mitochondri", "mito", ""),
    (r"endoplasmic reticulum|microsome|sarcoplasmic reticulum", "ER", ""),
    (r"golgi", "golgi", ""),
    (r"peroxisom", "cytosol", "no_peroxisome"), (r"lipid droplet", "cytosol", "no_lipid_droplet"),
    (r"stress granule|p-body", "cytosol", ""),
    (r"endosom|lysosom|vesic|vacuol|phagosom|autophag|melanosom|granule|multivesicular|endomembrane|secretory", "endolysosomal", ""),
    (r"cell membrane|plasma membrane|cell surface|membrane raft|cell junction|focal adhesion|synap|uropod|immunological|"
     r"sarcolemma|apical|basolateral|projection.*membrane|ruffle membrane|lamellipodium membrane|microvillus membrane", "PM", ""),
    (r"nucleol", "nucleus", "no_nucleolus"),
    (r"nucle|chromosom|kinetochore|centromere|telomere|pml body|cajal|speckle", "nucleus", ""),
    (r"cell cortex|cortical actin", "cytosol", "cortex_not_modelled"),
    (r"cytoplasm|cytosol|cytoskeleton|centrosome|centriol|midbody|perikaryon|myofibril|spindle|cilium|flagell|sarcomere|"
     r"cell projection|axon|dendrit|lamellipod|filopod|ruffle|microvill|growth cone|actin|microtubule|intermediate filament", "cytosol", ""),
    (r"secreted|extracellular|cell wall", "secretory", ""),
    (r"^membrane$|^membrane;|^membrane ", "anymem", ""),
]
UP_RULES_C = [(re.compile(a), b, c) for a, b, c in UP_RULES]
GO_SKIP = re.compile(r"extracellular exosome|extracellular vesicle|protein-containing complex|^intracellular|"
                     r"ribonucleoprotein complex$|^organelle|^cellular_component")
MEMCOMP = {"PM", "NE", "ER", "golgi", "endolysosomal", "mito"}
SRC2COMP = {"plasma_membrane": "PM", "golgi": "golgi", "er": "ER", "endolysosomal": "endolysosomal", "nuclear_envelope": "NE",
            "mitochondrion": "mito", "nucleus": "nucleus", "cytosol": "cytosol", "cytoskeleton": "cytosol",
            "unassigned": "cytosol", "peroxisome": "cytosol", "lipid_droplet": "cytosol", "secreted": "secretory", "prepass": None}
REL_W = {"Enhanced": 1.0, "Supported": 0.8, "Approved": 0.5, "Uncertain": 0.0}
W_UNIPROT, W_GO = 0.6, 0.3
DISAGREE = 0.25                     # combined fit score at or above which a gene counts as disagreeing


def up_comp(head):
    h = head.strip().lower()
    for rx, c, fl in UP_RULES_C:
        if rx.search(h): return c, fl
    return None, ""


def parse_uniprot(text):
    """UniProt 'Subcellular location [CC]' -> list of (head, comp, flag, anchor) per statement"""
    if not isinstance(text, str): return []
    t = re.sub(r"\{[^}]*\}", "", text).replace("SUBCELLULAR LOCATION:", " ")
    t = re.sub(r"Note=.*?(?=SUBCELLULAR|$)", "", t, flags=re.S)
    out = []
    for st in re.split(r"\.\s+|\.$", t):
        st = re.sub(r"^\s*\[[^\]]*\]:\s*", "", st.strip())
        if not st: continue
        parts = [x.strip() for x in st.split(";")]
        head = parts[0]; rest = " ".join(parts[1:]).lower()
        anchor = "gpi" if "gpi-anchor" in rest or "gpi-like" in rest else "lipid" if "lipid-anchor" in rest else \
            "peripheral" if "peripheral membrane protein" in rest else "tm" if ("pass membrane" in rest) else ""
        c, fl = up_comp(head)
        out.append((head, c, fl, anchor))
    return out


def dist(terms):
    """terms: list of (comp, weight) -> {comp: max weight}"""
    q = {}
    for c, w in terms:
        if c: q[c] = max(q.get(c, 0.0), w)
    return q


def score(p, q, main_needed=True):
    """p: model comp distribution (sums to 1). q: {comp: weight}; weight 1 = main.
    unsup = model mass in compartments the source does not annotate; miss = share of the source's main compartments
    with < 5% of the model's copies; score = 0.5 unsup + 0.5 miss; tvd = total variation to the normalised source."""
    sup = {c for c in q if c in COMPS}
    if "secretory" in q: sup |= {"ER", "golgi", "endolysosomal", "PM"}
    if "anymem" in q: sup |= MEMCOMP
    unsup = sum(v for c, v in p.items() if c not in sup)
    main = [c for c, w in q.items() if w >= 0.999 and c in COMPS]
    miss = (sum(p.get(c, 0.0) < 0.05 for c in main) / len(main)) if (main and main_needed) else 0.0
    qq = {c: w for c, w in q.items() if c in COMPS}; s = sum(qq.values())
    tvd = 0.5 * sum(abs(p.get(c, 0.0) - qq.get(c, 0.0) / s) for c in COMPS) if s else np.nan
    top = max(p, key=p.get) if p else None
    return dict(unsup=unsup, miss=miss, score=0.5 * unsup + 0.5 * miss, tvd=tvd, tophit=float(top in sup) if top else np.nan)


def mapping_table():
    rows = []
    reg_of = {"PM": "PM (TM) | soluble -> cytosol", "cytosol": "cytosol", "nucleus": "nucleoplasm (no nucleolus compartment)",
              "NE": "NE membrane (TM) | perinuclear lumen | soluble -> nucleoplasm", "ER": "ER membrane (TM) | ER_lumen (signal peptide) | else cytosol",
              "mito": "MITO_OM / MITO_IM (TM) | matrix", "golgi": "GOLGI membrane (TM) | golgi_lumen (signal peptide) | else cytosol",
              "endolysosomal": "ENDO membrane (TM) | endo_lumen (signal peptide) | else cytosol",
              "secretory": "no compartment: supports ER/golgi/endolysosomal/PM (model: ER_lumen)", "anymem": "supports any membrane compartment"}
    H = pd.read_csv(f"{REF}/hpa_subcellular/subcellular_location.tsv", sep="\t")
    hterms = sorted({t for col in ["Main location", "Additional location"] for s_ in H[col].dropna() for t in s_.split(";")})
    for t in hterms:
        c, fl = HPA_MAP.get(t.lower(), (None, "sperm_excluded" if t.lower() in SPERM else "unmapped"))
        rows.append(dict(source="HPA", term=t, comp=c or "", model_regions=reg_of.get(c, ""), flag=fl,
                         weight="main 1 / additional 0.3 x reliability (E 1, S 0.8, A 0.5, Uncertain 0)"))
    rows.append(dict(source="HPA", term="Predicted to be secreted (extracellular column)", comp="", model_regions="", flag="ignored", weight=""))
    for t, c in OC_MAP.items():
        rows.append(dict(source="OpenCell", term=t, comp=c, model_regions=reg_of[c],
                         flag="no_nucleolus" if "nucleolus" in t else "", weight="grade 3: 1, grade 2: 0.6, grade 1: 0.3 (held out)"))
    for rx, c, fl in UP_RULES:
        rows.append(dict(source="UniProt CC / GO-CC (regex on location head)", term=rx, comp=c, model_regions=reg_of.get(c, ""), flag=fl,
                         weight=f"UniProt: every statement main, x{W_UNIPROT}; GO-CC: support only, x{W_GO}"))
    rows.append(dict(source="GO-CC", term=GO_SKIP.pattern, comp="", model_regions="", flag="skipped (uninformative)", weight=""))
    for k, c in SRC2COMP.items():
        rows.append(dict(source="model inventory comp_weights", term=k, comp=c or "pre-pass region",
                         model_regions=f"soluble: {SOL.get(k, '-')}; TM: {TMR.get(k, '-')}", flag="", weight="intended copies"))
    for r_, c in REG2COMP.items():
        rows.append(dict(source="model region (pack)", term=r_, comp=c, model_regions=r_, flag="", weight="placed copies"))
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/loc_mapping.tsv", sep="\t", index=False)
    return T


# ----------------------------------------------------------------- gene table
def gene_table():
    M = pd.read_parquet(f"{CACHE}/loc_model{SFX}.parquet")
    V0 = pd.read_csv(f"{D}/inventory_v0.tsv", sep="\t").drop_duplicates("gene").set_index("gene")
    M["comp"] = M.region.map(REG2COMP); M["soluble"] = M.region.isin(SOLUBLE)
    M["src_c"] = M.src_comp.map(SRC2COMP)
    # membrane-intended share packed soluble into a different comparison compartment (peripheral rule target)
    M["lost_mem"] = M.src_c.isin(MEMCOMP) & M.soluble & (M.src_c != M.comp)
    gi = M.groupby(["gene", "comp"]).intended.sum().unstack(fill_value=0).reindex(columns=COMPS, fill_value=0)
    gp = M.groupby(["gene", "comp"]).placed.sum().unstack(fill_value=0).reindex(columns=COMPS, fill_value=0)
    gr = M.groupby(["gene", "region"]).intended.sum().unstack(fill_value=0)
    G = pd.DataFrame(index=gi.index)
    G["copies"] = gi.sum(1); G["copies_placed"] = gp.reindex(G.index).sum(1)
    G["placed_frac"] = G.copies_placed / G.copies
    G["mw_kda"] = V0.mass_da.reindex(G.index) / 1000
    G["mass_pg"] = G.copies * G.mw_kda * 1000 * 1.66054e-12
    G["uniprot"] = V0.uniprot.reindex(G.index); G["loc_source"] = V0.loc_source.reindex(G.index)
    G["inv_compartments"] = V0.compartments.reindex(G.index)
    for c in COMPS:
        G[f"m_{c}"] = (gi[c] / G.copies).round(4)
        G[f"pl_{c}"] = (gp.reindex(G.index)[c] / G.copies_placed.replace(0, np.nan)).round(4)
    G["model_regions"] = [";".join(f"{r_}:{v / tot:.2f}" for r_, v in row.sort_values(ascending=False).items() if v / tot >= 0.01)
                          for (g, row), tot in zip(gr.iterrows(), gr.sum(1))]
    G["soluble_frac"] = M[M.soluble].groupby("gene").intended.sum().reindex(G.index).fillna(0) / G.copies
    G["tm_ing_soluble_frac"] = M[M.soluble & M.ing_tm].groupby("gene").intended.sum().reindex(G.index).fillna(0) / G.copies
    G["lost_mem_frac"] = M[M.lost_mem].groupby("gene").intended.sum().reindex(G.index).fillna(0) / G.copies
    sol = M[M.soluble & ~M.ing_tm]
    G["unit_kda"] = (sol.intended * sol.ing_mass_da).groupby(sol.gene).sum() / sol.groupby("gene").intended.sum() / 1000
    G["n_ingredients"] = M.groupby("gene").ing_id.nunique()
    G["in_complex"] = M[M.kind != "protein"].groupby("gene").intended.sum().reindex(G.index).fillna(0) / G.copies
    return G, M


def model_p(G, prefix="m_"):
    return {g: {c: float(v) for c, v in zip(COMPS, row) if v > 0} for g, row in zip(G.index, G[[f"{prefix}{c}" for c in COMPS]].values)}


# ----------------------------------------------------------------- sources per gene
def sources(G):
    hpa = hpa_table()
    up = pd.read_csv(f"{REF}/uniprot_human_full.tsv.gz", sep="\t", usecols=["Entry", "Gene Names (primary)", "Subcellular location [CC]",
                     "Gene Ontology (cellular component)", "Transmembrane", "Intramembrane", "Lipidation", "Signal peptide", "Mass"]).set_index("Entry")
    oc = pd.read_csv(f"{REF}/opencell/opencell-localization-annotations.csv").drop_duplicates("target_name").set_index("target_name")
    S = {}
    for g in G.index:
        d = {}
        if g in hpa.index:
            h = hpa.loc[g]; rel = h["Reliability"]
            terms, flags = [], set()
            for col, w in (("Main location", 1.0), ("Additional location", 0.3)):
                for t in str(h[col]).split(";") if isinstance(h[col], str) else []:
                    c, fl = HPA_MAP.get(t.lower(), (None, ""))
                    if c: terms.append((c, w))
                    if t.lower() == "actin filaments" and VER != "v4": terms.append(("PM", 0.3 * w))   # cortex = PM support
                    if fl: flags.add(fl)
            d["hpa"] = dict(q=dist(terms), rel=rel, w=REL_W.get(rel, 0.0), main=h["Main location"], add=h["Additional location"],
                            flags=flags)
        u = G.at[g, "uniprot"]
        if isinstance(u, str) and u in up.index:
            r = up.loc[u]; st = parse_uniprot(r["Subcellular location [CC]"])
            q = dist([(c, 1.0) for _, c, _, _ in st])
            lip = r["Lipidation"] if isinstance(r["Lipidation"], str) else ""
            lipid = sorted({k for k, rx in (("myristoyl", "myristoyl"), ("palmitoyl", "palmitoyl"), ("prenyl", "farnesyl|geranylgeranyl"),
                                            ("GPI", "GPI-anchor amidated")) if re.search(rx, lip)})
            anchors = sorted({f"{a}@{c}" for _, c, _, a in st if a in ("lipid", "gpi", "peripheral") and c})
            go = []; gocc = str(r["Gene Ontology (cellular component)"]).lower()
            actin = bool(re.search(r"actin|cell cortex|cortical", gocc + " " + str(r["Subcellular location [CC]"]).lower()))
            if isinstance(r["Gene Ontology (cellular component)"], str):
                for t in r["Gene Ontology (cellular component)"].split(";"):
                    name = re.sub(r"\s*\[GO:\d+\]", "", t).strip().lower()
                    if not name or GO_SKIP.search(name): continue
                    c, _ = up_comp(name); go.append((c, 1.0))
            d["uniprot"] = dict(q=q, w=W_UNIPROT if q else 0.0, text=";".join(sorted({h_ for h_, c, _, _ in st if c})),
                                flags={fl for _, _, fl, _ in st if fl}, lipid=lipid, anchors=anchors, actin=actin,
                                tm=(isinstance(r["Transmembrane"], str) and "TRANSMEM" in r["Transmembrane"]) or
                                   (isinstance(r["Intramembrane"], str) and "INTRAMEM" in r["Intramembrane"]),
                                sp=isinstance(r["Signal peptide"], str) and len(r["Signal peptide"]) > 0)
            d["go"] = dict(q=dist(go), w=W_GO if go else 0.0)
        if g in oc.index:
            o = oc.loc[g]; terms = []
            for col, w in (("annotations_grade_3", 1.0), ("annotations_grade_2", 0.6), ("annotations_grade_1", 0.3)):
                if isinstance(o[col], str):
                    for t in o[col].split(";"): terms.append((OC_MAP.get(t.strip()), w))
            q = dist(terms)
            if q and max(q.values()) < 1:                       # no grade-3 term: promote the best grade to "main"
                mx = max(q.values()); q = {c: (1.0 if w == mx else w) for c, w in q.items()}
            d["opencell"] = dict(q=q, w=1.0, text=" | ".join(f"g{k}:{o[f'annotations_grade_{k}']}" for k in (3, 2, 1)
                                                             if isinstance(o[f"annotations_grade_{k}"], str)))
        inv = G.at[g, "inv_compartments"]
        if isinstance(inv, str) and inv:
            tm = bool(d.get("uniprot", {}).get("tm")); sp = bool(d.get("uniprot", {}).get("sp")); pi = {}
            for kv in inv.split(";"):
                k, v = kv.rsplit(":", 1)
                reg = TMR.get(k, "ER") if tm else SOL.get(k, "cytosol")
                if not tm and reg in ("ER_lumen", "golgi_lumen", "endo_lumen") and not sp: reg = "cytosol"
                c = REG2COMP[reg]; pi[c] = pi.get(c, 0) + float(v)
            t_ = sum(pi.values()); d["_pinv"] = {c: v / t_ for c, v in pi.items()} if t_ else None
        S[g] = d
    return S


def jurkat_refs():
    f = f"{OUT}/hpa_celllines.tsv"
    if not os.path.exists(f): return {}
    C = pd.read_csv(f, sep="\t", keep_default_na=False)
    J = C[(C.cell_line.str.upper() == "JURKAT")]
    out = {}
    for g, sub in J.groupby("gene"):
        locs = [t for x in sub.locations for t in x.split(";") if t]
        terms = [(HPA_MAP.get(t.lower(), (None, ""))[0], 1.0) for t in locs]
        if VER != "v4": terms += [("PM", 0.3) for t in locs if t.lower() == "actin filaments"]   # cortex = PM support
        out[g] = dict(q=dist([(c, w) for c, w in terms if c]), text=";".join(sorted(set(locs))),
                      rel=";".join(sorted(set(sub.ab_reliability))), n_ab=sub.antibody.nunique())
    return out


# ----------------------------------------------------------------- compare
def combine(p, S_g):
    num = den = 0.0; per = {}
    for src in ("hpa", "uniprot", "go"):
        if src not in S_g or S_g[src]["w"] <= 0 or not S_g[src]["q"]: continue
        sc = score(p, S_g[src]["q"], main_needed=(src != "go")); per[src] = sc
        x = sc["unsup"] if src == "go" else sc["score"]
        num += S_g[src]["w"] * x; den += S_g[src]["w"]
    return (num / den if den else np.nan), per


def classify(g, G, S_g, per, comb):
    """primary cause class of a disagreeing gene, judged against the driving source (largest weighted contribution)"""
    r = G.loc[g]; up = S_g.get("uniprot", {}); hp = S_g.get("hpa", {})
    p = {c: r[f"m_{c}"] for c in COMPS}
    contrib = {k: S_g[k]["w"] * (v["unsup"] if k == "go" else v["score"]) for k, v in per.items()}
    drv = max(contrib, key=contrib.get); q = S_g[drv]["q"]; sc = per[drv]
    main = [c for c, w in q.items() if w >= 0.999 and c in COMPS]
    miss_mem = [c for c in main if c in MEMCOMP and p.get(c, 0) < 0.05]
    anchored = bool(up.get("lipid") or up.get("anchors"))
    lipidated = bool(up.get("lipid")) or any(a.startswith(("lipid@", "gpi@")) for a in up.get("anchors", []))
    tgts = MEMCOMP if lipidated else ({a.split("@")[1] for a in up.get("anchors", [])} & MEMCOMP or MEMCOMP)
    # (c) TM protein / TM complex packed soluble
    if (r.tm_ing_soluble_frac >= 0.2) or (up.get("tm") and r.soluble_frac >= 0.5):
        return "c_tm_in_soluble", drv
    # (a) peripheral / lipid-anchored, packed soluble, while the driving source puts it on that membrane
    if anchored and r.soluble_frac >= 0.5 and set(miss_mem) & tgts:
        return "a_peripheral_or_lipid_anchored", drv
    # (b) small soluble cytosolic protein without a nuclear pool while the driving source lists the nucleus
    if (p["nucleus"] < 0.05 and p["cytosol"] >= 0.8 and (r.unit_kda if r.unit_kda == r.unit_kda else r.mw_kda) < 60
            and q.get("nucleus", 0) > 0):
        return "b_small_soluble_no_nuclear_pool", drv
    # (h) assembly localisation: the gene's own inventory localisation fits the driving source clearly better than the
    #     model, i.e. complexes / dimers it is built into carry other compartment weights (subunit-averaged comp_weights)
    pinv = S_g.get("_pinv")
    if pinv and r.in_complex >= 0.3 and sc["score"] - score(pinv, q, main_needed=(drv != "go"))["score"] >= 0.2:
        return "h_assembly_localisation", drv
    # (i) nucleus/cytosol balance set by fractionation proteomics (Itzhak 2016 HeLa 1K pellet) against an imaging source
    if "itzhak" in str(r.loc_source) and set(main) <= {"nucleus", "cytosol"} and p["nucleus"] + p["cytosol"] >= 0.9:
        return "i_fractionation_nc_share", drv
    # (d) source conflict: HPA and UniProt disagree with each other and the model sides with one of them
    if "hpa" in per and "uniprot" in per:
        hq, uq = hp["q"], up["q"]; cs = set(hq) | set(uq)
        h_n = {c: hq.get(c, 0) / sum(hq.values()) for c in cs}; u_n = {c: uq.get(c, 0) / sum(uq.values()) for c in cs}
        if 0.5 * sum(abs(h_n[c] - u_n[c]) for c in cs) >= 0.5 and min(per["hpa"]["score"], per["uniprot"]["score"]) < DISAGREE:
            return "d_source_conflict", drv
    # (g) membrane main location (often PM / cortex) without anchor annotation, packed soluble
    if miss_mem and r.soluble_frac >= 0.8:
        return "g_membrane_main_no_anchor_packed_soluble", drv
    # (e) multi-localisation shares: the model's mass is supported, but a main location is missing / shares differ
    if sc["unsup"] < 0.2 or len([c for c in q if c in COMPS]) >= 3:
        return "e_multilocalisation_shares", drv
    return "f_single_protein", drv


def compare():
    os.makedirs(OUT, exist_ok=True)
    mapping_table()
    G, M = gene_table(); S = sources(G); J = jurkat_refs()
    P = model_p(G); PP = model_p(G, "pl_")
    rows = []
    for g in G.index:
        d = S[g]; p = P[g]
        comb, per = combine(p, d)
        combp, _ = combine(PP[g], d) if PP[g] else (np.nan, {})
        r = dict(gene=g, combined_score=comb, combined_score_placed=combp, hpa_split=hpa_split(g))
        for k in ("hpa", "uniprot", "go", "opencell"):
            if k in d and d[k]["q"]:
                sc = per.get(k) or score(p, d[k]["q"], main_needed=(k != "go"))
                for m_ in ("unsup", "miss", "score", "tvd", "tophit"): r[f"{k}_{m_}"] = sc[m_]
        if g in J and J[g]["q"]:
            sc = score(p, J[g]["q"])
            for m_ in ("unsup", "miss", "score", "tvd", "tophit"): r[f"jurkat_{m_}"] = sc[m_]
        hp = d.get("hpa", {}); up = d.get("uniprot", {})
        r["actin_cortex_annot"] = bool(up.get("actin")) or "Actin filaments" in str(hp.get("main")) + str(hp.get("add"))
        r["tm_in_soluble_flag"] = bool(G.at[g, "tm_ing_soluble_frac"] >= 0.2 or (up.get("tm") and G.at[g, "soluble_frac"] >= 0.5))
        r.update(hpa_main=hp.get("main"), hpa_add=hp.get("add"), hpa_reliability=hp.get("rel"),
                 hpa_flags=";".join(sorted(hp.get("flags", set()) | up.get("flags", set()))),
                 uniprot_loc=up.get("text"), uniprot_lipid=";".join(up.get("lipid", [])),
                 uniprot_anchor=";".join(up.get("anchors", [])), uniprot_tm=up.get("tm"), uniprot_sp=up.get("sp"),
                 opencell=d.get("opencell", {}).get("text"), hpa_jurkat=J.get(g, {}).get("text"),
                 hpa_jurkat_rel=J.get(g, {}).get("rel"))
        if comb == comb and comb >= DISAGREE: r["cause"], r["driver"] = classify(g, G, d, per, comb)
        else: r["cause"], r["driver"] = ("agree" if comb == comb else "no_fit_source"), ""
        rows.append(r)
    R = pd.DataFrame(rows).set_index("gene")
    T = G.join(R)
    T["disagree_mass_pg"] = (T.combined_score * T.mass_pg)
    T = T.sort_values("disagree_mass_pg", ascending=False)
    T.round(4).to_csv(f"{OUT}/loc_compare{SFX}.tsv", sep="\t")
    log(f"loc_compare.tsv: {len(T)} genes, {T.mass_pg.sum():.2f} pg")
    rules(T, M, S)
    return T


# ----------------------------------------------------------------- rule-level fixes (counterfactuals on the model distribution)
def _rescore(T, S, newp):
    """combined fit score for genes in newp (dict gene -> distribution)"""
    return {g: combine(p, S[g])[0] for g, p in newp.items()}


def rules(T, M, S):
    Mv = pd.read_parquet(f"{W}/tiles.parquet", columns=["cytosol", "nucleoplasm"]).sum()
    Vn, Vc = float(Mv["nucleoplasm"]), float(Mv["cytosol"])
    P = model_p(T)
    fit = T.combined_score.notna()
    base_dm = float((T.combined_score * T.mass_pg)[fit].sum())
    out = {}
    M = M.copy(); M["comp"] = M.region.map(REG2COMP); M["soluble"] = M.region.isin(SOLUBLE); M["src_c"] = M.src_comp.map(SRC2COMP)
    tot = T.copies

    ch = pd.read_csv(f"{D}/inventory_v4_changes.tsv", sep="\t")
    cur = ch[ch.field.isin(["comp_weights", "transmembrane"]) & ch.reason.astype(str).str.match(r'^"?[bcf]')]
    curated = {g for x in cur.gene.astype(str) for g in x.split(";")}
    out["curated_v4_excluded"] = sorted(curated & set(T.index))

    def apply(name, moves, genes_affected, desc, keep_curated=False):
        """moves: DataFrame gene, from_comp, to_comp, copies (v4-curated localisations are not overridden, except by R2,
        whose nuclear pool adds to the curated compartments instead of replacing them)"""
        if not keep_curated: moves = moves[~moves.gene.isin(curated)]
        genes_affected = set(moves.gene)
        newp = {}
        for g, sub in moves.groupby("gene"):
            p = dict(P[g])
            for fc, tc, x in sub[["from_comp", "to_comp", "copies"]].values:
                f = x / tot[g]; p[fc] = p.get(fc, 0) - f; p[tc] = p.get(tc, 0) + f
            newp[g] = {c: v for c, v in p.items() if v > 1e-9}
        ns = _rescore(T, S, newp)
        moved_pg = float((moves.copies * moves.gene.map(T.mw_kda) * 1000 * 1.66054e-12).sum())
        before = T.loc[list(ns), "combined_score"]; after = pd.Series(ns)
        m = before.notna() & after.notna()
        dm = float(((after[m] - before[m]) * T.loc[after[m].index, "mass_pg"]).sum())
        dis_b = int((before[m] >= DISAGREE).sum()); dis_a = int((after[m] >= DISAGREE).sum())
        out[name] = dict(desc=desc, proteins=int(len(genes_affected)), mass_moved_pg=round(moved_pg, 3),
                         gene_mass_pg=round(float(T.loc[list(genes_affected), "mass_pg"].sum()), 3),
                         delta_disagree_mass_pg=round(dm, 3), disagreeing_before=dis_b, disagreeing_after=dis_a,
                         improved=int((after[m] < before[m] - 1e-9).sum()), worsened=int((after[m] > before[m] + 1e-9).sum()),
                         top_genes=list((moves.copies * moves.gene.map(T.mw_kda)).groupby(moves.gene).sum()
                                        .sort_values(ascending=False).index[:12]))
        return newp

    # R1 peripheral class: lipid/peripheral-anchored, non-TM proteins: membrane-intended shares packed soluble -> membrane surface
    anch = {g for g in T.index if isinstance(T.at[g, "uniprot_lipid"], str) and T.at[g, "uniprot_lipid"] or
            isinstance(T.at[g, "uniprot_anchor"], str) and T.at[g, "uniprot_anchor"]}
    mv = M[M.gene.isin(anch) & M.soluble & M.src_c.isin(MEMCOMP) & (M.src_c != M.comp) & ~M.ing_tm]
    mv = mv.groupby(["gene", "comp", "src_c"]).intended.sum().reset_index()
    mv.columns = ["gene", "from_comp", "to_comp", "copies"]
    apply("R1_peripheral_class", mv, set(mv.gene), "lipid-anchored / peripheral (UniProt) non-TM proteins: the inventory's membrane "
          "compartment shares (PM, Golgi, NE, ER, endolysosomal, mito) are packed on that membrane's cytosolic/nucleoplasmic surface "
          "instead of the bulk soluble region")
    lip = {g for g in anch if isinstance(T.at[g, "uniprot_lipid"], str) and T.at[g, "uniprot_lipid"]
           or "lipid@" in str(T.at[g, "uniprot_anchor"]) or "gpi@" in str(T.at[g, "uniprot_anchor"])}
    mvl = mv[mv.gene.isin(lip)]
    apply("R1_lipid_only", mvl, set(mvl.gene), "as R1, restricted to lipidated proteins (UniProt LIPID feature or Lipid-/GPI-anchor)")
    # R1b: same, using the UniProt anchor target when the inventory has no membrane share for it
    rows = []
    for g in anch:
        tgt = {a.split("@")[1] for a in str(T.at[g, "uniprot_anchor"]).split(";") if "@" in a} & MEMCOMP
        if not tgt or T.at[g, "lost_mem_frac"] >= 0.05: continue
        if any(P[g].get(c, 0) >= 0.05 for c in tgt): continue
        share = 0.5 * T.at[g, "soluble_frac"] * T.at[g, "copies"]
        src = "nucleus" if (P[g].get("nucleus", 0) > P[g].get("cytosol", 0) and "NE" in tgt) else "cytosol"
        for c in tgt: rows.append((g, src, c, share / len(tgt)))
    mv = pd.DataFrame(rows, columns=["gene", "from_comp", "to_comp", "copies"])
    apply("R1b_anchor_target_without_inventory_share", mv, set(mv.gene), "anchored proteins whose inventory has no share in the "
          "UniProt anchor membrane: put 50% of the soluble pool on that membrane surface")
    # R2 nuclear pool by size: soluble cytosolic proteins, unit < 60 kDa, nucleus < 5%
    def r_nc(kda): return float(np.clip(1.0 - 0.5 * (kda - 30) / 30, 0.5, 1.0))
    cand = T[(T.m_nucleus < 0.05) & (T.m_cytosol >= 0.8) & (T.unit_kda.fillna(T.mw_kda) < 60) & (T.tm_ing_soluble_frac == 0)
             & ~(T.uniprot_tm.fillna(False).astype(bool)) & ~(T.uniprot_sp.fillna(False).astype(bool))]
    rows = []
    for g, r in cand.iterrows():
        k = r_nc(r.unit_kda if r.unit_kda == r.unit_kda else r.mw_kda)
        s_ = Vn * k / (Vn * k + Vc)
        rows.append((g, "cytosol", "nucleus", r.m_cytosol * r.copies * s_))
    mv = pd.DataFrame(rows, columns=["gene", "from_comp", "to_comp", "copies"])
    apply("R2_size_nuclear_pool_all", mv, set(mv.gene), f"soluble cytosolic proteins (packed unit < 60 kDa, no TM/signal peptide, model "
          f"nucleus < 5%): nuclear share = Vn r/(Vn r + Vc), N/C r = 1 at <= 30 kDa falling to 0.5 at 60 kDa "
          f"(Vn {Vn / 1e9:.0f}, Vc {Vc / 1e9:.0f} um3: share 0.40 -> 0.25)", keep_curated=True)
    nucsrc = {g for g in cand.index if any(S[g].get(k, {}).get("q", {}).get("nucleus", 0) > 0 for k in ("hpa", "uniprot"))}
    mv2 = mv[mv.gene.isin(nucsrc)]
    apply("R2b_size_nuclear_pool_annotated", mv2, set(mv2.gene), "as R2, only where HPA or UniProt lists a nuclear location",
          keep_curated=True)
    # R3 TM proteins/complexes packed soluble -> their membrane (TMR region of the source compartment)
    tmr_comp = {k: REG2COMP[v] for k, v in TMR.items()}
    upt = set(T.index[T.uniprot_tm.fillna(False).astype(bool)])
    mv = M[M.soluble & (M.ing_tm | M.gene.isin(upt)) & (M.src_comp != "prepass")].copy()
    mv["to_comp"] = mv.src_comp.map(tmr_comp)
    mv.loc[(mv.src_comp == "mitochondrion"), "to_comp"] = "mito"
    mv = mv[mv.to_comp != mv.comp].groupby(["gene", "comp", "to_comp"]).intended.sum().reset_index()
    mv.columns = ["gene", "from_comp", "to_comp", "copies"]
    tm_ing = M[M.soluble & M.ing_tm & (M.src_comp != "prepass")]
    u = tm_ing.drop_duplicates(["ing_id", "src_comp", "region"])
    out["R3_note"] = dict(tm_ingredients_in_soluble=int(tm_ing.ing_id.nunique()),
                          tm_ingredients_in_soluble_pg=round(float((u.intended / u.stoich * u.ing_mass_da).sum() * 1.66054e-12), 3),
                          no_membrane_library=int(tm_ing[~tm_ing.ing_memlib].ing_id.nunique()))
    apply("R3_tm_to_membrane", mv, set(mv.gene), "TM proteins (UniProt TM or TM ingredient) whose copies sit in a soluble region "
          "(no membrane library, or non-TM complex/dimer containing a TM subunit): route to the membrane of the intended compartment "
          "(jpack_test.TMR; nucleus -> NE, matrix -> MITO_IM)")
    # R6 cortical layer: soluble actin/cortex-annotated proteins whose fit source has PM as a main location the model lacks
    rows = []
    for g in T.index[(T.cause.isin(["a_peripheral_or_lipid_anchored", "g_membrane_main_no_anchor_packed_soluble"]))
                     & T.actin_cortex_annot.astype(bool) & (T.m_PM < 0.05)]:
        if not any(S[g].get(k, {}).get("q", {}).get("PM", 0) >= 0.999 and S[g].get(k, {}).get("w", 0) > 0 for k in ("hpa", "uniprot")):
            continue
        rows.append((g, "cytosol", "PM", 0.5 * T.at[g, "m_cytosol"] * T.at[g, "copies"]))
    mv = pd.DataFrame(rows, columns=["gene", "from_comp", "to_comp", "copies"])
    apply("R6_cortical_layer", mv, set(mv.gene), "actin/cortex-annotated soluble proteins (GO-CC/UniProt actin or cell cortex, HPA "
          "Actin filaments) whose HPA/UniProt main location is the PM: place 50% of the cytosolic pool in a cortical shell "
          "(cytosol within ~150 nm of the PM; counted as PM at light-microscopy resolution)")
    # R7 fractionation N/C capped by imaging: Itzhak-split proteins whose HPA (Enhanced/Supported) main location is nuclear only
    rows = []
    for g in T.index[T.loc_source.astype(str).str.contains("itzhak") & T.hpa_reliability.isin(["Enhanced", "Supported"])]:
        q = S[g].get("hpa", {}).get("q", {}); main = {c for c, w in q.items() if w >= 0.999}
        if main == {"nucleus"} and T.at[g, "m_cytosol"] > 0.2:
            rows.append((g, "cytosol", "nucleus", (T.at[g, "m_cytosol"] - 0.2) * T.at[g, "copies"]))
    mv = pd.DataFrame(rows, columns=["gene", "from_comp", "to_comp", "copies"])
    apply("R7_fractionation_nc_cap", mv, set(mv.gene), "proteins whose N/C split came from Itzhak 2016 HeLa fractionation but whose "
          "HPA Enhanced/Supported main location is nuclear only: cap the cytosolic share at 0.2 (fractionation leaks soluble "
          "nuclear proteins into the cytosolic fraction)")
    # R4 information only: nucleolus not modelled
    nl = T[T.hpa_main.fillna("").str.contains("Nucleoli")]
    out["R4_nucleolus_info"] = dict(desc="HPA main location Nucleoli*: mapped to nucleoplasm; the model has no nucleolus",
                                    proteins=int(len(nl)), gene_mass_pg=round(float(nl.mass_pg.sum()), 3))
    ax = T[T.hpa_main.fillna("").str.contains("Actin filaments")]
    out["R5_cortex_info"] = dict(desc="HPA main location Actin filaments (cortex in T cells): mapped to cytosol; no cortex layer",
                                 proteins=int(len(ax)), gene_mass_pg=round(float(ax.mass_pg.sum()), 3))
    out["baseline_disagree_mass_pg"] = round(base_dm, 3)
    T["v4_curated"] = T.index.isin(curated)
    json.dump(out, open(f"{CACHE}/loc_rules{SFX}.json", "w"), indent=1)
    log(json.dumps(out, indent=1))


# ----------------------------------------------------------------- hold-out test set (Tier 4)
_JH = None


def jurkat_half(g):
    """'holdout' / 'fit' for the seeded random halves of the HPA-JURKAT genes (tier2.py split), None otherwise"""
    global _JH
    if _JH is None:
        rd = lambda f: {l.strip() for l in open(f) if l.strip() and not l.startswith("#")} if os.path.exists(f) else set()
        _JH = (rd(f"{OUT}/jurkat_holdout_genes.txt"), rd(f"{OUT}/jurkat_fit_genes.txt"))
    return "holdout" if g in _JH[0] else "fit" if g in _JH[1] else None


def hpa_split(g):
    """deterministic HPA fit/test half (md5 of the gene symbol); the test half is never used for per-protein overrides"""
    import hashlib
    return "test" if int(hashlib.md5(g.encode()).hexdigest(), 16) % 2 else "fit"


def _agg(df, sc_col, mass):
    m = df[sc_col].notna(); w = mass[m]
    return dict(unweighted=round(float(df[sc_col][m].mean()), 4), mass_weighted=round(float((df[sc_col][m] * w).sum() / w.sum()), 4)) if m.any() else None


# v7+ (DECISION Pedro 2026-11-10): CD3E and CD3D are excluded from every hold-out score. Their v7 rule (R4b, TCR/CD3 ER and
# endosome pools) was motivated by the held-out HPA-JURKAT disagreement, so they are no longer clean. Stated in the paper.
HOLDOUT_EXCLUDE_V7 = {"CD3E", "CD3D"}
HOLDOUT_EXCLUDE = HOLDOUT_EXCLUDE_V7 if VER not in ("v4", "v5", "v6") else set()


def holdout():
    T = pd.read_csv(f"{OUT}/loc_compare{SFX}.tsv", sep="\t", index_col=0)
    S = sources(T); J = jurkat_refs()
    C = pd.read_csv(f"{OUT}/hpa_celllines.tsv", sep="\t", keep_default_na=False)
    Jrel = C[C.cell_line == "JURKAT"].groupby("gene").ab_reliability.agg(lambda x: set(x))
    P = model_p(T); PP = model_p(T, "pl_")
    sets = {}
    rows = []
    for g in T.index:
        if g in HOLDOUT_EXCLUDE: continue                    # v7+: CD3E / CD3D excluded (see HOLDOUT_EXCLUDE_V7)
        jh = jurkat_half(g)
        for name, q in (("hpa_jurkat", J.get(g, {}).get("q")), ("opencell", S[g].get("opencell", {}).get("q")),
                        (f"hpa_jurkat_{jh}_half" if jh else None, J.get(g, {}).get("q")),
                        ("hpa_test_half", S[g].get("hpa", {}).get("q") if (hpa_split(g) == "test" and S[g].get("hpa", {}).get("w", 0) > 0) else None)):
            if not q or not name: continue
            if name.startswith("hpa_jurkat") and Jrel.get(g, set()) <= {"uncertain"}: continue
            for mode, pp in (("intended", P[g]), ("placed", PP[g])):
                if not pp: continue
                sc = score(pp, q)
                rows.append(dict(gene=g, set=name, mode=mode, mass_pg=T.at[g, "mass_pg"], agree=1 - sc["score"],
                                 agree_tvd=1 - sc["tvd"], tophit=sc["tophit"], supported_mass=1 - sc["unsup"], main_present=1 - sc["miss"]))
    R = pd.DataFrame(rows); R.to_csv(f"{CACHE}/loc_holdout_pergene{SFX}.tsv", sep="\t", index=False)
    out = dict(model=f"pack {VER} (inventory {VER}, ingredients_v0.tsv; tiles data/derived/wc/tiles)", date=time.strftime("%Y-%m-%d"),
               definitions=dict(
                   agree="1 - (0.5 x model share in compartments the reference does not annotate + 0.5 x share of the reference's "
                         "main compartments holding < 5% of the model's copies)",
                   agree_tvd="1 - total-variation distance between model and normalised reference distribution (8 compartments)",
                   tophit="model's largest compartment is annotated by the reference",
                   supported_mass="model share in compartments the reference annotates",
                   main_present="share of reference main compartments with >= 5% of the model's copies",
                   compartments=COMPS, mass="per-gene protein mass in the model (pg)"),
               sets={})
    desc = {"hpa_jurkat": "HPA ICC/IF annotations made on JURKAT cells (per-antibody, per-line XML annotation; antibodies with "
                          "only 'uncertain' reliability excluded); images in data/reference/hpa_subcellular/jurkat_images/",
            "opencell": "OpenCell (HEK293T endogenous split-mNeonGreen tags), grades 3/2/1 = 1/0.6/0.3; best grade = main",
            "hpa_test_half": "HPA consensus (main 1 / additional 0.3, reliability Approved or better) for the md5 test half of genes. "
                             "NOT a clean hold-out from v5 on (rules used the HPA consensus for all genes)",
            "hpa_jurkat_holdout_half": "CLEAN hold-out: HPA JURKAT per-line annotations for the seeded random half of the JURKAT genes "
                                       "(validation/jurkat_holdout_genes.txt); never used to fit or calibrate (v6 rules used a "
                                       "non-JURKAT HPA consensus for these genes)",
            "hpa_jurkat_fit_half": "HPA JURKAT per-line annotations for the other half (jurkat_fit_genes.txt); used to calibrate v6 rules "
                                   "(not a hold-out)"}
    for (name, mode), sub in R.groupby(["set", "mode"]):
        sub = sub.set_index("gene"); m = sub.mass_pg
        d = out["sets"].setdefault(name, dict(description=desc[name], n_genes=int(len(sub)), mass_pg=round(float(m.sum()), 3),
                                              clean=name in ("opencell", "hpa_jurkat_holdout_half")))
        d[mode] = {k: _agg(sub, k, m) for k in ("agree", "agree_tvd", "tophit", "supported_mass", "main_present")}
    out["leakage_notes"] = [
        "HPA consensus locations (subcellular_location.tsv) pool all imaged lines, including JURKAT, and the inventory used them "
        "(50/50 with UniProt): HPA-JURKAT agreement is partly circular. Rule: no HPA-driven per-protein override for genes with "
        "JURKAT images; rules are fitted on the HPA fit half only.",
        "OpenCell was never used by the inventory and must not be used for fixes.",
        "Literature T-cell localisations (e.g. Jurkat/primary T-cell imaging papers) are not compiled yet; proposed as a third "
        "held-out set, curated with PMIDs, before v5 is evaluated.",
        "v6 accounting (applied to the v5 baseline too): pre-pass F-actin counts as cortex (PM at light resolution); HPA 'Actin "
        "filaments' adds PM as a supporting (0.3) compartment. Clean hold-outs: opencell and hpa_jurkat_holdout_half. The base "
        "inventory (v0) used the pooled HPA consensus, which includes JURKAT, for all genes: a residual leak for the JURKAT halves."]
    if HOLDOUT_EXCLUDE:
        out["leakage_notes"].append(f"v7+: {sorted(HOLDOUT_EXCLUDE)} excluded from all hold-out sets (rule R4b was motivated by their "
                                    "held-out HPA-JURKAT disagreement; DECISION Pedro 2026-11-10)")
        out["excluded_genes"] = sorted(HOLDOUT_EXCLUDE)
    json.dump(out, open(f"{OUT}/loc_holdout_baseline.json" if VER == "v4" else f"{OUT}/loc_holdout{SFX}.json", "w"), indent=1)
    log(json.dumps({k: {m: v[m]["agree"] for m in ("intended", "placed") if m in v} | {"n": v["n_genes"], "pg": v["mass_pg"]}
                    for k, v in out["sets"].items()}, indent=1))
    return out


# ----------------------------------------------------------------- report
def report():
    T = pd.read_csv(f"{OUT}/loc_compare.tsv", sep="\t", index_col=0)
    RU = json.load(open(f"{CACHE}/loc_rules.json"))
    HO = json.load(open(f"{OUT}/loc_holdout_baseline.json")) if os.path.exists(f"{OUT}/loc_holdout_baseline.json") else None
    tot = T.mass_pg.sum(); L = []
    L.append("# Localisation comparison, pack v4 (Tier 1, annotation level)\n")
    L.append(f"Code: `jurkat/validate/loc_compare.py` (steps fetch, celllines, jurkat, model, compare, holdout, report). "
             f"Run {time.strftime('%Y-%m-%d')}. Per-gene table `loc_compare.tsv`, mapping `loc_mapping.tsv`, "
             f"HPA cell lines `hpa_celllines.tsv`, hold-out baseline `loc_holdout_baseline.json`.\n")
    L.append("## Method\n")
    L.append(f"- **Model.** {len(T):,} genes, {tot:.1f} pg. Gene copies = Σ over ingredients of copies × stoichiometry "
             "(histone families of the nucleosome split over their inventory members). Intended region per comp_weights key follows "
             "`jwc.step_counts` (TM ingredient with a membrane library → `TMR`, else `SOL`; lumens only with a signal peptide); "
             "pre-pass copies keep their pre-pass region. Placed = intended × (placed / requested) per (library, region) over the "
             "7,485 v4 tiles (job arrays only), plus the pre-pass. Regions are collapsed to 8 comparison compartments "
             "(PM, cytosol, nucleus, NE, ER, mito, golgi, endolysosomal).")
    L.append("- **Sources.** HPA consensus main (1) / additional (0.3) with reliability weight Enhanced 1, Supported 0.8, Approved 0.5, "
             f"Uncertain 0; sperm-only terms dropped. UniProt subcellular location (each statement = main; weight {W_UNIPROT}); "
             f"GO-CC (support only, weight {W_GO}). OpenCell and the HPA JURKAT annotations are held out (not in the fit score).")
    L.append("- **Score per source** = 0.5 × model share in compartments the source does not annotate + 0.5 × share of the "
             "source's main compartments that hold < 5% of the model's copies (0 = agree, 1 = disjoint). "
             f"**Combined** = reliability-weighted mean over HPA, UniProt, GO. Disagreeing: combined ≥ {DISAGREE}. "
             "Disagreeing mass = combined × gene mass.")
    L.append("- **Cause class** (first match; judged against the source with the largest weighted contribution): "
             "c TM protein in a soluble region; a lipid-anchored/peripheral (UniProt) packed soluble while the source puts it on a "
             "membrane; b small (< 60 kDa packed unit) cytosolic protein without nuclear pool while the source lists the nucleus; "
             "h the gene's own inventory localisation fits better than the assemblies it is built into; i nucleus/cytosol "
             "balance from Itzhak 2016 fractionation; d HPA vs UniProt conflict (model sides with one); g membrane main location "
             "without anchor annotation, packed soluble; e multi-localisation shares; f the rest (single-protein).\n")
    L.append("## Agreement per source (intended localisation)\n")
    L.append("| Source | Genes | Mass (pg) | Score (mean) | Score (mass-weighted) | Top-compartment hit (mass-w.) | Unsupported mass share (mass-w.) |")
    L.append("|---|---|---|---|---|---|---|")
    for k, lab in (("hpa", "HPA (Approved+)"), ("uniprot", "UniProt"), ("go", "GO-CC"), ("opencell", "OpenCell (held out)"),
                   ("jurkat", "HPA JURKAT (held out)")):
        c = f"{k}_score"
        if c not in T: continue
        m = T[c].notna()
        if k == "hpa": m &= T.hpa_reliability.isin(["Enhanced", "Supported", "Approved"])
        w = T.mass_pg[m]
        L.append(f"| {lab} | {m.sum():,} | {w.sum():.1f} | {T[c][m].mean():.3f} | {(T[c][m] * w).sum() / w.sum():.3f} | "
                 f"{(T[f'{k}_tophit'][m] * w).sum() / w.sum():.3f} | {(T[f'{k}_unsup'][m] * w).sum() / w.sum():.3f} |")
    m = T.combined_score.notna()
    L.append(f"\nCombined fit score: mean {T.combined_score[m].mean():.3f}, mass-weighted "
             f"{(T.combined_score[m] * T.mass_pg[m]).sum() / T.mass_pg[m].sum():.3f}; disagreeing genes "
             f"{(T.combined_score >= DISAGREE).sum():,} ({T.mass_pg[T.combined_score >= DISAGREE].sum():.1f} pg); total disagreeing "
             f"mass {RU['baseline_disagree_mass_pg']:.2f} pg. Placed instead of intended: mass-weighted "
             f"{(T.combined_score_placed[m] * T.mass_pg[m]).sum() / T.mass_pg[m].sum():.3f}.\n")
    L.append("## Cause clusters\n")
    L.append("| Class | Proteins | Gene mass (pg) | Disagreeing mass (pg) | Top genes |")
    L.append("|---|---|---|---|---|")
    g = T.groupby("cause").agg(n=("mass_pg", "size"), mass=("mass_pg", "sum"), dm=("disagree_mass_pg", "sum")).sort_values("dm", ascending=False)
    for c, r in g.iterrows():
        top = ", ".join(T[T.cause == c].head(10).index) if c not in ("agree", "no_fit_source") else ""
        L.append(f"| {c} | {int(r.n):,} | {r.mass:.2f} | {r.dm:.2f} | {top} |")
    f = T[T.tm_in_soluble_flag.astype(bool)]
    L.append(f"\nTM proteins in soluble regions regardless of score (hidden at compartment level, e.g. TOMM20 in the matrix): "
             f"{len(f)} genes, {f.mass_pg.sum():.2f} pg; {RU['R3_note']['tm_ingredients_in_soluble']} TM ingredients "
             f"({RU['R3_note']['tm_ingredients_in_soluble_pg']:.2f} pg), all without a membrane library.\n")
    L.append("## Proposed rule-level fixes for v5 (counterfactual on the intended distribution)\n")
    L.append(f"Genes with a v4 curated localisation ({len(RU.get('curated_v4_excluded', []))}: "
             f"{', '.join(RU.get('curated_v4_excluded', [])[:20])} …) are excluded from every rule except R2/R2b (which add a nuclear pool on top).\n")
    L.append("Δ disagreeing mass is measured against the fit sources (HPA, UniProt, GO), which also built the inventory, so it is an "
             "upper bound on real improvement; the held-out sets decide after the repack.\n")
    L.append("| Rule | Proteins | Mass moved (pg) | Δ disagreeing mass (pg) | Disagreeing genes before → after | Improved / worsened |")
    L.append("|---|---|---|---|---|---|")
    for k, v in RU.items():
        if not isinstance(v, dict) or "mass_moved_pg" not in v: continue
        L.append(f"| **{k}**: {v['desc']} | {v['proteins']:,} | {v['mass_moved_pg']:.2f} | {v['delta_disagree_mass_pg']:+.2f} | "
                 f"{v['disagreeing_before']} → {v['disagreeing_after']} | {v['improved']} / {v['worsened']} |")
    L.append("")
    for k, v in RU.items():
        if isinstance(v, dict) and v.get("top_genes"): L.append(f"- {k} largest movers: {', '.join(v['top_genes'])}")
    L.append("")
    L.append("| Information only | Proteins | Gene mass | | | |")
    L.append("|---|---|---|---|---|---|")
    for k in ("R4_nucleolus_info", "R5_cortex_info"):
        v = RU[k]; L.append(f"| {k}: {v['desc']} | {v['proteins']} | ({v['gene_mass_pg']:.2f} pg of gene mass) | – | – | – |")
    L.append("")
    R = lambda k: RU[k]
    L.append("## Recommendations for v5 (for Pedro's review; nothing applied)\n")
    L += [
        f"1. **Peripheral placement class, lipid-anchored first** (R1_lipid_only: {R('R1_lipid_only')['proteins']} proteins, "
        f"{R('R1_lipid_only')['mass_moved_pg']:.2f} pg moved; LMNB1/LMNB2 lamina, LCK, Gα, ARF/RAB, RAC1). Place on the membrane "
        "surface (cytosolic / nucleoplasmic side; the lamina as a ~15 nm shell under the INM). Extend to UniProt 'Peripheral membrane "
        f"protein' (full R1: {R('R1_peripheral_class')['proteins']:,} proteins, {R('R1_peripheral_class')['mass_moved_pg']:.2f} pg) only "
        "with a cap (e.g. membrane share ≤ 0.3 unless HPA Enhanced/Supported names the membrane as main), because many are mostly "
        "cytosolic with a minor membrane pool (HSPA8, HMGB1, CFL1, GDI2).",
        f"2. **Cortical shell** (R6: {R('R6_cortical_layer')['proteins']} proteins, {R('R6_cortical_layer')['mass_moved_pg']:.2f} pg; LCP1, "
        "MYH9, CFL1, MYO1G, EZR, IQGAP2). Actin-binding proteins annotated at the PM are cortical; the HPA JURKAT images agree "
        "(LCP1, MYO1G, ADD3, PLS3, ARHGAP45, FNBP1 all 'plasma membrane' in JURKAT). Place a share in cytosol within ~150 nm of the PM.",
        f"3. **TM routing** (R3: {R('R3_tm_to_membrane')['proteins']} proteins, {R('R3_tm_to_membrane')['mass_moved_pg']:.2f} pg; "
        f"{R('R3_note')['tm_ingredients_in_soluble']} TM ingredients without a membrane library). Includes hidden cases "
        "(TOMM20/TOM complex in the matrix scores 'agree' at compartment level) and intramembrane hairpins (RTN3). Build membrane "
        "libraries or route via TMR.",
        f"4. **Size-dependent nuclear pool**: R2b (HPA/UniProt list the nucleus) is safe: {R('R2b_size_nuclear_pool_annotated')['proteins']} "
        f"proteins, {R('R2b_size_nuclear_pool_annotated')['mass_moved_pg']:.2f} pg, Δ {R('R2b_size_nuclear_pool_annotated')['delta_disagree_mass_pg']:+.2f} pg. "
        f"R2 for all small cytosolic proteins ({R('R2_size_nuclear_pool_all')['proteins']:,} proteins, {R('R2_size_nuclear_pool_all')['mass_moved_pg']:.2f} pg) "
        "worsens annotation agreement because HPA does not annotate minor nuclear pools; decide it with Tier 2 N/C image ratios. Exclude "
        "actin and tubulin (exportin-6 / polymer pools) and annotated NES proteins.",
        f"5. **Fractionation N/C cap** (R7: {R('R7_fractionation_nc_cap')['proteins']} proteins, {R('R7_fractionation_nc_cap')['mass_moved_pg']:.2f} pg; "
        "TKT, HMGB1, SET, UBA1, RAN, TRIM28, MCM2-7). Itzhak 2016 HeLa 1K-pellet fractionation leaks soluble nuclear proteins to the "
        "cytosolic fraction. The rule is HPA-driven, so its Δ is circular; check on OpenCell/JURKAT after the repack. R2 + R7 together "
        "move up to ~7 pg into the nucleus, so recheck the nucleus/cytosol density ratio (0.69 in v4).",
        "6. **Nucleolus** (information: "
        f"{R('R4_nucleolus_info')['proteins']} proteins, {R('R4_nucleolus_info')['gene_mass_pg']:.1f} pg with HPA main Nucleoli). Class h "
        "(SSU processome UTP20, BMS1, NOL6, HEATR1, WDR75; free r-proteins) comes from subunit-averaged complex comp_weights. Needs a "
        "nucleolus sub-region in the nucleoplasm and per-complex curated compartments.",
        "7. **Source blending (class d, 665 proteins)**: when HPA is Enhanced/Supported, let HPA main/additional decide the compartments "
        "and UniProt only add compartments HPA lists; keep the 50/50 blend for Approved. Per-protein overrides only for the HPA fit half "
        "and never for JURKAT-imaged genes.",
        "8. **Class g (membrane main without anchor, 935 proteins)**: mostly coats/adaptors and partner-anchored proteins (AP2, COPI, "
        "Gβ via prenylated Gγ), which an assembly-level peripheral rule (GO-CC coated vesicle / heterotrimeric G-protein complex) would "
        "cover, plus probable HPA artefacts for abundant cytosolic proteins (HSP90AB1, ENO1/GAPDH 'Plasma membrane'): review per protein.",
        "9. **HPA consensus contains sperm-only terms** (HSPA8, HSPA6, PGK1: 'Annulus', 'Mid piece'…): rebuild the HPA consensus from "
        "per-line annotations without sperm (`hpa_celllines.tsv`) before using it for fixes.",
        ""]
    L.append("## Top 50 single-protein disagreements (by disagreeing mass; classes d, e, f, g, h, i)\n")
    L.append("| Gene | Mass (pg) | Score | Class (driver) | Model (intended) | HPA main / add (rel.) | UniProt | OpenCell |")
    L.append("|---|---|---|---|---|---|---|---|")
    sp = T[T.cause.str.match(r"^[defghi]_")].head(50)
    for gname, r in sp.iterrows():
        def s_(x): return "" if not isinstance(x, str) else x.replace("|", "/")[:70]
        L.append(f"| {gname} | {r.mass_pg:.3f} | {r.combined_score:.2f} | {r.cause} ({r.driver}) | {s_(r.model_regions)} | "
                 f"{s_(r.hpa_main)} / {s_(r.hpa_add)} ({s_(r.hpa_reliability)}) | {s_(r.uniprot_loc)} | {s_(r.opencell)} |")
    if HO:
        L.append("\n## Hold-out baseline (v4)\n")
        L.append("| Set | Genes | Mass (pg) | Agree intended (unw. / mass-w.) | Agree placed (unw. / mass-w.) | Top hit intended (unw. / mass-w.) |")
        L.append("|---|---|---|---|---|---|")
        for k, v in HO["sets"].items():
            a, b, t = v["intended"]["agree"], v["placed"]["agree"], v["intended"]["tophit"]
            L.append(f"| {k} | {v['n_genes']} | {v['mass_pg']:.1f} | {a['unweighted']:.3f} / {a['mass_weighted']:.3f} | "
                     f"{b['unweighted']:.3f} / {b['mass_weighted']:.3f} | {t['unweighted']:.3f} / {t['mass_weighted']:.3f} |")
        L += ["- " + x for x in HO["leakage_notes"]]
    open(f"{OUT}/loc_rules.md", "w").write("\n".join(L) + "\n")
    log(f"wrote {OUT}/loc_rules.md")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step == "fetch": fetch()
    if step == "celllines": celllines()
    if step == "jurkat": jurkat_images()
    if step in ("model", "all"): model()
    if step in ("compare", "all"): compare()
    if step in ("holdout", "all"): holdout()
    if step in ("report", "all"): report()
