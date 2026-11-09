"""Protein organisation from association data: a tunable hypothesis layer on top of random packing.

Random placement is itself an assumption (no organisation at all). This module replaces it with a weak, data-driven
one, at two scales. Neither scale is calibrated against measured protein positions; the parameters below are set so
the packing stays random-looking but measurably organised.

Large scale (across the 200 nm tiles; network-informed sampling of which proteins go where):
  - STRING (v12, combined score >= STRING_MIN) is split into modules by Louvain clustering (MODULE_RES); modules
    with fewer than MODULE_MIN proteins are dropped. Each protein's membership is the share of its STRING weight
    that goes to each module; a library type (monomer or complex) takes the mean over its subunits.
  - Each module gets a smooth Gaussian random field over the cell (random Fourier features, correlation length
    FIELD_L nm, unit variance). A type's copies are split over tiles in proportion to region volume (or membrane
    area) x exp(BETA * sum_m membership_m * field_m), so proteins of one module become co-enriched in the same parts
    of the cell. Cell totals and compartments are unchanged.

Meso and small scale (within a tile; who sits next to whom):
  - Partners of each library type come from the AlphaFold 3 pool scores (best chain-pair PAE min-min <= POOL_PAE_MAX
    Å: "contact" partners, proposed in contact) and STRING (combined >= STRING_MIN, not already a pool partner:
    "loose" partners, proposed 2-15 nm away). Paralog artefacts and hubs are filtered and down-weighted.
  - During packing, a copy is placed next to an already placed partner (in its tile or the neighbouring halo) with a
    probability that saturates with the partner's local copy number, like mass action:
        a_p = strength_p x A_kind x n_p / (K_HALF + n_p),   P(attract) = min(P_MAX, sum_p a_p)
    The partner is chosen in proportion to a_p. The copy gets ATTRACT_TRIES proposals near it; if none passes the
    collision and compartment checks it falls back to a uniformly random position, as before.
"""
import os, json
import numpy as np
import pandas as pd
from config import DATA, WORK

ODIR = f"{WORK}/organisation"
STRING_DIR = f"{DATA}/string"
STRING_F = f"{STRING_DIR}/4932.protein.links.v12.0.txt.gz"
POOL_SCORES = os.environ.get("YC_POOL_SCORES", "")             # summary_models.parquet of the pool predictions
POOL_PAIRS_F = f"{ODIR}/pool_pairs.parquet"                     # per protein pair: best PAE min-min (cached)

# large scale
STRING_MIN = 700
MODULE_RES = 2.0
MODULE_MIN = 10
FIELD_L = 500.0        # nm
FIELD_NF = 128         # Fourier features per field
BETA = 0.8             # log-density amplitude for a protein fully in one module
# meso and small scale
POOL_PAE_MAX = 8.0     # Å
MAX_PARTNERS = 16
A_CONTACT = 0.6
A_LOOSE = 0.25
K_HALF = 8.0           # copies in the tile (+ halo) at which a partner reaches half its pull
P_MAX = 0.6
ATTRACT_TRIES = 40
LOOSE_GAP = (2.0, 15.0)    # nm, surface gap of loose proposals
SEED = 11
CONTACT, LOOSE = 0, 1


def _uniprot_maps():
    up = pd.read_csv(f"{DATA}/uniprot_yeast.tsv", sep="\t")
    loc2u = {}
    for e, l in zip(up.Entry, up["Gene Names (ordered locus)"].fillna("")):
        for x in str(l).replace(";", " ").split():
            loc2u.setdefault(x, e)
    return loc2u


def string_edges(min_score=STRING_MIN):
    """STRING links as UniProt pairs (a < b), combined score >= min_score"""
    loc2u = _uniprot_maps()
    s = pd.read_csv(STRING_F, sep=" ")
    s = s[s.combined_score >= min_score]
    a = s.protein1.str.split(".", n=1).str[1].map(loc2u); b = s.protein2.str.split(".", n=1).str[1].map(loc2u)
    d = pd.DataFrame(dict(a=a, b=b, score=s.combined_score.values)).dropna()
    d = d[d.a < d.b]
    return d.groupby(["a", "b"], as_index=False).score.max()


def pool_pairs():
    """best (lowest) chain-pair PAE min-min per UniProt pair over all pool and pair predictions; cached"""
    if os.path.exists(POOL_PAIRS_F):
        return pd.read_parquet(POOL_PAIRS_F)
    import pyarrow.parquet as pq, pyarrow as pa, pyarrow.compute as pc
    if not POOL_SCORES:
        raise SystemExit("set YC_POOL_SCORES to the pool predictions' summary_models.parquet")
    t = pq.read_table(POOL_SCORES, columns=["af3_id1", "af3_id2", "chain_pair_pae_min_min"])
    a, b = pc.utf8_upper(t["af3_id1"]), pc.utf8_upper(t["af3_id2"])
    lt = pc.less(a, b)
    t = pa.table(dict(a=pc.if_else(lt, a, b), b=pc.if_else(lt, b, a), pae=t["chain_pair_pae_min_min"]))
    g = t.group_by(["a", "b"]).aggregate([("pae", "min")]).to_pandas().rename(columns={"pae_min": "pae"})
    g = g[g.a != g.b]
    os.makedirs(ODIR, exist_ok=True); g.to_parquet(POOL_PAIRS_F)
    return g


def pool_edges(pae_max=POOL_PAE_MAX):
    """confident pool pairs without paralog artefacts: drop (A, B) if a paralog A' of A scores better with B
    (homologous interfaces, e.g. Pol II subunits against Pol III paralogs)"""
    g = pool_pairs()
    g = g[g.pae <= pae_max].copy()
    par = pd.read_csv(f"{DATA}/alliance_yeast_paralog_pairs.csv", usecols=["gene1_uniprot", "gene2_uniprot"]).dropna()
    para = {}
    for x, y in zip(par.gene1_uniprot, par.gene2_uniprot):
        para.setdefault(x, set()).add(y); para.setdefault(y, set()).add(x)
    best = {(a, b): p for a, b, p in zip(g.a, g.b, g.pae)}
    best.update({(b, a): p for (a, b), p in list(best.items())})
    keep = []
    for a, b, p in zip(g.a, g.b, g.pae):
        bad = any(best.get((a2, b), 99) < p for a2 in para.get(a, ())) or any(best.get((a, b2), 99) < p for b2 in para.get(b, ()))
        keep.append(not bad)
    return g[np.array(keep, bool)]


def lib_subunits(exp3, ing):
    """library type -> set of UniProt accessions of its subunits"""
    u = ing.set_index("ing_id").uniprot
    out = {}
    for ing_id, lib in exp3[["ing_id", "lib_id"]].drop_duplicates().itertuples(index=False):
        acc = u.get(ing_id)
        if isinstance(acc, str):
            out.setdefault(lib, set()).update(x for x in acc.split(";") if x)
    return out


def modules(edges):
    """Louvain modules of the STRING graph; returns (membership DataFrame protein x module, module table)"""
    import networkx as nx
    G = nx.Graph()
    G.add_weighted_edges_from(zip(edges.a, edges.b, (edges.score - (STRING_MIN - 100)) / 400.0))
    comms = sorted(nx.community.louvain_communities(G, weight="weight", resolution=MODULE_RES, seed=SEED), key=len, reverse=True)
    comms = [c for c in comms if len(c) >= MODULE_MIN]
    mod_of = {u: m for m, c in enumerate(comms) for u in c}
    rows = {}
    for u in G.nodes:
        w = np.zeros(len(comms))
        for v, d in G[u].items():
            m = mod_of.get(v)
            if m is not None:
                w[m] += d["weight"]
        if u in mod_of:                       # count itself once, so isolated members keep their own module
            w[mod_of[u]] += max(w.sum(), 1e-9) / max(G.degree(u), 1)
        if w.sum() > 0:
            rows[u] = w / w.sum()
    memb = pd.DataFrame.from_dict(rows, orient="index")
    return memb, [sorted(c) for c in comms]


def field_values(points, n_mod, seed=SEED):
    """unit-variance smooth random fields (correlation length FIELD_L) of n_mod modules at points (nm); n x n_mod"""
    rng = np.random.default_rng(seed)
    K = rng.normal(0, 1.0 / FIELD_L, (n_mod, FIELD_NF, 3)); ph = rng.uniform(0, 2 * np.pi, (n_mod, FIELD_NF))
    return np.sqrt(2.0 / FIELD_NF) * np.cos(np.einsum("pc,mfc->pmf", points, K) + ph[None]).sum(2)


def tile_logmod(lib_ids, subunits, memb, centres):
    """per library type, the log weight over tiles (centres, nm): BETA * membership . fields"""
    F = field_values(centres, memb.shape[1])
    out = {}
    for lib in lib_ids:
        us = [u for u in subunits.get(lib, ()) if u in memb.index]
        if us:
            s = memb.loc[us].values.mean(0)
            out[lib] = BETA * (F @ s)
    return out


def partners(lib_ids, subunits, pool, string):
    """partner lists per library type for the packer, as CSR arrays over lp['ids'] indices:
    off (n+1), idx, strength (0..1, hub down-weighted), kind (CONTACT / LOOSE)"""
    lib_ids = [str(x) for x in lib_ids]
    by_u = {}
    for i, lib in enumerate(lib_ids):
        for u in subunits.get(lib, ()):
            by_u.setdefault(u, []).append(i)
    # protein-level strengths: pool contact 1 at PAE <= 2 Å to 0 at POOL_PAE_MAX; STRING loose 0.2..1 over 700..1000
    E = {}
    for a, b, p in zip(pool.a, pool.b, pool.pae):
        E[(a, b)] = (CONTACT, float(np.clip((POOL_PAE_MAX - p) / (POOL_PAE_MAX - 2.0), 0.05, 1.0)))
    for a, b, s in zip(string.a, string.b, string.score):
        if (a, b) not in E:
            E[(a, b)] = (LOOSE, float(np.clip(0.2 + 0.8 * (s - STRING_MIN) / (1000 - STRING_MIN), 0.2, 1.0)))
    deg = {}
    for a, b in E:
        deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    P = [dict() for _ in lib_ids]
    for (a, b), (kind, s) in E.items():
        if a not in by_u or b not in by_u:
            continue
        s = s / np.sqrt(max(1.0, max(deg[a], deg[b]) / 20.0))          # hubs pull less
        for i in by_u[a]:
            for j in by_u[b]:
                if i == j or subunits[lib_ids[i]] & subunits[lib_ids[j]]:
                    continue
                for x, y in ((i, j), (j, i)):
                    old = P[x].get(y)
                    if old is None or (kind, -s) < (old[0], -old[1]):     # contact beats loose, then stronger
                        P[x][y] = (kind, s)
    off = [0]; idx = []; st = []; kd = []
    for i in range(len(lib_ids)):
        top = sorted(P[i].items(), key=lambda kv: (kv[1][0], -kv[1][1]))[:MAX_PARTNERS]
        for j, (kind, s) in top:
            idx.append(j); st.append(s); kd.append(kind)
        off.append(len(idx))
    return dict(off=np.array(off, np.int64), idx=np.array(idx, np.int64), strength=np.array(st, np.float64),
                kind=np.array(kd, np.int64))


def params():
    return np.array([A_CONTACT, A_LOOSE, K_HALF, P_MAX, ATTRACT_TRIES, LOOSE_GAP[0], LOOSE_GAP[1]], np.float64)


def build(exp3, ing, lib_ids, centres, log=print):
    """everything the packing needs: tile log weights per library type and partner CSR arrays; also saved to ODIR"""
    os.makedirs(ODIR, exist_ok=True)
    subunits = lib_subunits(exp3, ing)
    st = string_edges()
    memb, comms = modules(st)
    logmod = tile_logmod(lib_ids, subunits, memb, centres)
    pool = pool_edges()
    pr = partners(lib_ids, subunits, pool, st)
    np.savez(f"{ODIR}/partners.npz", ids=np.array([str(x) for x in lib_ids]), **pr)
    memb.to_parquet(f"{ODIR}/module_membership.parquet")
    json.dump(dict(modules=comms, params=dict(STRING_MIN=STRING_MIN, MODULE_RES=MODULE_RES, FIELD_L=FIELD_L, BETA=BETA,
                   POOL_PAE_MAX=POOL_PAE_MAX, A_CONTACT=A_CONTACT, A_LOOSE=A_LOOSE, K_HALF=K_HALF, P_MAX=P_MAX)),
              open(f"{ODIR}/modules.json", "w"))
    npart = np.diff(pr["off"])
    log(f"organisation: {len(st)} STRING links >= {STRING_MIN}, {len(comms)} modules ({sum(map(len, comms))} proteins); "
        f"{len(logmod)} of {len(lib_ids)} types modulated over tiles; {len(pool)} pool pairs (PAE <= {POOL_PAE_MAX:g}); "
        f"{(npart > 0).sum()} types with partners (median {np.median(npart[npart > 0]) if (npart > 0).any() else 0:.0f}; "
        f"{(pr['kind'] == CONTACT).sum()} contact, {(pr['kind'] == LOOSE).sum()} loose links)")
    return logmod, pr


MODULE_DOMINANT = 0.3      # a type is coloured by its strongest module if that holds at least this share


def encode_viewer(out_dir, log=print):
    """library/organisation.json for the viewer: modules (label: the three most abundant members, size) and per type
    index its dominant module (types without one are left out)"""
    tj = json.load(open(f"{out_dir}/library/types.json"))["types"]
    memb = pd.read_parquet(f"{ODIR}/module_membership.parquet")
    comms = json.load(open(f"{ODIR}/modules.json"))["modules"]
    gene, copies = {}, {}
    for t in tj:
        us = [u for u in (t["uniprot"] or "").split(";") if u]
        if len(us) == 1:
            gene[us[0]] = t["gene"]; copies[us[0]] = copies.get(us[0], 0) + t["copies"]
    mods = [dict(label=" ".join(gene.get(u, u) for u in sorted(c, key=lambda u: -copies.get(u, 0))[:3]), n=len(c)) for c in comms]
    types = {}
    for i, t in enumerate(tj):
        us = [u for u in (t["uniprot"] or "").split(";") if u in memb.index]
        if us:
            s = memb.loc[us].values.mean(0)
            if s.max() >= MODULE_DOMINANT:
                types[str(i)] = int(s.argmax())
    json.dump(dict(modules=mods, types=types), open(f"{out_dir}/library/organisation.json", "w"), separators=(",", ":"))
    log(f"organisation.json: {len(mods)} modules, {len(types)} of {len(tj)} types with a dominant module")
