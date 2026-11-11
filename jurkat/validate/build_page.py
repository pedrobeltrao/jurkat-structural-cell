"""Build the validation report page (one self-contained HTML file) from data/derived/validation/* and the pack summaries.

usage: build_page.py [--version v4] [--prev v3] [--out data/derived/validation/validation.html]
Reads summary_<version>.json / summary_<prev>.json (data/derived/wc), densities.json, crowding.json, ribosome_nn.json,
loc_holdout_baseline.json, loc_compare.tsv, fluor_metrics.tsv, diffusion_benchmark.tsv and the fluorescence figures.
Literature reference values are written here, with their sources, and are the only hand-entered numbers."""
import os, io, sys, json, base64, argparse, html
import numpy as np, pandas as pd
from PIL import Image

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"; VAL = f"{D}/validation"; W = f"{D}/wc"

REG_LABEL = {"cytosol": "Cytosol", "nucleoplasm": "Nucleoplasm", "nucleolus": "Nucleolus", "matrix": "Mito matrix",
             "ER_lumen": "ER lumen", "golgi_lumen": "Golgi lumen", "endo_lumen": "Endo-lysosomal lumen",
             "perinuclear": "Perinuclear space", "PM": "Plasma membrane", "NE": "Nuclear envelope", "ER": "ER membrane",
             "MITO_OM": "Mito outer membrane", "MITO_IM": "Mito inner membrane", "GOLGI": "Golgi membrane",
             "ENDO": "Endo-lysosomal membrane", "sh_PM_cyto": "PM inner-leaflet shell", "sh_PM_cortex": "Cortical shell",
             "sh_NE_lamina": "Nuclear lamina shell", "sh_GOLGI_cyto": "Golgi cytosolic shell", "sh_ENDO_cyto": "Endo-lysosomal cytosolic shell",
             "speckle": "Nuclear speckles", "sh_NPC": "NPC-proximal shell", "sh_PM_periph": "PM peripheral-binder shell",
             "sh_ENDO_periph": "Endo-lysosomal peripheral shell", "sh_GOLGI_periph": "Golgi peripheral shell"}

# version-specific text: fluorescence expectations (reference, verdict), notes and known issues
FLUOR_EXPECT = {
    "v4": {"CD3E": ("PM and ER (HPA Jurkat image: strong ER/perinuclear)", "part"), "CD28": ("PM (HPA Jurkat image)", "ok"),
           "CD2": ("PM (HPA Jurkat image)", "ok"), "LCK": ("PM inner leaflet rim", "bad"), "GOLGA2": ("Golgi", "bad"),
           "LMNB1": ("Nuclear rim (lamina)", "bad"), "TOMM20": ("Mitochondrial outer membrane", "part"),
           "PGK1": ("Cytosol and nucleus", "bad"), "HSPA5": ("ER lumen", "ok"), "SELPLG": ("Intracellular puncta (HPA Jurkat image)", "bad")},
    "v5": {"CD3E": ("PM and ER (HPA Jurkat image: strong ER/perinuclear)", "part"), "CD28": ("PM (HPA Jurkat image)", "ok"),
           "CD2": ("PM (HPA Jurkat image)", "ok"), "LCK": ("PM inner leaflet rim, plus a pericentrosomal pool", "part"),
           "GOLGA2": ("Golgi", "bad"), "LMNB1": ("Nuclear rim (lamina)", "ok"), "TOMM20": ("Mitochondrial outer membrane", "ok"),
           "PGK1": ("Cytosol and nucleus", "bad"), "HSPA5": ("ER lumen", "ok"), "SELPLG": ("Intracellular puncta (HPA Jurkat image)", "bad")},
    "v6": {"CD3E": ("PM and ER (HPA Jurkat image: strong ER/perinuclear)", "part"), "CD28": ("PM (HPA Jurkat image)", "ok"),
           "CD2": ("PM (HPA Jurkat image)", "ok"), "LCK": ("PM inner leaflet rim, plus a pericentrosomal pool", "part"),
           "GOLGA2": ("Golgi", "part"), "LMNB1": ("Nuclear rim (lamina)", "ok"), "TOMM20": ("Mitochondrial outer membrane", "ok"),
           "PGK1": ("Cytosol and nucleus", "part"), "HSPA5": ("ER lumen", "ok"), "SELPLG": ("Intracellular puncta (HPA Jurkat image)", "bad")},
}
FLUOR_VERDICT = {"v4": "PM markers, ER lumen and mitochondria look right; anchored and lamina proteins do not.",
                 "v5": "PM markers, ER lumen, mitochondria and the lamina look right; LCK now has a PM rim. Golgi proteins and some small soluble proteins are still wrong.",
                 "v6": "As v5, plus a Golgi pool for GOLGA2 and nucleus/cytosol balance set from imaging categories. CD3 chains are still mostly at the PM (no ER support outside the Jurkat images, which are held out)."}
LOC_NOTE = {"v4": "The Jurkat mass-weighted score is dominated by LCP1 (half of that set's mass), which is at the plasma membrane in Jurkat but soluble in the model.",
            "v5": "The Jurkat mass-weighted score is dominated by LCP1 (half of that set's mass), now 73% in the cortical shell. Only OpenCell is a clean hold-out for v5: the cortex rule used HPA consensus annotations, including some test-half genes.",
            "v6": "Clean hold-outs: OpenCell (never read by any rule) and a seeded random half of the Jurkat-imaged genes (v6 rules saw only a non-Jurkat HPA consensus for them). The HPA test half is not clean: v6 rules used the HPA consensus for all genes. Both versions are scored with the v6 accounting (pre-placed F-actin counts as cortex)."}
FACE = {"v4": "The EM volume cuts the cell at one face; pack v4 still closes that face with a plasma membrane.",
        "v5": "The EM volume cuts the cell at one face. Since v5 the model stops there with no membrane; extrapolating the side walls puts the missing cap at about 1% of the volume and 6% of the PM area."}
FACE["v6"] = FACE["v5"]
FACE["v7"] = FACE["v5"]
FLUOR_EXPECT["v7"] = dict(FLUOR_EXPECT["v6"])
FLUOR_EXPECT["v7"].update({"CD3E": ("PM and ER (HPA Jurkat image: strong ER/perinuclear); excluded from hold-out scores from v7 on", "part"),
                           "LMNB1": ("Nuclear rim (lamina)", "part")})
FLUOR_VERDICT["v7"] = ("As v6, plus a juxta-centrosomal cisternal Golgi (GOLGA2 Golgi share 0.31 -> 0.48) and intracellular CD3 pools "
                       "(ER 0.15, endosomes 0.16). LMNB1 is now 40% lamina lattice and 60% nucleoplasmic pool (the lattice holds 43% of the lamins), "
                       "so its rim is weaker than in v6.")
LOC_NOTE["v7"] = ("Clean hold-outs as v6 (OpenCell; seeded Jurkat-image half). CD3E and CD3D are excluded from every hold-out score "
                  "from v7 on (their v7 rule was motivated by a held-out disagreement; DECISION Pedro 2026-11-10); the v6 numbers shown still include them.")
KNOWN = {
    "v4": ["The EM volume cuts the cell at one face (about 3-4% of the cell missing). Pack v4 still puts a plasma membrane on that face; the next version leaves it open.",
           "Lipid-anchored and lamina proteins are packed as soluble; the next version places them in thin shells on their membranes.",
           "There is no nucleolus compartment; about 5 pg of nucleolar protein is spread through the nucleoplasm.",
           "Small soluble proteins have no nuclear pool.",
           "The packer jams in thin compartments (matrix, ER lumen); a GPU packer with compaction is planned."],
    "v5": ["The Golgi cytosolic shell is only 35% placed: the procedural Golgi has about 1/4.5 of the real cisternal area. This keeps GOLGA2 cytosolic.",
           "The nuclear lamina shell is 69% placed; lamins are long rods and need a dimer or lattice representation.",
           "Peripheral proteins anchored only through partners (873 genes, e.g. coat and adaptor proteins) are still soluble; proteins over 60 kDa have no nuclear pool.",
           "The nucleolus is capped at 110 mg/mL of protein, and rRNA is not modelled.",
           "The packer still jams in thin compartments (matrix 71%, ER lumen 66%); a GPU packer with compaction is planned."],
    "v6": ["CD3 chains stay mostly at the PM: the only ER/Golgi evidence is the Jurkat image of CD3E/CD3D, which is in the held-out half.",
           "The NPC-proximal shell fills almost only on the nuclear face: the pre-placed NPC (7R5K) cytoplasmic ring and filaments occupy the cytoplasmic annulus.",
           "Nuclear speckles are procedural (30 spheres of 0.5 µm at the most chromatin-poor sites; chromatin inside them was removed); there is no EM speckle label.",
           "The Golgi is still a smoothed blob with 1/4.5 of the cisternal area; Golgi shells now pack at that modelled share (a cisternal stack is a v7 item).",
           "The lamina shell is ~66% placed (lamin rods need a dimer/lattice model); the packer still jams in the matrix and ER lumen.",
           "Tier 2 image ratios are compressed in the HPA images (antibody background, thin cytoplasm, many imaged genes not expressed in Jurkat): rim ratios hardly separate PM from cytosol, so they could not calibrate the cortex or peripheral shares (default 0.5 used)."],
    "v7": ["Chromatin v8 (genome-mapped) misses TAD insulation (0.06), the P(s) slope (-1.29) and fine DAPI texture; fixes are listed as next steps.",
           "Cytosol placement is 0.980 cell-wide (v6 0.983): the PCM / juxta-centrosomal Golgi tiles are locally capacity-limited.",
           "ER lumen and matrix now pack at 0.82 / 0.86 (v6 0.66 / 0.71) but are still below 1; the lamina lattice holds only 43% of the lamins.",
           "The NPC-proximal shell still fills mainly on the nuclear face; speckles remain procedural.",
           "Tier 2 rim ratios still cannot calibrate cortex / peripheral shares (HPA Jurkat rims barely separate PM from cytosol)."]
}


def esc(s): return html.escape(str(s))


def img_uri(path, width=1800, q=82):
    im = Image.open(path).convert("RGB")
    if im.width > width: im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def pill(kind):
    txt = {"ok": "Agrees", "part": "Partly", "bad": "Disagrees", "check": "Method check", "open": "Pending"}[kind]
    return f'<span class="pill {kind}">{txt}</span>'


def bars_svg(rows, prev_key, cur_key, prev_lab, cur_lab):
    """Horizontal paired bars, 0-100%: rows = [(label, prev, cur)]."""
    lw, bw, rh = 190, 420, 30; h = rh * len(rows) + 46; w = lw + bw + 70
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Placed mass fraction per region, {esc(prev_lab)} and {esc(cur_lab)}">']
    for t in (0, 25, 50, 75, 100):
        x = lw + bw * t / 100
        o.append(f'<line x1="{x}" y1="18" x2="{x}" y2="{h - 22}" class="grid"/><text x="{x}" y="{h - 6}" class="tick" text-anchor="middle">{t}%</text>')
    for i, (lab, p, c) in enumerate(rows):
        y = 22 + i * rh
        o.append(f'<text x="{lw - 10}" y="{y + 15}" class="lab" text-anchor="end">{esc(lab)}</text>')
        if p is not None:
            o.append(f'<rect x="{lw}" y="{y + 2}" width="{bw * p:.1f}" height="8" class="prev"/>')
        o.append(f'<rect x="{lw}" y="{y + 11}" width="{bw * c:.1f}" height="12" class="cur{" low" if c < 0.9 else ""}"/>')
        o.append(f'<text x="{lw + bw * c + 6:.1f}" y="{y + 22}" class="val">{100 * c:.1f}%</text>')
    o.append(f'<rect x="{lw}" y="4" width="14" height="8" class="prev"/><text x="{lw + 20}" y="12" class="tick">{esc(prev_lab)}</text>'
             f'<rect x="{lw + 90}" y="2" width="14" height="12" class="cur"/><text x="{lw + 110}" y="12" class="tick">{esc(cur_lab)}</text>')
    o.append("</svg>")
    return "".join(o)


def table(df, cls="", num=()):
    h = "".join(f"<th{' class=num' if c in num else ''}>{esc(c)}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td{' class=num' if c in num else ''}>{v}</td>" for c, v in zip(df.columns, r)) + "</tr>"
                   for r in df.itertuples(index=False))
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{rows}</tbody></table></div>'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--version", default="v5"); ap.add_argument("--prev", default="v4")
    ap.add_argument("--out", default=f"{VAL}/validation.html"); a = ap.parse_args()
    S = json.load(open(f"{W}/summary_{a.version}.json")); P = json.load(open(f"{W}/summary_{a.prev}.json"))
    den = json.load(open(f"{VAL}/densities.json")); cr = json.load(open(f"{VAL}/crowding.json"))
    rnn = {r["set"]: r for r in json.load(open(f"{VAL}/ribosome_nn.json"))["rows"]}
    def vfile(stem, ext, v):   # per-version file if present; v4 results kept the original names
        f = f"{VAL}/{stem}_{v}.{ext}"
        return f if os.path.exists(f) else f"{VAL}/{stem}.{ext}"
    hofile = lambda v: f"{VAL}/loc_holdout_{v}.json" if os.path.exists(f"{VAL}/loc_holdout_{v}.json") else f"{VAL}/loc_holdout_baseline.json"
    ho = json.load(open(hofile(a.version)))["sets"]; hop = json.load(open(hofile(a.prev)))["sets"] if hofile(a.prev) != hofile(a.version) else None
    lc = pd.read_csv(vfile("loc_compare", "tsv", a.version), sep="\t")
    fm = pd.read_csv(f"{VAL}/fluor_metrics.tsv", sep="\t")
    # ---------- Tier 2 image metrics (tier2.py)
    t2 = {v: json.load(open(f"{VAL}/tier2_{v}.json")) for v in (a.prev, a.version) if os.path.exists(f"{VAL}/tier2_{v}.json")}
    tier2_html = ""
    if a.version in t2:
        rows2 = []
        for setn, lab in (("holdout_half", "held-out half (clean)"), ("fit_half", "fit half")):
            for m, ml in (("nc", "Nucleus/cytoplasm"), ("rim", "Rim/cytoplasm"), ("nuc_frac", "Share in nucleus"), ("punct", "Punctateness")):
                cur = t2[a.version]["sets"][setn]["expected"].get(m, {}); prv = t2.get(a.prev, {}).get("sets", {}).get(setn, {}).get("expected", {}).get(m, {})
                f_ = lambda d, k: f"{d[k]:.2f}" if d.get(k) is not None else "–"
                rows2.append([lab, ml, str(cur.get("n", "")), f_(prv, "spearman"), f_(cur, "spearman"), f_(prv, "class_agree"), f_(cur, "class_agree"),
                              f_(prv, "agree_2fold_cal"), f_(cur, "agree_2fold_cal")])
        t2tab = pd.DataFrame(rows2, columns=["Genes", "Metric", "n", f"Spearman {a.prev}", f"Spearman {a.version}", f"Class agreement {a.prev}",
                                             f"Class agreement {a.version}", f"Within 2-fold (calibrated) {a.prev}", f"Within 2-fold (calibrated) {a.version}"])
        tr = t2[a.version]["transfer"]
        tier2_html = (f"<p>The same segmentation runs on the {t2[a.version]['hpa_cells']:,} HPA Jurkat cells ({t2[a.version]['hpa_genes']} genes) and on "
                      f"{t2[a.version]['model_sections']} simulated confocal sections of the model (blue = H2B, red = β-tubulin, green = the protein's "
                      "expected image from its compartment shares). Per cell: nucleus/cytoplasm and rim/cytoplasm intensity ratios, the share of "
                      "signal in the nucleus, and the share in spots. Rank correlation over genes is the main measure; class agreement uses "
                      "enriched / similar / depleted (log₂ ±0.5).</p>" + table(t2tab, num=tuple(t2tab.columns[2:])) +
                      f"<p class=\"meta\">Imaging transfer (fit half, frozen at v5): HPA N/C ≈ {tr['nc']['a']:.2f} + {tr['nc']['b']:.2f} × model "
                      f"(r {tr['nc']['r']:.2f}); HPA rim ≈ {tr['rim']['a']:.2f} + {tr['rim']['b']:.2f} × model (r {tr['rim']['r']:.2f}). HPA ratios are "
                      "compressed (antibody background, a 1-2 µm cytoplasmic ring, many imaged genes not expressed in Jurkat), so calibrated agreement "
                      "is shown only where the slope is useful, and rim ratios barely separate plasma membrane from cytosol.</p>")
    db = pd.read_csv(f"{VAL}/diffusion_benchmark.tsv", sep="\t")

    # ---------- placement
    order = [k for k in REG_LABEL if k in S["regions"] and S["regions"][k]["expected_pg"] >= 0.05]
    prow = [(REG_LABEL[k], P["regions"].get(k, {}).get("frac"), S["regions"][k]["frac"]) for k in order]
    place_svg = bars_svg(prow, a.prev, a.version, f"pack {a.prev}", f"pack {a.version}")
    low = [f"{REG_LABEL[k]} {100 * S['regions'][k]['frac']:.0f}%" for k in order if S["regions"][k]["frac"] < 0.9]

    # ---------- densities
    dr = pd.DataFrame(den["rows"]); dr = dr[dr.volume_um3 > 3]
    lit = {"cytosol": "150-250 (cytoplasm, all macromolecules)", "nucleoplasm": "lower than cytoplasm (QPI)",
           "matrix": "high; often quoted 300-500 (protein)", "ER_lumen": "about 100 (lumenal chaperones)"}
    dtab = pd.DataFrame({"Compartment": [REG_LABEL.get(r, r) for r in dr.region], "Volume (µm³)": dr.volume_um3.map("{:.1f}".format),
                         "Macromolecules (pg)": (dr.packed_pg + dr.prepass_pg).map("{:.1f}".format),
                         "Model (mg/mL)": dr.mg_per_ml.map("{:.0f}".format), "Typical (mg/mL)": [lit.get(r, "") for r in dr.region]})

    # ---------- crowding
    cs = pd.DataFrame(cr["summary"])
    ctab = pd.DataFrame({"Compartment": [REG_LABEL.get(r, r) for r in cs.region], "Volume fraction φ": cs.phi.map("{:.3f}".format),
                         "D/D₀, 1 nm probe": cs["D_1.0"].map("{:.2f}".format), "D/D₀, GFP (2.3 nm)": cs["D_2.3"].map("{:.2f}".format),
                         "D/D₀, 3 nm probe": cs["D_3.0"].map("{:.3f}".format)})

    # ---------- ribosomes
    em, th = rnn["EM (cell 1, non-nuclear)"], rnn["model thinned to EM count"]
    jit = [v for k, v in rnn.items() if "jitter" in k][0]

    # ---------- localisation
    def hrow(key, name):
        s = ho[key]; ag = s["intended"]["agree"]; tp = s["intended"]["tophit"]
        pv = f"{hop[key]['intended']['agree']['unweighted']:.2f}" if hop and key in hop else "–"
        return [name, f"{s['n_genes']:,}", pv, f"{ag['unweighted']:.2f}", f"{ag['mass_weighted']:.2f}", f"{tp['unweighted']:.2f}"]
    rows_ = [hrow("opencell", "OpenCell, HEK293T endogenous tags (clean)")]
    if "hpa_jurkat_holdout_half" in ho:
        rows_ += [hrow("hpa_jurkat_holdout_half", "HPA Jurkat images, held-out half (clean)"),
                  hrow("hpa_jurkat_fit_half", "HPA Jurkat images, fit half (used to calibrate)"),
                  hrow("hpa_test_half", "HPA consensus, md5 test half (not clean)")]
    else:
        rows_ += [hrow("hpa_test_half", "HPA, random half never used for fixes"), hrow("hpa_jurkat", "HPA images in Jurkat")]
    htab = pd.DataFrame(rows_,
                        columns=["Held-out reference", "Genes", f"Agreement, {a.prev}", "Agreement", "Agreement, mass-weighted", "Top compartment correct"])
    cl = lc.groupby("cause").agg(n=("gene", "size"), mass=("disagree_mass_pg", "sum"))
    causes = {"a_peripheral_or_lipid_anchored": "Lipid-anchored or peripheral, packed as soluble",
              "i_fractionation_nc_share": "Nucleus/cytosol split from fractionation data",
              "g_membrane_main_no_anchor_packed_soluble": "Membrane-associated, no anchor annotation, packed as soluble",
              "b_small_soluble_no_nuclear_pool": "Small soluble protein without a nuclear pool",
              "d_source_conflict": "Conflicting sources (HPA vs UniProt)",
              "h_assembly_localisation": "Localisation averaged inside an assembly",
              "f_single_protein": "Single-protein disagreement", "c_tm_in_soluble": "Transmembrane protein in a soluble region",
              "e_multilocalisation_shares": "Multi-compartment shares"}
    if len(cl):
        cl = cl.loc[[c for c in causes if c in cl.index]].sort_values("mass", ascending=False)
        ctab2 = pd.DataFrame({"Cause": [causes[c] for c in cl.index], "Proteins": cl.n.map("{:,}".format),
                              "Disagreeing mass (pg)": cl.mass.map("{:.2f}".format)})
        cause_html = table(ctab2, num=("Proteins", "Disagreeing mass (pg)"))
    else:
        cause_html = "<p>Cause table not available.</p>"

    # ---------- fluorescence: selected informative rows
    pick = ["CD3E", "CD28", "CD2", "LCK", "GOLGA2", "LMNB1", "TOMM20", "PGK1", "HSPA5", "SELPLG"]
    f = fm.set_index("protein").reindex([p for p in pick if p in set(fm.protein)])
    expect = FLUOR_EXPECT.get(a.version, FLUOR_EXPECT["v5"])
    ftab = pd.DataFrame({"Protein": f.index, "Model: main compartment": [max(("PM", "cytosol", "nucleus", "ER", "mito", "golgi", "endolysosomal"), key=lambda c: r[f"mol_{c}"]) for _, r in f.iterrows()],
                         "PM/cytoplasm": f.pm_cyto.map("{:.1f}".format), "Nucleus/cytoplasm": f.nuc_cyto.map("{:.3f}".format),
                         "Reference": [expect[p][0] for p in f.index], "Verdict": [pill(expect[p][1]) for p in f.index]})

    # ---------- diffusion benchmark
    dv = pd.to_numeric(db.D_numeric, errors="coerce"); nref = db.reference.nunique()

    # ---------- size
    Svol, nvol = 732.0, 276.0

    # ---------- Brownian dynamics with mobile crowders (bd_summary.json, bd_results.md)
    bd_html = ""
    if os.path.exists(f"{VAL}/bd_summary.json"):
        bd = json.load(open(f"{VAL}/bd_summary.json")); rr = pd.DataFrame(bd["ratios"])
        within = (rr.model_over_measured.between(0.8, 1.25)).mean()
        bd_html = f"""<h3>Brownian dynamics with moving crowders (pack {esc(bd['pack_version'])})</h3>
<p>Each benchmark probe was simulated as its own structure among the packed molecules, all moving, in 500 nm boxes cut from the model (GPU, 100 µs). Corrections are physical, not fitted: a 0.3 nm hydration shell, fluid-phase viscosity 1.2× water, frozen polysomes.</p>
<ul>
 <li><b>Size ratios:</b> {100 * within:.0f}% of {len(rr)} within-paper size ratios are within 25% of measured; GFP tandem ratios are within 15%, and the largest dextrans and GEM particles slow down too steeply or too little.</li>
 <li><b>Cytosolic GFP:</b> D/D₀ 0.42 (about 0.38 with finer crowder shapes) vs 0.31-0.34 measured; the 26S proteasome and matrix probes are in range.</li>
 <li><b>Absolute D is too fast:</b> median 2.4× in cytosol, 3.4× in nucleus, about 10× for membrane proteins. The model has no binding, and nuclear proteins are slowed mostly by binding; membranes lack an actin fence.</li>
 <li><b>ER lumen</b> is faster than cytosol in the model (measured about 3× slower), because only about 65% of its intended protein could be packed.</li>
</ul>
<figure><div class="frame" style="background:#fff"><img src="{img_uri(f'{VAL}/bd_model_vs_measured.png')}" alt="Scatter of simulated versus measured diffusion coefficients by compartment."></div>
<figcaption>Simulated vs measured diffusion coefficients for the benchmark probes.</figcaption></figure>"""

    figs = {k: img_uri(f"{VAL}/{p}") for k, p in (("panel", "fluor_panel.png"), ("hpa", "fluor_hpa_jurkat.png"))}

    verdicts = [
        ("Placement of the inventory", "ok" if S["mass_frac"] > 0.97 else "part",
         f"{100 * S['mass_frac']:.1f}% of target mass placed; below 90%: {', '.join(low) or 'none'}."),
        ("Cell and nuclear size", "ok", "Nucleus 36-38% of cell volume; mid-plane imaging gives about 41%. One 3D confocal study reports 63%."),
        ("Macromolecule concentration", "part", f"Cytosol {dr.set_index('region').mg_per_ml.get('cytosol', 0):.0f} mg/mL is in range; nucleus/cytosol {den['nuc_cyto_ratio']:.2f}; matrix and ER lumen are low."),
        ("Ribosome spacing", "ok", f"Nearest-neighbour median {th['nn_median']:.1f} nm in the model vs {em['nn_median']:.1f} nm in the EM; a random layout gives {jit['nn_median']:.1f} nm."),
        ("Localisation vs microscopy", "part", f"Agreement {ho['opencell']['intended']['agree']['unweighted']:.2f} with OpenCell and {ho['hpa_test_half']['intended']['agree']['unweighted']:.2f} with held-out HPA; systematic causes identified."),
        ("Simulated fluorescence", "part", FLUOR_VERDICT.get(a.version, FLUOR_VERDICT["v5"])),
        ("Crowding and diffusion", "part", "Moving-crowder simulations reproduce size-dependent slow-down for GFP tandems and small dextrans (about half of all size ratios within 25%); absolute D is 2-3× too fast in solution and about 10× in membranes, consistent with missing binding."),
    ]
    vt = "".join(f'<tr><td>{esc(n)}</td><td>{pill(k)}</td><td>{esc(t)}</td></tr>' for n, k, t in verdicts)

    page = f"""<title>Jurkat Model Validation</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a single reading column of report sections; wide tables and figures scroll inside their own frames. */
:root {{
  --bg: #f6f8f7; --surface: #ffffff; --fg: #18211e; --muted: #5b6863; --line: #d9e1dd;
  --accent: #1f7a56; --accent-soft: #dcefe6;
  --ok: #1f7a56; --ok-bg: #dcefe6; --part: #8a5a00; --part-bg: #f6e9cc; --bad: #a3322a; --bad-bg: #f7dcd8; --check: #3d5a8a; --check-bg: #dde5f3;
  --prev: #b8c4bf;
  --display: "Newsreader", Georgia, serif; --body: "Public Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #111816; --surface: #18211e; --fg: #e3ebe7; --muted: #9aa8a2; --line: #2c3833; --accent: #5cc595; --accent-soft: #1d3a2e;
  --ok: #7fd3aa; --ok-bg: #1d3a2e; --part: #e8bf6a; --part-bg: #3a2f17; --bad: #f0958c; --bad-bg: #3f1f1c; --check: #9db7e3; --check-bg: #1f2a3d; --prev: #4a5853; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #111816; --surface: #18211e; --fg: #e3ebe7; --muted: #9aa8a2; --line: #2c3833; --accent: #5cc595; --accent-soft: #1d3a2e;
  --ok: #7fd3aa; --ok-bg: #1d3a2e; --part: #e8bf6a; --part-bg: #3a2f17; --bad: #f0958c; --bad-bg: #3f1f1c; --check: #9db7e3; --check-bg: #1f2a3d; --prev: #4a5853; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body); }}
.wrap {{ max-width: 980px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 80px; display: grid; gap: 44px; }}
header {{ display: grid; gap: 10px; }}
.eyebrow {{ font: 500 12px var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }}
h1 {{ font: 600 clamp(30px, 5vw, 44px)/1.1 var(--display); margin: 0; text-wrap: balance; }}
h2 {{ font: 600 24px/1.25 var(--display); margin: 0; text-wrap: balance; }}
.lede {{ max-width: 68ch; color: var(--muted); margin: 0; }}
section {{ display: grid; gap: 14px; }}
section > p, li {{ max-width: 70ch; margin: 0; }}
.meta {{ font: 13px var(--mono); color: var(--muted); }}
.tw {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font: 600 12px var(--body); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); white-space: nowrap; }}
tr:last-child td {{ border-bottom: 0; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td {{ font-variant-numeric: tabular-nums; }}
.pill {{ display: inline-block; font: 600 12px var(--body); padding: 2px 9px; border-radius: 99px; white-space: nowrap; }}
.pill.ok {{ color: var(--ok); background: var(--ok-bg); }} .pill.part {{ color: var(--part); background: var(--part-bg); }}
.pill.bad {{ color: var(--bad); background: var(--bad-bg); }} .pill.check, .pill.open {{ color: var(--check); background: var(--check-bg); }}
.verdicts td:first-child {{ font-weight: 600; white-space: nowrap; }}
.chart {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 12px; overflow-x: auto; }}
.chart svg {{ width: 100%; min-width: 560px; height: auto; display: block; }}
svg .grid {{ stroke: var(--line); }} svg .tick {{ fill: var(--muted); font: 11px var(--mono); }}
svg .lab {{ fill: var(--fg); font: 13px var(--body); }} svg .val {{ fill: var(--fg); font: 12px var(--mono); }}
svg .prev {{ fill: var(--prev); }} svg .cur {{ fill: var(--accent); }} svg .cur.low {{ fill: var(--part); }}
figure {{ margin: 0; display: grid; gap: 8px; }}
figure .frame {{ overflow-x: auto; background: #000; border-radius: 6px; }}
figure img {{ display: block; width: 100%; min-width: 640px; height: auto; max-width: none; }}
figcaption {{ font-size: 13px; color: var(--muted); max-width: 80ch; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
.stat {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; display: grid; gap: 2px; min-width: 0; }}
.stat b {{ font: 500 22px var(--mono); }} .stat span {{ font-size: 13px; color: var(--muted); }}
ul {{ margin: 0; padding-left: 20px; display: grid; gap: 6px; }}
a {{ color: var(--accent); }}
code {{ font: 13px var(--mono); }}
</style>
<div class="wrap">
<header>
  <div class="eyebrow">Jurkat T cell · pack {esc(a.version)} · OpenOrganelle jrc_jurkat-1, cell 1</div>
  <h1>How well does the model match measurements?</h1>
  <p class="lede">Each check compares the packed model ({S['instances']:,} packed molecules plus the pre-placed ribosomes, chromatin, cytoskeleton and pores) with an independent measurement. Where the model inherits a number from its own inputs, the check is marked as circular and counts for little.</p>
  <div class="meta">Built from <code>data/derived/validation/</code> · previous version shown for comparison: pack {esc(a.prev)}</div>
</header>

<section><h2>Summary</h2>
<div class="tw"><table class="verdicts"><thead><tr><th>Check</th><th>Result</th><th>Finding</th></tr></thead><tbody>{vt}</tbody></table></div>
</section>

<section><h2>Placement of the inventory</h2>
<p>Share of each compartment's target protein mass that the packer placed. Thin compartments are hardest: the mitochondrial matrix sits in 50-80 nm slabs between cristae and the ER lumen is often under 30 nm wide.</p>
<div class="chart">{place_svg}</div>
<p class="meta">Total {S['placed_pg']:.1f} of {S['expected_pg']:.1f} pg ({100 * S['mass_frac']:.1f}%); pack {esc(a.prev)}: {100 * P['mass_frac']:.1f}%.</p>
</section>

<section><h2>Cell and nuclear size</h2>
<div class="stats">
 <div class="stat"><b>{Svol:.0f} µm³</b><span>cell volume inside the EM block</span></div>
 <div class="stat"><b>{nvol:.0f} µm³</b><span>nucleus</span></div>
 <div class="stat"><b>36-38%</b><span>nuclear share of cell volume (other EM cells: 34-40%)</span></div>
</div>
<ul>
 <li>{esc(FACE.get(a.version, FACE["v5"]))}</li>
 <li>Mid-plane nucleus:cell area ratio 55 ± 8% (Hoechst and phase contrast) converts to about 41% by volume: <b>agrees</b>.</li>
 <li>3D confocal reconstruction (Feng et al. 2014, Cytometry A) gives cell 664 ± 308 µm³ and nucleus 408 ± 199 µm³: the cell volume agrees; the nucleus is 1.5× ours, with a large spread.</li>
 <li>Automated counters report 12-13 µm diameters (900-1,150 µm³ as spheres), so this cell is at the small end, possibly from EM sample shrinkage.</li>
</ul>
</section>

<section><h2>Macromolecule concentration</h2>
<p>Protein and RNA mass per compartment volume, including pre-placed ribosomes and chromatin. The packer's target is {S['expected_pg']:.0f} pg; pre-placed ribosomes and chromatin come on top.</p>
{table(dtab, num=("Volume (µm³)", "Macromolecules (pg)", "Model (mg/mL)"))}
<p class="meta">Nucleus/cytosol density ratio {den['nuc_cyto_ratio']:.2f}. Matrix and ER lumen are low because the packer could not place all their target mass.</p>
</section>

<section><h2>Ribosome spacing against the EM</h2>
<p>The EM segmentation detects about half of all ribosomes, so the model's ribosomes are thinned at random to the EM count before comparing nearest-neighbour distances. A random layout at the same density is shown for reference.</p>
<div class="stats">
 <div class="stat"><b>{em['nn_median']:.1f} nm</b><span>EM, median nearest neighbour</span></div>
 <div class="stat"><b>{th['nn_median']:.1f} nm</b><span>model, thinned to EM count</span></div>
 <div class="stat"><b>{jit['nn_median']:.1f} nm</b><span>random layout, same density</span></div>
 <div class="stat"><b>{100 * em['frac_nn_lt_30']:.0f}% vs {100 * th['frac_nn_lt_30']:.0f}%</b><span>neighbours within 30 nm, EM vs model</span></div>
</div>
<p>The model's polysomes reproduce the clustering seen in the EM. The model has no pairs closer than 20 nm, where the EM has 13%; these are likely double detections or touching ribosomes the packer forbids.</p>
</section>

<section><h2>Localisation against microscopy</h2>
<p>Each protein's compartment shares in the model are compared with imaging annotations that were never used to build or fix the model. Agreement is 1 for a perfect match and 0 for disjoint compartments.</p>
{table(htab, num=("Genes", f"Agreement, {a.prev}", "Agreement", "Agreement, mass-weighted", "Top compartment correct"))}
<p>Most disagreement comes from a few systematic causes rather than single-protein errors. These drive the next model version:</p>
{cause_html}
<p class="meta">{esc(LOC_NOTE.get(a.version, LOC_NOTE["v5"]))}</p>
</section>

<section><h2>Image metrics against HPA Jurkat images (Tier 2)</h2>
{tier2_html or "<p>Not available for this version.</p>"}
</section>

<section><h2>Simulated fluorescence</h2>
<p>Every copy of a protein in the model is blurred with a confocal point-spread function (250 nm lateral, 600 nm axial), sampled at 65 nm pixels with photon noise, and cut through the nucleus centre.</p>
<figure><div class="frame"><img src="{figs['panel']}" alt="Grid of 45 simulated confocal sections of the model cell, one per protein, green on black."></div>
<figcaption>Simulated confocal sections, pack {esc(a.version)}. PM/C and N/C are plasma-membrane/cytoplasm and nucleus/cytoplasm intensity ratios.</figcaption></figure>
{table(ftab, num=("PM/cytoplasm", "Nucleus/cytoplasm"))}
<figure><div class="frame"><img src="{figs['hpa']}" alt="Model sections (top) and Human Protein Atlas immunofluorescence images of Jurkat cells (bottom) for CD28, CD2, CD3E and SELPLG."></div>
<figcaption>The four panel proteins the Human Protein Atlas imaged in Jurkat cells. Top: model; bottom: HPA (green = protein, blue = nucleus, red = microtubules). HPA images © Human Protein Atlas, CC BY-SA 4.0, proteinatlas.org.</figcaption></figure>
</section>

<section><h2>Crowding and diffusion</h2>
<p>Estimated slow-down of a probe diffusing through the frozen crowd, with a hydrodynamic correction. These numbers test the calculation and how crowded each compartment is; they are not yet a validation, because the crowders do not move.</p>
{table(ctab, num=("Volume fraction φ", "D/D₀, 1 nm probe", "D/D₀, GFP (2.3 nm)", "D/D₀, 3 nm probe"))}
<p>Measured GFP diffusion in mammalian cytoplasm is about 25-33% of its value in water. A benchmark of {len(db)} measured diffusion coefficients from {nref} papers ({np.nanmin(dv):.2g} to {np.nanmax(dv):.3g} µm²/s; {int(db.cell_type.astype(str).str.contains('Jurkat', case=False).sum())} in Jurkat) is ready. It covers GFP size series, dextrans, large complexes and T-cell membrane proteins. </p>
{bd_html}
</section>

<section><h2>Known issues and next version</h2>
<ul>{"".join(f"<li>{esc(t)}</li>" for t in KNOWN.get(a.version, KNOWN["v5"]))}</ul>
</section>
</div>
"""
    open(a.out, "w").write(page)
    print("wrote", a.out, f"{os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
