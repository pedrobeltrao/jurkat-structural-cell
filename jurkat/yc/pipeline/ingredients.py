"""Assemble packing ingredients: curated macromolecular complexes (PDB assemblies) + residual
proteins (AlphaFold monomers, or PDB homo-oligomers for abundant proteins)."""
import json, re
import numpy as np
import pandas as pd

from config import DATA as D
GENOME_BP = 12.07e6      # S288C nuclear genome (chromosomal, excl. rDNA repeat expansion)
NRL_BP = 165             # assumed nucleosome repeat length

# curated complexes: (id, label, complex-portal ids or explicit gene->stoich, PDB assembly, compartment weights, membrane)
COMPLEXES = [
    ("ribosome_80S", "80S ribosome", "RIBO", "4V88", {"cytosol": 0.97, "ER": 0.03}, None),
    ("proteasome_26S", "26S proteasome", ["CPX-2262"], "4CR2", {"nucleoplasm": 0.5, "cytosol": 0.5}, None),
    ("FAS", "Fatty acid synthase (a6b6)", {"FAS1": 6, "FAS2": 6}, "2UV8", {"cytosol": 1.0}, None),
    ("CCT", "TRiC/CCT chaperonin", ["CPX-2156"], "4V94", {"cytosol": 1.0}, None),
    ("PolII", "RNA polymerase II", ["CPX-2662"], "1WCM", {"nucleoplasm": 1.0}, None),
    ("PolI", "RNA polymerase I", ["CPX-1664"], "4C2M", {"nucleolus": 1.0}, None),
    ("PolIII", "RNA polymerase III", ["CPX-2660"], "5FJ8", {"nucleoplasm": 1.0}, None),
    ("NPC", "Nuclear pore complex", ["CPX-824"], "7N9F", {"nuclear_envelope": 1.0}, "NE"),
    ("Pma1_hexamer", "Pma1 H+-ATPase hexamer", {"PMA": (6, ["PMA1", "PMA2"])}, "7VH5", {"plasma_membrane": 1.0}, "PM"),
    ("V-ATPase", "Vacuolar V-ATPase V1Vo", {"VMA1": 3, "VMA2": 3, "VMA4": 3, "VMA10": 3, "VMA8": 1, "VMA7": 1,
                                             "VMA5": 1, "VMA13": 1, "VPH1": 1, "VMA3": 8, "VMA11": 1, "VMA16": 1,
                                             "VMA6": 1, "VMA9": 1}, "6O7V", {"vacuole": 1.0}, "VAC"),
    ("nucleosome", "Nucleosome core particle", {"H3": (2, ["HHT1; HHT2"]), "H4": (2, ["HHF1; HHF2"]),
                                                "H2A": (2, ["HTA1", "HTA2"]), "H2B": (2, ["HTB1", "HTB2"])},
     "1ID3", {"nucleoplasm": 1.0}, None),
]
FIXED_COPIES = {"NPC": 117, "nucleosome": GENOME_BP / NRL_BP}


def complex_families(cp, spec, gene_of_acc, genes):
    """Return {family: (stoich, [paralog genes])}."""
    if spec == "RIBO":
        fam = {}
        for g in genes:
            if re.match(r"^(RP[LS]\d+|RPP[012])[AB]?$", g):
                fam.setdefault(re.sub(r"[AB]$", "", g), []).append(g)
        return {f: (1, gs) for f, gs in fam.items()}
    if isinstance(spec, dict):
        return {k: (v if isinstance(v, tuple) else (v, [k])) for k, v in spec.items()}
    out = {}
    for cid in spec:
        row = cp[cp["#Complex ac"] == cid].iloc[0]
        for tok in row["Identifiers (and stoichiometry) of molecules in complex"].split("|"):
            m = re.match(r"([A-Z0-9]+)(?:-\w+)?\((\d+)\)", tok)
            if m and gene_of_acc.get(m.group(1)):
                g = gene_of_acc[m.group(1)]
                out[g] = (max(1, int(m.group(2))), [g])
    return out


def build_ingredients(df):
    cp = pd.read_csv(f"{D}/complexportal_559292.tsv", sep="\t")
    gene_of_acc = dict(zip(df.uniprot, df.gene))
    pool = dict(zip(df.gene, df.copies_per_cell))
    mass = dict(zip(df.gene, df.mass_da))
    rows, consumed = [], {}
    for cid, label, spec, pdb, cw, memb in COMPLEXES:
        fams = complex_families(cp, spec, gene_of_acc, list(pool))
        fam_copies = {f: sum(pool.get(g, 0) for g in gs) / st for f, (st, gs) in fams.items()}
        if cid in FIXED_COPIES:
            copies = FIXED_COPIES[cid]
        elif cid == "Pma1_hexamer":
            copies = fam_copies["PMA"]
        else:
            vals = [v for v in fam_copies.values() if v > 0]
            copies = float(np.median(vals)) if vals else 0.0
        m_tot = 0.0
        for f, (st, gs) in fams.items():
            tot = sum(pool.get(g, 0) for g in gs)
            m_tot += st * np.mean([mass.get(g, 0) for g in gs])
            for g in gs:
                share = pool.get(g, 0) / tot if tot > 0 else 1 / len(gs)
                consumed[g] = consumed.get(g, 0) + copies * st * share
        genes_all = sorted(g for _, gs in fams.values() for g in gs)
        rows.append(dict(ing_id=cid, label=label, kind="complex", genes=";".join(genes_all), n_subunit_types=len(fams),
                         copies_per_cell=copies, comp_weights=json.dumps(cw), mass_da=m_tot, pdb=pdb,
                         membrane=memb or "", oligomer_n=1, uniprot=""))
    # residual monomer pool
    for r in df.itertuples():
        left = max(0.0, r.copies_per_cell - consumed.get(r.gene, 0.0))
        if left < 1:
            continue
        rows.append(dict(ing_id=r.gene, label=r.name, kind="protein", genes=r.gene, n_subunit_types=1,
                         copies_per_cell=left, comp_weights=r.comp_weights, mass_da=r.mass_da, pdb="",
                         membrane="", oligomer_n=r.oligomer_n, uniprot=r.uniprot, n_tm=r.n_tm))
    ing = pd.DataFrame(rows)
    ing["n_tm"] = ing.n_tm.fillna(0).astype(int)
    return ing, consumed
