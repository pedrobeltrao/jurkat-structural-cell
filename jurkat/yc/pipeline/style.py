"""Goodsell-inspired palette and per-bead colour/ID assignment for the column model."""
import numpy as np
from render import hex2rgb
from column import LAY, L, NPC_HOLE

PAL = {
    "exterior_bg": "#F6F1E7", "wall_bg": "#EEE3CC", "mannan_bg": "#F1E9D3", "cyto_bg": "#E4ECF5",
    "lumen_bg": "#E8F0DF", "nuc_bg": "#F1E7F1",
    "glucan": "#CDA866", "mannan": "#C3C27A", "wall_protein": "#9A8352",
    "lipid_head": "#F3C643", "lipid_tail": "#F8E3A0", "ergosterol": "#E7A93C",
    "PM_protein": "#5FAE75", "ER_protein": "#7DBE86", "NE_protein": "#6BB596", "NPC": "#3E9C9A",
    "cyto_protein": "#7EA7D3", "cyto_big": "#8B88C9", "ER_lumen": "#A3C48C",
    "rprot": "#EE9BC0", "rRNA": "#C9558F", "tRNA": "#E27FA8", "mRNA": "#B0437A",
    "nuc_protein": "#B993C9", "pol": "#9A72B8", "histone": "#F4B26A", "DNA": "#E07636",
}
BIG = {"proteasome_26S", "FAS", "CCT"}
POL = {"PolI", "PolII", "PolIII"}


def jitter(rgb, rng, amt=0.07):
    return np.clip(rgb * (1 + rng.uniform(-amt, amt)), 0, 1)


def instance_colors(pk, rng):
    """returns per-instance (primary, secondary) colour for kind 0 / kind 1 beads"""
    c0, c1 = [], []
    for (lid, _, _), tag in zip(pk.placed, pk.tags):
        sec = None
        if lid == "ribosome_80S":
            p, sec = PAL["rprot"], PAL["rRNA"]
        elif lid == "nucleosome":
            p, sec = PAL["histone"], PAL["DNA"]
        elif lid == "tRNA":
            p = PAL["tRNA"]
        elif lid == "NPC":
            p = PAL["NPC"]
        elif tag == "PM":
            p = PAL["PM_protein"]
        elif tag.startswith("ER_membrane"):
            p = PAL["ER_protein"]
        elif tag.startswith("NE_"):
            p = PAL["NE_protein"]
        elif tag == "cell_wall":
            p = PAL["wall_protein"]
        elif tag == "ER_lumen":
            p = PAL["ER_lumen"]
        elif tag == "nucleoplasm":
            p = PAL["pol"] if lid in POL else PAL["nuc_protein"]
        else:
            p = PAL["cyto_big"] if lid in BIG else PAL["cyto_protein"]
        p = jitter(hex2rgb(p), rng)
        c0.append(p); c1.append(jitter(hex2rgb(sec), rng, 0.03) if sec else p)
    return np.array(c0), np.array(c1)


def assemble_scene(pk, ppos, pinst, pkind, lip, glc, glc_fid, man, man_owner, linker, rng, r_prot=0.6, r_na=0.8, mrna=()):
    c0, c1 = instance_colors(pk, rng)
    pos = [ppos]; rad = [np.where(pkind == 1, r_na, r_prot).astype(np.float32)]
    col = [np.where((pkind == 1)[:, None], c1[pinst], c0[pinst])]; mol = [pinst.astype(np.int64)]
    off = len(pk.placed)
    # lipids
    head = lip["rad"] > 0.5
    lc = np.where(head[:, None], hex2rgb(PAL["lipid_head"]), hex2rgb(PAL["lipid_tail"]))
    erg = np.array(lip["classes"])[lip["cls"]] == "ergosterol"
    lc[erg] = hex2rgb(PAL["ergosterol"])
    lc = lc * (1 + rng.uniform(-0.04, 0.04, (len(lc), 1)))
    pos.append(lip["pos"]); rad.append(lip["rad"]); col.append(lc); mol.append(lip["leaf"].astype(np.int64) + off)  # one outline per leaflet (Goodsell-style bilayer)
    lip_range = (off, off + int(lip["leaf"].max()) + 1)
    off += int(lip["leaf"].max()) + 1
    # glucan
    gcol = np.array([jitter(hex2rgb(PAL["glucan"]), rng, 0.06) for _ in range(glc_fid.max() + 1)])
    pos.append(glc); rad.append(np.full(len(glc), 0.75, np.float32)); col.append(gcol[glc_fid]); mol.append(glc_fid.astype(np.int64) + off)
    off += glc_fid.max() + 1
    # mannan (one molecule id per owning protein -> outlined as a glycan cloud distinct from the protein)
    mc = hex2rgb(PAL["mannan"]) * (1 + rng.uniform(-0.05, 0.05, (len(man), 1)))
    pos.append(man); rad.append(np.full(len(man), 0.6, np.float32)); col.append(mc); mol.append(man_owner.astype(np.int64) + off)
    off += man_owner.max() + 1
    # linker DNA
    if len(linker):
        pos.append(linker.astype(np.float32)); rad.append(np.full(len(linker), 1.0, np.float32))
        col.append(np.tile(hex2rgb(PAL["DNA"]), (len(linker), 1))); mol.append(np.full(len(linker), off, np.int64))
    off += 1
    for i, ch in enumerate(mrna):
        pos.append(ch.astype(np.float32)); rad.append(np.full(len(ch), 0.65, np.float32))
        col.append(np.tile(jitter(hex2rgb(PAL["mRNA"]), rng, 0.05), (len(ch), 1))); mol.append(np.full(len(ch), off + i, np.int64))
    return (np.vstack(pos).astype(np.float64), np.concatenate(rad).astype(np.float64),
            np.vstack(col).astype(np.float64), np.concatenate(mol), lip_range)


def make_bg(ymid):
    a = (LAY["inm"] - LAY["onm"]) / 2
    def bg(zz, xx):
        zz = np.broadcast_to(zz, np.broadcast(zz, xx).shape); xx = np.broadcast_to(xx, zz.shape)
        out = np.empty(zz.shape + (3,))
        out[:] = hex2rgb(PAL["cyto_bg"])
        out[zz < LAY["wall_in"]] = hex2rgb(PAL["wall_bg"])
        out[zz < LAY["mannan_end"]] = hex2rgb(PAL["mannan_bg"])
        out[zz < 0] = hex2rgb(PAL["exterior_bg"])
        lum = (xx >= LAY["er_x"][0]) & (xx < LAY["er_x"][1]) & (zz > LAY["erA"]) & (zz < LAY["erB"])
        out[lum] = hex2rgb(PAL["lumen_bg"])
        ar = (LAY["erB"] - LAY["erA"]) / 2; zc = (LAY["erA"] + LAY["erB"]) / 2
        for xe in (LAY["er_x"][1], LAY["er_x"][0], LAY["er_x"][0] + L):
            out[(xx - xe) ** 2 + (zz - zc) ** 2 < ar ** 2] = hex2rgb(PAL["lumen_bg"])
        r = np.hypot(xx - L / 2, ymid - L / 2)
        peri = (zz > LAY["onm"]) & (zz < LAY["inm"]) & (r > NPC_HOLE + a)
        out[peri] = hex2rgb(PAL["lumen_bg"])
        out[zz >= LAY["inm"]] = hex2rgb(PAL["nuc_bg"])
        return out
    return bg
