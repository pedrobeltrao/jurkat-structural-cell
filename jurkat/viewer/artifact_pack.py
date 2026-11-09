"""Assemble the viewer as a claude.ai Artifact bundle: web/index.html + assets + an encoded data directory.

The Artifact host wraps the page in its own document skeleton, serves at most 15 MB per binary file and 511 files per
version, and blocks other hosts. So: the doctype/html/head/body wrappers are stripped from the built page, files over
SPLIT_MB are cut into path.000, path.001, ... listed in data/split.json (read by data.ts fetchBin), Mol* is left out (the
app falls back to the jsdelivr build; remote structure downloads are blocked there). Dynamics files and the "Diffusion"
bookmarks are copied from web/data when their boxes lie inside the bundle's cube.
usage: artifact_pack.py DATA_DIR OUT_DIR      (DATA_DIR from jenc.py --out; OUT_DIR is rebuilt)"""
import os, sys, re, json, shutil, base64

V = os.path.dirname(os.path.abspath(__file__))
SPLIT_MB = 11           # base64 adds a third: 11 MB -> 14.7 MB text, under the 16 MB text-file limit


def main(data, out):
    web = f"{V}/web"
    if os.path.exists(out): shutil.rmtree(out)
    os.makedirs(out)
    html = open(f"{web}/index.html").read()
    html = re.sub(r"<!doctype html>\s*", "", html, flags=re.I)
    html = re.sub(r"</?(html|head|body)[^>]*>\s*", "", html, flags=re.I)
    html = re.sub(r'<meta (charset|name="viewport")[^>]*>\s*', "", html)
    html = re.sub(r"<title>.*?</title>", "<title>Jurkat Cell Model</title>", html, count=1)
    open(f"{out}/index.html", "w").write(html)
    shutil.copytree(f"{web}/assets", f"{out}/assets")
    shutil.copytree(data, f"{out}/data")
    em = json.load(open(f"{out}/data/encode_meta.json")); half = em["cube_nm"] / 2
    bm = [b for b in json.loads(open(f"{out}/data/bookmarks.json").read().replace("NaN", "null"))
          if None not in b["target"] + b["normal"] + b["up"]]          # jenc writes NaN when a region is absent from the cube
    json.dump(bm, open(f"{out}/data/bookmarks.json", "w"), indent=1)
    if os.path.exists(f"{web}/data/dynamics.json"):
        dyn = json.load(open(f"{web}/data/dynamics.json")); wm = json.load(open(f"{web}/data/encode_meta.json"))
        off = [a - b for a, b in zip(wm["centre_nm"], em["centre_nm"])]       # web frame -> bundle frame
        keep = []
        for s in dyn.get("sets", []):
            lo = [a + o for a, o in zip(s["box_lo"], off)]
            if all(-half - 10 <= x and x + dyn["box_nm"] <= half + 10 for x in lo):        # boxes may graze the cube edge
                s["box_lo"] = lo; keep.append(s)
        if keep:
            dyn["sets"] = keep; os.makedirs(f"{out}/data/dynamics", exist_ok=True)
            for s in keep:
                for t in s["tracers"]: shutil.copy(f"{web}/data/{t['file']}", f"{out}/data/{t['file']}")
            json.dump(dyn, open(f"{out}/data/dynamics.json", "w"), indent=1)
            bm = json.load(open(f"{out}/data/bookmarks.json")); regs = {s["region"] for s in keep}
            for b in json.load(open(f"{web}/data/bookmarks.json")):
                if b.get("dynamics") in regs:
                    b["target"] = [round(a + o, 3) for a, o in zip(b["target"], off)]; bm.append(b)
            json.dump(bm, open(f"{out}/data/bookmarks.json", "w"), indent=1)
        print(f"dynamics sets kept: {[s['region'] for s in keep]}")
    split = {}
    for root, _, files in os.walk(f"{out}/data"):
        for f in files:
            p = os.path.join(root, f)
            if os.path.getsize(p) > SPLIT_MB * 2 ** 20:
                rel = os.path.relpath(p, f"{out}/data"); b = open(p, "rb").read(); n = SPLIT_MB * 2 ** 20
                k = (len(b) + n - 1) // n
                for i in range(k): open(f"{p}.{i:03d}", "wb").write(b[i * n:(i + 1) * n])
                os.remove(p); split[rel] = k
    for root, _, files in os.walk(f"{out}/data"):           # the host serves no binary types: base64 text at path.txt
        for f in files:
            p = os.path.join(root, f)
            if not f.endswith(".json"):
                open(p + ".txt", "w").write(base64.b64encode(open(p, "rb").read()).decode()); os.remove(p)
    split["_b64"] = 1
    json.dump(split, open(f"{out}/data/split.json", "w"))
    files = [os.path.relpath(os.path.join(r, f), out) for r, _, fs in os.walk(out) for f in fs]
    tot = sum(os.path.getsize(f"{out}/{f}") for f in files)
    print(f"{len(files)} files, {tot / 2 ** 20:.1f} MB, split: {split}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
