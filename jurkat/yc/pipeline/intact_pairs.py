"""Compact physical-interaction evidence from an IntAct MITAB export (S. cerevisiae) -> data/intact_pairs.tsv.

Kept: protein-protein records of type association, physical association or direct interaction.
Per unordered UniProt pair: number of records, distinct publications, and whether any record is a direct interaction.
Usage: python intact_pairs.py /path/to/IntActYeast.txt
"""
import re, sys, collections
import pandas as pd
from config import DATA

KEEP = ("MI:0914", "MI:0915", "MI:0407")


def build(path, out=f"{DATA}/intact_pairs.tsv"):
    rec = collections.Counter(); pubs = collections.defaultdict(set); direct = set()
    acc = re.compile(r"uniprotkb:([A-Z0-9]{6,10})")
    with open(path) as f:
        next(f)
        for line in f:
            c = line.split("\t", 15)
            a, b = acc.match(c[0]), acc.match(c[1])
            if not (a and b) or not any(k in c[11] for k in KEEP):
                continue
            key = tuple(sorted((a.group(1), b.group(1))))
            rec[key] += 1
            pubs[key].update(re.findall(r"pubmed:(\d+)", c[8]))
            if "MI:0407" in c[11]:
                direct.add(key)
    df = pd.DataFrame([(a, b, n, len(pubs[(a, b)]), (a, b) in direct) for (a, b), n in rec.items()],
                      columns=["acc_a", "acc_b", "n_records", "n_publications", "direct"])
    df.to_csv(out, sep="\t", index=False)
    return df


if __name__ == "__main__":
    d = build(sys.argv[1])
    print(len(d), "pairs;", int(d.direct.sum()), "with direct-interaction records;", int((d.n_publications >= 2).sum()), "with >= 2 publications")
