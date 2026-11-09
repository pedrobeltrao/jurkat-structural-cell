import requests, re, sys
url="https://rest.uniprot.org/uniprotkb/search?query=proteome:UP000002311&format=tsv&size=500&fields=accession,reviewed,gene_oln,gene_primary,length,mass,cc_subcellular_location,go_c,ft_transmem,cc_subunit,protein_name"
out=[];hdr=None
while url:
    r=requests.get(url,timeout=120); r.raise_for_status()
    lines=r.text.rstrip("\n").split("\n")
    if hdr is None: hdr=lines[0]; out.append(hdr)
    out+=lines[1:]
    m=re.search(r'<([^>]+)>; rel="next"', r.headers.get("Link",""))
    url=m.group(1) if m else None
open("uniprot_yeast.tsv","w").write("\n".join(out)+"\n"); print(len(out)-1)
