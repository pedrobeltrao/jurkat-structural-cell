# Uploading the viewer data to Hugging Face

The viewer data (`data/derived/web_full_v3/` in the workspace, about 16 GB in about 2,000 files) go to the dataset
repository **`pedrobeltrao/jurkat-structural-cell-data`**, in the folder `web_full_v3/`. The site then reads
`https://huggingface.co/datasets/pedrobeltrao/jurkat-structural-cell-data/resolve/main/web_full_v3/`.

These are instructions only; nothing has been uploaded.

## 1. Check the folder

```sh
cd <workspace>                     # VCELL_ROOT
conda_envs/vcell/bin/python jurkat/viewer/check_data.py data/derived/web_full_v3
cat data/derived/web_full_v3/encode_meta.json       # file count, total size, largest file
```

Every file should be under 50 MB. The encoder splits larger files into `.000`, `.001`, … parts, listed in
`split.json`. The folder `data/derived/web_full_v3_work/` holds the encoder's resume state. Do **not** upload it.

## 2. Log in and create the dataset

```sh
pip install -U "huggingface_hub[cli]"      # provides the `hf` command (older versions: huggingface-cli)
hf auth login                              # token with write access for pedrobeltrao (do not commit it)
hf repo create pedrobeltrao/jurkat-structural-cell-data --repo-type dataset          # add --private to start private
```

A **private** dataset cannot be read by the public site: the browser has no token. Keep it private while you review,
for example with a local viewer and a token-authenticated download. Make it public before you switch the site to it.

## 3. Upload

Use the resumable large-folder uploader. It hashes, chunks and retries, so you can run it again after an
interruption.

```sh
hf upload-large-folder pedrobeltrao/jurkat-structural-cell-data --repo-type dataset \
    data/derived/web_full_v3 --num-workers 8
```

`upload-large-folder` mirrors the local folder at the repository root. To put the files under `web_full_v3/`
instead, upload from a parent folder that contains only that subfolder:

```sh
mkdir -p /tmp/hfup && ln -s "$PWD/data/derived/web_full_v3" /tmp/hfup/web_full_v3
hf upload-large-folder pedrobeltrao/jurkat-structural-cell-data --repo-type dataset /tmp/hfup --num-workers 8
```

If the symlink is not followed, copy the folder or use `rsync -a` into `/tmp/hfup/web_full_v3`. For a single
commit of a smaller folder (for example the 1.6 µm cube), `hf upload <repo> <local> <path-in-repo> --repo-type
dataset` is enough.

Add a dataset card (`README.md` at the dataset root). It should give:

- what the files are, with a link to the GitHub repository;
- the licence (to be chosen; CC BY 4.0 keeps the attribution chain of the sources);
- the source attributions from the main README.

## 4. Check hosting from a browser context

```sh
B=https://huggingface.co/datasets/pedrobeltrao/jurkat-structural-cell-data/resolve/main/web_full_v3
curl -sIL "$B/version.json" | grep -i "access-control-allow-origin\|HTTP/"
# one tile = a byte range of its block shard (columns 'off' and 'gz_bytes' of tiles/index.json); expect 206
read SHARD RANGE <<< "$(python -c "
import json; j = json.load(open('data/derived/web_full_v3/tiles/index.json')); c = j['tile_columns']; t = j['tiles'][len(j['tiles']) // 2]
b = j['shard']['block_tiles']; ix, iy, iz = (t[c.index(k)] // b for k in ('ix', 'iy', 'iz')); o = t[c.index('off')]
print(f's_{ix}_{iy}_{iz}.bin', f'{o}-{o + t[c.index(\"gz_bytes\")] - 1}')")"
curl -sL -r "$RANGE" -H "Origin: https://pedrobeltrao.github.io" -o /tmp/tile.gz -w "%{http_code} %{size_download}\n" "$B/tiles/$SHARD"
gzip -t /tmp/tile.gz && echo "tile decompresses"
```

Then open `https://pedrobeltrao.github.io/jurkat-structural-cell/` (or a local build with
`?data=$B/`). In the HUD, check that tiles stream. The browser console should show no CORS errors.

## 5. Point the site at the data

The Pages workflow writes `config.json` from the repository variable `DATA_BASE`. It defaults to the URL above. To
use another dataset or version, set `DATA_BASE` under Settings, then Secrets and variables, then Actions, then
Variables, and re-run the workflow. No code change is needed.

## Updating to pack v4 later

Encode v4 into a new folder, for example `data/derived/web_full_v4`. Upload it as `web_full_v4/` next to v3, then
change `DATA_BASE`. Old links keep working with `?data=…/web_full_v3/`. Each build has its own `version.json`, so
browsers never mix cached files from two builds.

## Updating to pack v5

Encode v5 into `data/derived/web_full_v5` (done on the cluster, see PLANNING_NOTES §7g "v5"). Upload it as
`web_full_v5/` next to v4, then set `DATA_BASE` to `…/resolve/main/web_full_v5/` and re-run the Pages workflow. The
v5 app also reads v4 data, so old links keep working with `?data=…/web_full_v4/`. The v5 build fixes a field-offset
bug that mislabelled part of the Golgi in the v4 cut faces; re-encoding only the fields of v4 (`jenc_full.py
--fields-only`) would fix the old data too.

## Pack v7 (published 2026-11-11)

`data/derived/web_full_v7` (2,367 files, 16.7 GB) was uploaded with
`hf upload pedrobeltrao/jurkat-structural-cell-data data/derived/web_full_v7 web_full_v7 --repo-type dataset` (resumable:
run it again after an interruption). The default `DATA_BASE` of the Pages workflow is now `…/resolve/main/web_full_v7/`.
The v5 and v6 encodes were not uploaded. Old links keep working with `?data=…/web_full_v4/` or `?data=…/web_full_v3/`.
