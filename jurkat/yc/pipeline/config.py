"""Paths and machine settings for the pipeline. Every module imports its locations from here.

Override with environment variables:
  YC_ROOT       code + inputs (data/, lib/)            default: this directory
  YC_WORK       all generated output                    default: <bundle>/work
  YC_PDB        cache of downloaded PDB assemblies      default: $YC_WORK/pdb
  YC_AFDB_TAR   AlphaFold DB yeast proteome tar (v6)    default: $YC_ROOT/afdb/yeast_afdb.tar
  YC_DIMERS     AFDB predicted yeast dimers (homodimers/, heterodimers/)
  YC_NPROC      worker processes                        default: os.cpu_count()
"""
import os

ROOT = os.environ.get("YC_ROOT", os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("YC_WORK", os.path.join(os.path.dirname(ROOT), "work"))
DATA = f"{ROOT}/data"
LIB = f"{ROOT}/lib"
PDB = os.environ.get("YC_PDB", f"{WORK}/pdb")
def _first(*paths):
    """first existing path (the bundle default, then where this machine keeps the downloads)"""
    return next((p for p in paths if os.path.exists(p)), paths[0])


AFDB_TAR = os.environ.get("YC_AFDB_TAR") or _first(f"{ROOT}/afdb/yeast_afdb.tar",
                                                   os.path.expanduser("~/Downloads/UP000002311_559292_YEAST_v6.tar"))
DIMERS = os.environ.get("YC_DIMERS") or _first(f"{ROOT}/afdb_dimers",
                                               os.path.expanduser("~/Projects/bioemu_docker/afdb_yeast_dimers"))
NPROC = int(os.environ.get("YC_NPROC", os.cpu_count() or 8))

W3D = f"{WORK}/yc3d"                 # 3D intermediates (geometry, tile measures, prepass, libpack meta)
RAW = f"{W3D}/tiles_raw"             # packed instances per tile
GDIR = f"{W3D}/geom"                 # per-tile membrane meshes
WEB_TMP = f"{W3D}/web_tmp"           # encoder stage-1 intermediates
WEB_OUT = f"{W3D}/web/data"          # encoded viewer data (copy to the viewer's data/)
RESULTS = f"{WORK}/results"          # inventory tables, validation json

for _d in (WORK, PDB, W3D, RESULTS):
    os.makedirs(_d, exist_ok=True)
