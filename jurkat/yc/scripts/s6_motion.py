"""Stage 6 (exploration branch): pool-ensemble motion modes for the viewer. Run after s5_encode.py.

  1. kept residues of every AlphaFold monomer type (this env)           -> work/conformers/kept_residues.npz
  2. pipeline/pool_modes.py in an env with foldcomp (POOL_PY)           -> work/conformers/pool_modes.npz
  3. modes.bin / modes.json into the encoded library (this env)

Usage: s6_motion.py [--skip-fit]     POOL_DB: the pool foldcomp database; POOL_PY: a Python with foldcomp
                                     (default: the prot_dashboard conda env); MOTION_CAP: amplitude cap in nm (0 = none)
"""
from _common import *
import subprocess, json
import conformers as cf


def main():
    skip_fit = "--skip-fit" in sys.argv
    types = json.load(open(f"{C.WEB_OUT}/library/types.json"))["types"]
    if not skip_fit:
        cf.kept_residues(types, log=log)
        py = os.environ.get("POOL_PY", os.path.expanduser("~/miniconda3/envs/prot_dashboard/bin/python"))
        subprocess.run([py, f"{C.ROOT}/pool_modes.py", cf.RES_F, cf.MODES_F, "--procs", str(C.NPROC)], check=True)
    cf.encode_modes(cf.MODES_F, C.WEB_OUT, log=log)


if __name__ == "__main__":
    main()
