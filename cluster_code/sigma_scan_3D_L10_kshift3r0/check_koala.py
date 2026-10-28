# Which koala is installed? The package version is always the same, so this
# looks at the installed code and at a numerical fingerprint of the moved points.
#     python check_koala.py
# Compare the output with the local reference in README.md.
import inspect
import subprocess
from pathlib import Path
import numpy as np
from koala import pointsets
from tai_localiser.perulizer.lattices import iterative_move_point

print("koala file:", pointsets.__file__)

# git commit, if koala was installed from a clone (pip install -e)
repo = Path(pointsets.__file__).resolve().parents[2]
try:
    commit = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%h %ci %s"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
except (OSError, subprocess.CalledProcessError):
    commit = "not a git clone"
print("koala commit:", commit)

# features that changed the 3D lattices
print("grid in any dimension (8736e2f):",
      "n_vals" in str(inspect.signature(pointsets.grid)))
print("repulsion kappa/r^(d-1) (7e95ea3):",
      "n_sphere_sa" in inspect.getsource(pointsets.move_point))
print("original_points option (8a079f8):",
      "original_points" in str(inspect.signature(pointsets.move_all_points)))

# numerical fingerprint: same moves as the 3D scan (L=4, sigma=0.25 r0,
# kappa_shift=3 r0, resolution=10, seed 0)
L = 4
r0 = 1 / L
p0 = pointsets.grid(L, L, L)
p = iterative_move_point(p0, 0.25 * r0, 3 * r0, resolution=10,
                         rng=np.random.default_rng(0))
disp = (p - p0 + 0.5) % 1 - 0.5
print(f"fingerprint 3D: mean |disp| = {np.linalg.norm(disp, axis=1).mean() / r0:.8f} r0, "
      f"first point = {np.round(p[0], 8)}")
