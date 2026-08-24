"""
Scene diagram: the 3-DOF arm, the eye-to-hand camera, and the six motion
shapes used for the observability analysis (EE-only trajectories).
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

import sim_intrinsic as sim
from analyze import savefig, set_style

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def main():
    set_style()
    R_true, t_true = sim.true_extrinsic()
    fig = plt.figure(figsize=(10.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # camera
    ax.scatter(*t_true, s=60, color="k", marker="^", label="camera")
    for j, axis_col in enumerate(R_true.T):
        ax.quiver(*t_true, *(0.25 * axis_col), color="tab:gray", lw=1)
    # arm at a representative pose
    q = sim.arm_ik(sim.C0)
    fk = sim.arm_fk(q)
    ax.scatter(*fk["p_elbow"], s=20, color="tab:brown")
    ax.plot([0, fk["p_elbow"][0]], [0, fk["p_elbow"][1]], [0, fk["p_elbow"][2]],
            color="tab:brown", lw=2, label="arm links")
    ax.plot([fk["p_elbow"][0], fk["p_ee"][0]],
            [fk["p_elbow"][1], fk["p_ee"][1]],
            [fk["p_elbow"][2], fk["p_ee"][2]], color="tab:brown", lw=2)

    # motion shapes (EE-only, seed 0)
    colors = {"line": "tab:red", "line_frontal": "tab:olive", "plane": "tab:orange",
              "volume": "tab:blue", "rot_arc": "tab:purple", "wave": "tab:green"}
    labels = {"line": "line", "line_frontal": "frontal line", "plane": "plane",
              "volume": "volume", "rot_arc": "rot_arc (yaw)", "wave": "wave (yaw+pitch)"}
    for kind in colors:
        feats, _ = sim.build_features(kind, 60.0, 0.05, 0, use_elbow=False)
        pts = np.stack([f[0] for f in feats[::5]])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=1.6, color=colors[kind],
                alpha=0.9, label=labels[kind])

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=20, azim=-60)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Scene: 3-DOF arm, eye-to-hand camera at (0,-2.2,0.5), "
                 "motion shapes (EE point, 60 s)")
    savefig(fig, os.path.join(FIGDIR, "fig0_setup.png"))
    print("figure ->", os.path.join(FIGDIR, "fig0_setup.png"))


if __name__ == "__main__":
    main()
