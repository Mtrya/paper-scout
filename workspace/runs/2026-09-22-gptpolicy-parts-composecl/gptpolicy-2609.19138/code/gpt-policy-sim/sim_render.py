"""Matplotlib (Agg) rendering of the sim world into protocol CapturedImages.

Two fixed views, both annotated so a reader who has never seen the world can
map pixels back to base-frame coordinates:

- ``top``: overhead view. Image up = world +x (forward), image left = world +y.
- ``front``: side view from -y looking toward +y. Image right = world +x,
  image up = world +z.
"""

from __future__ import annotations

import io
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow, Rectangle
from PIL import Image

from gpt_policy.hardware.camera import CapturedImage

from sim_world import WORKSPACE, World

COLOR = {
    "red": "#d62728", "green": "#2ca02c", "blue": "#1f77b4", "orange": "#ff7f0e",
    "gray": "#7f7f7f", "brown": "#8c564b", "magenta": "#d62796", "black": "#222222",
    "yellow": "#bcbd22",
}

# Fixed axes placement (figure fraction) so pixel<->world maps stay exact.
AXES_RECT = (0.07, 0.08, 0.86, 0.84)
TOP_YLIM = (0.05, 0.95)   # world x, screen up
TOP_XLIM = (0.95, -0.08)  # world y, reversed: +y on the left; y<0 keeps home visible
FRONT_XLIM = (0.05, 0.95)  # world x
FRONT_YLIM = (-0.03, 0.50)  # world z


class SimRenderer:
    def __init__(self, world: World, width: int = 640, height: int = 480, jpeg_quality: int = 80) -> None:
        self.world = world
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self._fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        self._ax = self._fig.add_axes(AXES_RECT)

    # ------------------------------------------------------------------ views
    def render(self, view: str) -> np.ndarray:
        ax = self._ax
        ax.clear()
        if view == "top":
            self._draw_top(ax)
        elif view == "front":
            self._draw_front(ax)
        else:
            raise ValueError(f"unknown view: {view}")
        self._fig.canvas.draw()
        rgba = np.asarray(self._fig.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    def capture(self, view: str) -> CapturedImage:
        rgb = self.render(view)
        image = Image.fromarray(rgb, "RGB")
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=self.jpeg_quality)
        return CapturedImage(
            name=view,
            data=stream.getvalue(),
            mime_type="image/jpeg",
            width=self.width,
            height=self.height,
            captured_at=time.time(),
            rgb_data=rgb.tobytes(),
            source_timestamp_s=float(self.world.time_s),
            source_clock="sim_clock",
        )

    def close(self) -> None:
        plt.close(self._fig)

    # -------------------------------------------------------------- top view
    def _draw_top(self, ax) -> None:
        ax.set_xlim(*TOP_XLIM)
        ax.set_ylim(*TOP_YLIM)
        ax.set_aspect("equal")
        ax.set_xlabel("world y (m)  <- left")
        ax.set_ylabel("world x (m)  ^ forward")
        ax.set_title(f"top view (overhead)   sim t={self.world.time_s:.1f}s", fontsize=9)
        self._draw_table(ax, top=True)
        for obj in self.world.objects.values():
            self._draw_object_top(ax, obj)
        self._draw_ee_top(ax)
        ax.grid(True, alpha=0.15, linewidth=0.3)

    def _draw_table(self, ax, top: bool) -> None:
        x0, x1 = WORKSPACE["x"]
        y0, y1 = WORKSPACE["y"]
        if top:
            ax.add_patch(Rectangle((0.10, 0.10), 0.80, 0.80, fill=True,
                                   facecolor="#f4ead5", edgecolor="#999999", linewidth=1.0, zorder=0))
            ax.add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False,
                                   edgecolor="#2ca02c", linestyle="--", linewidth=0.8, zorder=1))
            ax.text(0.5 * (y0 + y1), x0 - 0.015, "workspace", color="#2ca02c", fontsize=7,
                    ha="center")
        else:
            ax.axhline(0.0, color="#999999", linewidth=1.5)
            ax.add_patch(Rectangle((x0, -0.012), x1 - x0, 0.012, fill=False,
                                   edgecolor="#2ca02c", linestyle="--", linewidth=0.8, zorder=1))

    def _draw_object_top(self, ax, obj) -> None:
        x, y, z = obj.position
        color = COLOR.get(obj.color, obj.color)
        if obj.kind == "gate":
            self._draw_gate_top(ax, obj)
            return
        if obj.kind == "slot":
            self._draw_slot_top(ax, obj)
            return
        if obj.kind == "tube":
            self._draw_tube_top(ax, obj)
            return
        if obj.kind == "hook":
            self._draw_hook_top(ax, obj)
            return
        hx, hy = obj.half_extents[0], obj.half_extents[1]
        if obj.kind in ("bowl", "button", "plug"):
            ax.add_patch(Circle((y, x), max(hx, hy), fill=obj.kind != "bowl",
                                facecolor=color, edgecolor=color, alpha=0.9 if obj.kind != "bowl" else 0.35,
                                linewidth=1.5, zorder=3))
            if obj.kind == "bowl":
                ax.add_patch(Circle((y, x), max(hx, hy) * 0.7, fill=False,
                                    edgecolor=color, linewidth=1.0, zorder=3))
        elif obj.kind == "box":
            ax.add_patch(Rectangle((y - hy, x - hx), 2 * hy, 2 * hx, fill=True,
                                   facecolor=color, alpha=0.30, edgecolor=color, linewidth=1.5, zorder=2))
            socket = obj.meta.get("socket_xy")
            if socket is not None:
                ax.add_patch(Circle((socket[1], socket[0]), 0.015, fill=False,
                                    edgecolor="black", linewidth=1.5, linestyle=":", zorder=4))
                ax.text(socket[1], socket[0] + 0.05, "socket", fontsize=6, ha="center", color="black")
        else:  # cube and generic blocks
            ax.add_patch(Rectangle((y - hy, x - hx), 2 * hy, 2 * hx, fill=True,
                                   facecolor=color, edgecolor="black", linewidth=0.8, zorder=3))
        ax.text(y, x - hx - 0.022, obj.name, fontsize=6.5, ha="center", color="black", zorder=5)

    def _draw_gate_top(self, ax, obj) -> None:
        gate = obj.meta
        gx = gate["x"] + (gate["slide_offset"] if gate.get("open") else 0.0)
        y0, y1 = gate["y_span"]
        ax.add_patch(Rectangle((y0, gx - 0.006), y1 - y0, 0.012, fill=True,
                               facecolor="#9ecae1", alpha=0.55, edgecolor="#3182bd",
                               linewidth=1.2, zorder=2))
        state = "OPEN" if gate.get("open") else "CLOSED"
        ax.text(0.5 * (y0 + y1), gx + (0.02 if not gate.get("open") else 0.0),
                f"transparent gate ({state})", fontsize=7, ha="center", color="#08519c", zorder=5)

    def _draw_slot_top(self, ax, obj) -> None:
        slot = obj.meta
        y = slot["y"]
        x0, x1 = slot["x_span"]
        ax.add_patch(Rectangle((y - 0.010, x0), 0.020, x1 - x0, fill=True,
                               facecolor="#dddddd", edgecolor="#666666", linewidth=1.0, zorder=2))
        marker = slot["marker_x"]
        ax.plot([y - 0.02, y + 0.02], [marker, marker], color="red", linewidth=1.4, zorder=4)
        ax.text(y + 0.025, marker, "align mark", fontsize=6, color="red", ha="left", va="center")

    def _draw_tube_top(self, ax, obj) -> None:
        tube = obj.meta
        y, r = tube["y"], tube["r_outer"]
        x0, x1 = tube["x_span"]  # mouth .. closed end
        for sign in (-1, 1):
            ax.plot([y + sign * r, y + sign * r], [x0, x1], color="#7f7f7f", linewidth=2.0, zorder=2)
        ax.plot([y - r, y + r], [x1, x1], color="#7f7f7f", linewidth=2.0, zorder=2)
        ax.annotate("", xy=(y, x0 - 0.03), xytext=(y, x0 + 0.02),
                    arrowprops=dict(arrowstyle="->", color="#7f7f7f", lw=1.2), zorder=2)
        ax.text(y + r + 0.02, 0.5 * (x0 + x1), "tube (open end <-)", fontsize=6.5,
                color="#555555", va="center")

    def _draw_hook_top(self, ax, obj) -> None:
        x, y, _ = obj.position
        tip = obj.meta.get("tip_xy")
        ax.plot([y], [x], marker="s", color=COLOR["magenta"], markersize=5, zorder=4)
        if tip is not None:
            ax.plot([y, tip[1]], [x, tip[0]], color=COLOR["magenta"], linewidth=2.0, zorder=4)
            ax.plot([tip[1]], [tip[0]], marker="^", color=COLOR["magenta"], markersize=6, zorder=4)
        else:
            ax.plot([y, y + 0.10], [x, x], color=COLOR["magenta"], linewidth=2.0, zorder=4)
        ax.text(y + 0.02, x - 0.03, "hook", fontsize=6.5, color=COLOR["magenta"])

    def _draw_ee_top(self, ax) -> None:
        x, y, z = self.world.ee
        ax.plot([y], [x], marker="x", color="black", markersize=9, markeredgewidth=2.2, zorder=6)
        ax.add_patch(Circle((y, x), 0.012, fill=False, edgecolor="black", linewidth=1.2, zorder=6))
        if self.world.grasped:
            grip = f"holding {self.world.grasped}"
        elif self.world.gripper_measured < 0.2:
            grip = "gripper closed"
        elif self.world.gripper_measured > 0.9:
            grip = "gripper open"
        else:
            grip = f"gripper {self.world.gripper_measured:.2f}"
        label = f"EE z={z:.2f} {grip}"
        if y < 0.45:  # marker near the image right edge; keep text inside
            ax.text(y - 0.02, x + 0.012, label, fontsize=6.5, color="black", ha="right", zorder=6)
        else:
            ax.text(y + 0.02, x + 0.012, label, fontsize=6.5, color="black", zorder=6)

    # ------------------------------------------------------------ front view
    def _draw_front(self, ax) -> None:
        ax.set_xlim(*FRONT_XLIM)
        ax.set_ylim(*FRONT_YLIM)
        ax.set_xlabel("world x (m)  -> forward")
        ax.set_ylabel("world z (m)  ^ up")
        ax.set_title(f"front view (from -y)   sim t={self.world.time_s:.1f}s", fontsize=9)
        self._draw_table(ax, top=False)
        for index, obj in enumerate(self.world.objects.values()):
            self._draw_object_front(ax, obj, index)
        x, y, z = self.world.ee
        ax.plot([x], [z], marker="x", color="black", markersize=9, markeredgewidth=2.2, zorder=6)
        ax.plot([x, x], [z, z - 0.03], color="black", linewidth=1.5, zorder=6)
        ax.text(x + 0.015, z + 0.012, f"EE y={y:.2f}", fontsize=6.5, color="black", zorder=6)
        ax.grid(True, alpha=0.15, linewidth=0.3)

    def _draw_object_front(self, ax, obj, index: int = 0) -> None:
        x, y, z = obj.position
        color = COLOR.get(obj.color, obj.color)
        hx, hz = obj.half_extents[0], obj.half_extents[2]
        label_z = z + hz + 0.012 + (index % 3) * 0.024
        if obj.kind == "gate":
            gate = obj.meta
            gx = gate["x"] + (gate["slide_offset"] if gate.get("open") else 0.0)
            ax.add_patch(Rectangle((gx - 0.006, 0.0), 0.012, gate["height"], fill=True,
                                   facecolor="#9ecae1", alpha=0.55, edgecolor="#3182bd",
                                   linewidth=1.2, zorder=2))
            return
        if obj.kind == "slot":
            slot = obj.meta
            x0, x1 = slot["x_span"]
            ax.plot([x0, x1], [0.002, 0.002], color="#666666", linewidth=3.0, zorder=2)
            ax.plot([slot["marker_x"], slot["marker_x"]], [0.0, 0.02], color="red",
                    linewidth=1.4, zorder=4)
            return
        if obj.kind == "tube":
            tube = obj.meta
            zt, r = tube["z"], tube["r_outer"]
            x0, x1 = tube["x_span"]
            for sign in (-1, 1):
                ax.plot([x0, x1], [zt + sign * r, zt + sign * r], color="#7f7f7f", linewidth=2.0)
            ax.plot([x1, x1], [zt - r, zt + r], color="#7f7f7f", linewidth=2.0)
            return
        if obj.kind == "hook":
            tip = obj.meta.get("tip_xyz")
            if tip is not None:
                ax.plot([x, tip[0]], [z, tip[2]], color=COLOR["magenta"], linewidth=2.0, zorder=4)
                ax.plot([tip[0]], [tip[2]], marker="^", color=COLOR["magenta"], markersize=6, zorder=4)
            else:
                ax.plot([x, x + 0.10], [max(z, 0.008), max(z, 0.008)], color=COLOR["magenta"],
                        linewidth=2.0, zorder=4)
            return
        if obj.kind in ("bowl", "button", "plug"):
            width = 2 * max(hx, obj.half_extents[1])
            ax.add_patch(Rectangle((x - width / 2, z - hz), width, 2 * hz,
                                   fill=obj.kind != "bowl", facecolor=color, edgecolor=color,
                                   alpha=0.9 if obj.kind != "bowl" else 0.4, linewidth=1.5, zorder=3))
            ax.text(x, label_z, obj.name, fontsize=6.5, ha="center", color="black", zorder=5)
            return
        ax.add_patch(Rectangle((x - hx, z - hz), 2 * hx, 2 * hz, fill=True,
                               facecolor=color, edgecolor="black", linewidth=0.8,
                               alpha=0.9 if obj.kind != "box" else 0.35, zorder=3))
        ax.text(x, label_z, obj.name, fontsize=6.5, ha="center", color="black", zorder=5)

    # ------------------------------------------------------- pixel->world map
    def top_pixel_to_world(self, pixel_xy) -> tuple[float, float]:
        """Invert the top view's affine map (for the sim locate_point tool).

        Uses the last drawn axes window extent, so it stays exact under
        ``aspect="equal"`` letterboxing. Pixel origin is the image top-left.
        """
        px, py = float(pixel_xy[0]), float(pixel_xy[1])
        self._fig.canvas.draw()
        bbox = self._ax.get_window_extent()
        ax_x = (px - bbox.x0) / bbox.width
        ax_y = (py - bbox.y0) / bbox.height  # display y grows upward
        world_y = TOP_XLIM[0] + ax_x * (TOP_XLIM[1] - TOP_XLIM[0])
        world_x = TOP_YLIM[0] + ax_y * (TOP_YLIM[1] - TOP_YLIM[0])
        return world_x, world_y
