"""SpatialSTALE testbed reconstruction (When Memory Lies, 2608.04574).

Pure-Python generator: 8x8 FrozenLake grids, natural-language memory snapshots,
controlled L1/L2 change regimes, text observations, 384x384 renderings.

Deterministic: everything derives from an explicit integer seed.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

H, W = 8, 8
START = (0, 0)
GOAL = (7, 7)
CELLS = ("S", "F", "H", "G")

# Rendering constants (Appendix B)
IMG_SIZE = 384
CELL_PX = 48
COLORS = {
    "F": (0xA8, 0xC8, 0xE8),  # light blue
    "H": (0x20, 0x20, 0x20),  # near-black
    "S": (0x2E, 0x8B, 0x57),  # green
    "G": (0xDA, 0xA5, 0x20),  # gold
}
AGENT_COLOR = (0xD0, 0x20, 0x20)  # red triangle

HAZARD = {"H": 1, "F": 0, "S": 0, "G": 0}


@dataclass
class Instance:
    seed: int
    regime: str                    # "L1" | "L2"
    grid0: list[list[str]]         # original grid
    grid: list[list[str]]          # changed grid
    flipped: list[tuple[int, int]] # cells whose type changed


def _solvable(grid: list[list[str]]) -> bool:
    from collections import deque
    seen = {START}
    q = deque([START])
    while q:
        r, c = q.popleft()
        if (r, c) == GOAL:
            return True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen and grid[nr][nc] != "H":
                seen.add((nr, nc))
                q.append((nr, nc))
    return False


def generate_grid(seed: int) -> list[list[str]]:
    """8x8 grid, ~25% holes among non-terminal cells, solvable (rejection sampling)."""
    rng = np.random.default_rng(seed)
    while True:
        cells = np.array(["F"] * (H * W - 2))
        n_holes = int(round(0.25 * len(cells)))  # 16 of 62
        cells[rng.choice(len(cells), size=n_holes, replace=False)] = "H"
        rng.shuffle(cells)
        grid = [["F"] * W for _ in range(H)]
        it = iter(cells.tolist())
        for r in range(H):
            for c in range(W):
                if (r, c) == START:
                    grid[r][c] = "S"
                elif (r, c) == GOAL:
                    grid[r][c] = "G"
                else:
                    grid[r][c] = next(it)
        if _solvable(grid):
            return grid


def _flip(cell: str) -> str:
    return "H" if cell == "F" else "F"


def apply_changes(grid0: list[list[str]], regime: str, seed: int) -> tuple[list[list[str]], list[tuple[int, int]]]:
    """L1: k~U{5,6,7} uniform random cells flipped. L2: k~U{12..16} drawn with
    replacement from union of 2-3 Manhattan radius-2 cluster neighborhoods
    (duplicates collapse, matching the paper's ~14% effective stale ratio).
    Solvability enforced by rejection resampling."""
    rng = np.random.default_rng(seed)
    nonterminal = [(r, c) for r in range(H) for c in range(W) if (r, c) not in (START, GOAL)]

    for _attempt in range(1000):
        grid = [row[:] for row in grid0]
        if regime == "L1":
            k = int(rng.integers(5, 8))
            idx = rng.choice(len(nonterminal), size=k, replace=False)
            targets = [nonterminal[i] for i in idx]
        elif regime == "L2":
            k = int(rng.integers(12, 17))
            n_centers = int(rng.integers(1, 3))
            centers = [nonterminal[i] for i in rng.choice(len(nonterminal), size=n_centers, replace=False)]
            pool: set[tuple[int, int]] = set()
            for cr, cc in centers:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if abs(dr) + abs(dc) <= 2:
                            p = (cr + dr, cc + dc)
                            if 0 <= p[0] < H and 0 <= p[1] < W and p not in (START, GOAL):
                                pool.add(p)
            pool = sorted(pool)
            draws = rng.choice(len(pool), size=k, replace=True)  # with replacement
            targets = sorted({pool[i] for i in draws})
        else:
            raise ValueError(regime)
        for r, c in targets:
            grid[r][c] = _flip(grid[r][c])
        if _solvable(grid):
            flipped = [(r, c) for r, c in targets if grid[r][c] != grid0[r][c]]
            return grid, flipped
    raise RuntimeError(f"no solvable change for seed={seed} regime={regime}")


def make_instance(seed: int, regime: str) -> Instance:
    grid0 = generate_grid(seed)
    # change stream derives from a separate sub-seed so grid0 is regime-independent
    grid, flipped = apply_changes(grid0, regime, seed * 1_000_003 + (11 if regime == "L1" else 22))
    return Instance(seed=seed, regime=regime, grid0=grid0, grid=grid, flipped=flipped)


# ---------------- memory ----------------

def mem_id(r: int, c: int) -> str:
    return f"mem_{r * W + c:03d}"


_CLAIM = {
    "F": ("SAFE", "Frozen ice - safe to walk"),
    "H": ("DANGER", "Hole - dangerous, do not step"),
    "S": ("SAFE", "Start position - safe"),
    "G": ("SAFE", "Goal position - safe"),
}


def memory_entries(grid0: list[list[str]]) -> list[dict]:
    """One entry per cell, row-major ids (mem_000..mem_063), as in the paper."""
    entries = []
    for r in range(H):
        for c in range(W):
            claim, text = _CLAIM[grid0[r][c]]
            entries.append({
                "memory_id": mem_id(r, c),
                "pos": (r, c),
                "claim": claim,
                "type0": grid0[r][c],
                "text": f"[{mem_id(r, c)}] {claim} at ({r},{c}): {text}",
            })
    return entries


def gold_labels(inst: Instance) -> dict[str, bool]:
    """memory_id -> is_stale (SAFE over current H, or DANGER over current F)."""
    labels = {}
    for e in memory_entries(inst.grid0):
        r, c = e["pos"]
        labels[e["memory_id"]] = HAZARD[e["type0"]] != HAZARD[inst.grid[r][c]]
    return labels


# ---------------- observations ----------------

def text_observation(grid: list[list[str]], agent: tuple[int, int] = START) -> str:
    lines = []
    for r in range(H):
        lines.append(f"Row {r}: " + " ".join(f"({r},{c})={grid[r][c]}" for c in range(W)))
    lines.append(f"Agent at ({agent[0]},{agent[1]}). Goal at ({GOAL[0]},{GOAL[1]}).")
    return "\n".join(lines)


def render_grid(grid: list[list[str]], agent: tuple[int, int] = START):
    """384x384 RGB rendering, Appendix B style."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=11)
    except TypeError:
        font = ImageFont.load_default()
    for r in range(H):
        for c in range(W):
            x0, y0 = c * CELL_PX, r * CELL_PX
            draw.rectangle([x0, y0, x0 + CELL_PX - 1, y0 + CELL_PX - 1], fill=COLORS[grid[r][c]])
            dark_cell = grid[r][c] in ("H", "S")
            draw.text((x0 + 3, y0 + 2), f"({r},{c})", fill=(230, 230, 230) if dark_cell else (40, 40, 40), font=font)
    # white 1px grid lines
    for i in range(1, W):
        draw.line([(i * CELL_PX, 0), (i * CELL_PX, IMG_SIZE)], fill=(255, 255, 255))
        draw.line([(0, i * CELL_PX), (IMG_SIZE, i * CELL_PX)], fill=(255, 255, 255))
    # agent: red triangle overlay
    ar, ac = agent
    cx, cy = ac * CELL_PX + CELL_PX // 2, ar * CELL_PX + CELL_PX // 2
    s = CELL_PX // 3
    draw.polygon([(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)], fill=AGENT_COLOR)
    return img


def blank_image():
    from PIL import Image
    return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))


# ---------------- instance seeds ----------------

def instance_seeds(n: int, master: int = 2024) -> list[int]:
    rng = np.random.default_rng(master)
    return sorted(rng.choice(range(1000, 10_000), size=n, replace=False).tolist())
