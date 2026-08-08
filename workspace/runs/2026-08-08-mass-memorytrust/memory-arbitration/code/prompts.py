"""Prompt construction for the staleness-detection audit (OMCD-style, B=10)
and for the probe-3 arbitration experiments.

The paper does not release its verbatim prompt; this is a faithful
reconstruction from Sections 3.5, 4.2 and Appendix B.
"""
from __future__ import annotations

SYSTEM_DETECTION = (
    "You are auditing an embodied agent's persistent spatial memory against the "
    "current state of a FrozenLake grid world. The world is an 8x8 grid of cells "
    "of types S (start), F (frozen ice, safe to walk on), H (hole, lethal to step on), "
    "G (goal). The memory entries below were written from an EARLIER observation of "
    "the world. Since then, some frozen cells may have thawed into holes (F -> H) and "
    "some holes may have frozen over (H -> F). Start and goal cells never change.\n"
    "For EACH memory entry, compare its claim against the CURRENT observation at the "
    "SAME coordinate and judge whether the entry is still valid:\n"
    "- A SAFE claim is stale iff the current cell at that coordinate is a hole (H).\n"
    "- A DANGER claim is stale iff the current cell at that coordinate is frozen (F).\n"
    "Respond with a JSON array, one object per memory entry, in the same order, each "
    'of the form {"memory_id": "mem_XXX", "reasoning": "<one or two sentences: what the '
    'current observation shows at this coordinate, and whether it matches the claim>", '
    '"is_stale": true/false, "confidence": <0.0-1.0>}. Think through the comparison in '
    "the reasoning FIRST, then set is_stale accordingly. Output ONLY the JSON array."
)

VISION_LEGEND = (
    "The current observation is the attached 384x384 image rendering of the grid: "
    "each 48x48 cell is color-coded light blue = frozen (F, safe), near-black = hole "
    "(H, lethal), green = start (S), gold = goal (G), with a red triangle marking the "
    "agent. Every cell carries its coordinate label (row,col) in its top-left corner."
)

TEXT_OBS_HEADER = "Current observation (text grid):"
TEXT_OBS_HEADER_ACCURATE = "Current observation (real-time sensor feed, always accurate):"


def detection_user_prompt(entries: list[dict], mode: str, obs_text: str | None = None,
                          obs_header: str = TEXT_OBS_HEADER) -> str:
    """One OMCD audit batch. For mode='vision', the image is attached separately."""
    mem_block = "\n".join(e["text"] for e in entries)
    if mode == "vision":
        obs_block = VISION_LEGEND
    else:
        obs_block = f"{obs_header}\n{obs_text}"
    return (
        f"{obs_block}\n\n"
        f"Memory entries to audit:\n{mem_block}\n\n"
        f"Judge each of the {len(entries)} entries above as valid or stale. "
        "Remember: compare the memory claim with the CURRENT observation at the same "
        "coordinate. Output ONLY the JSON array."
    )


# ---------------- probe 3: arbitration factors ----------------
# Single-entry audit in text mode. Memory says SAFE, observation shows H (conflict,
# thaw direction = safety-critical) or F (control, memory valid).

F1_WORDING = {
    "definitely": "Frozen ice - definitely safe to walk",
    "probably": "Frozen ice - probably safe to walk",
    "neutral": "Frozen ice - safe to walk",
}
F2_AGE = {
    "now": " (observed just now)",
    "ago": " (observed long ago)",
    "none": "",
}
F3_PRIOR = {
    "frequent": ("Note about this world: the environment changes FREQUENTLY; "
                 "previously recorded information is often outdated."),
    "rare": ("Note about this world: the environment RARELY changes; "
             "previously recorded information is almost always still accurate."),
    "none": "",
}
F4_OBS = {
    "accurate": TEXT_OBS_HEADER_ACCURATE,
    "plain": TEXT_OBS_HEADER,
}


def probe3_prompt(entry_text: str, obs_text: str, obs_header: str = TEXT_OBS_HEADER,
                  prior_note: str = "") -> str:
    parts = []
    if prior_note:
        parts.append(prior_note + "\n")
    parts.append(f"{obs_header}\n{obs_text}\n")
    parts.append(f"Memory entry to audit:\n{entry_text}\n")
    parts.append(
        "Judge whether this memory entry is still valid or has become stale, given the "
        "current observation. Respond with a single JSON object of the form "
        '{"memory_id": "mem_XXX", "reasoning": "<one or two sentences: what the current '
        'observation shows at this coordinate, and whether it matches the claim>", '
        '"is_stale": true/false, "confidence": <0.0-1.0>}. Think through the comparison '
        "in the reasoning FIRST, then set is_stale accordingly. Output ONLY the JSON object."
    )
    return "\n".join(parts)
