"""Comparison table of synthetic robot data engines.

Feature coverage is taken from the LabVLA paper (Table 1) and verified against
public code releases where available. This script prints a Markdown table that
can be dropped directly into the report.

Run: python data_engine_comparison.py
"""
from __future__ import annotations


engines = [
    {
        "Engine": "RoboTwin 2.0",
        "Robots": 5,
        "Auto asset": False,
        "Auto scene": True,
        "Auto task": True,
        "Domain rand.": True,
        "Success QA": True,
        "Structured annotations": False,
        "Long-horizon composition": False,
        "Lab protocol": False,
        "Release": "github.com/OpenDriveLab/RoboTwin",
    },
    {
        "Engine": "RoboCasa 365",
        "Robots": 1,
        "Auto asset": False,
        "Auto scene": True,
        "Auto task": True,
        "Domain rand.": True,
        "Success QA": True,
        "Structured annotations": False,
        "Long-horizon composition": False,
        "Lab protocol": False,
        "Release": "robocasa.ai",
    },
    {
        "Engine": "ManiSkill 3",
        "Robots": 23,
        "Auto asset": False,
        "Auto scene": False,
        "Auto task": False,
        "Domain rand.": True,
        "Success QA": True,
        "Structured annotations": False,
        "Long-horizon composition": False,
        "Lab protocol": False,
        "Release": "github.com/haosulab/ManiSkill",
    },
    {
        "Engine": "RLBench",
        "Robots": 5,
        "Auto asset": False,
        "Auto scene": False,
        "Auto task": False,
        "Domain rand.": True,
        "Success QA": True,
        "Structured annotations": False,
        "Long-horizon composition": False,
        "Lab protocol": False,
        "Release": "github.com/stepjam/RLBench",
    },
    {
        "Engine": "RoboGen",
        "Robots": 6,
        "Auto asset": False,
        "Auto scene": True,
        "Auto task": True,
        "Domain rand.": False,
        "Success QA": True,
        "Structured annotations": False,
        "Long-horizon composition": False,
        "Lab protocol": False,
        "Release": "robogen-ai.github.io",
    },
    {
        "Engine": "RoboGenesis (LabVLA)",
        "Robots": 16,
        "Auto asset": True,
        "Auto scene": True,
        "Auto task": True,
        "Domain rand.": True,
        "Success QA": True,
        "Structured annotations": True,
        "Long-horizon composition": True,
        "Lab protocol": True,
        "Release": "Not public (training code only)",
    },
]


def fmt(v) -> str:
    if isinstance(v, bool):
        return "✓" if v else "—"
    return str(v)


def print_table() -> None:
    keys = [
        "Engine", "Robots", "Auto asset", "Auto scene", "Auto task",
        "Domain rand.", "Success QA", "Structured annotations",
        "Long-horizon composition", "Lab protocol", "Release",
    ]
    header = "| " + " | ".join(keys) + " |"
    sep = "|" + "|".join([" --- " for _ in keys]) + "|"
    print(header)
    print(sep)
    for row in engines:
        print("| " + " | ".join(fmt(row[k]) for k in keys) + " |")


if __name__ == "__main__":
    print_table()
