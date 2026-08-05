#!/usr/bin/env python3
"""Inspect a PyTorch checkpoint's structure without allocating tensor storage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def find_state_dict(value: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(value, dict):
        if value and all(isinstance(key, str) for key in value):
            tensor_fraction = sum(isinstance(item, torch.Tensor) for item in value.values()) / len(value)
            if tensor_fraction > 0.8:
                return "<root>", value
        for candidate in ("state_dict", "model", "module", "model_state_dict"):
            nested = value.get(candidate)
            if isinstance(nested, dict):
                return candidate, nested
    raise ValueError("No tensor-like state dict found")


def flatten_tensors(value: Any, prefix: str, output: dict[str, Any]) -> None:
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "numel"):
        output[prefix or "<root>"] = value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flatten_tensors(item, child, output)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            flatten_tensors(item, child, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="meta",
        weights_only=True,
        mmap=True,
    )

    state_name, state = find_state_dict(checkpoint)
    tensors: dict[str, Any] = {}
    flatten_tensors(state, "", tensors)
    selected_terms = (
        "cog_mem",
        "per_mem",
        "retrieval",
        "gate_fusion",
        "timestep",
        "per_compr",
    )
    selected = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in tensors.items()
        if any(term in key.lower() for term in selected_terms)
    }
    prefixes = Counter(key.split(".", 1)[0] for key in tensors)
    payload = {
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "top_level_type": type(checkpoint).__name__,
        "top_level_keys": list(checkpoint) if isinstance(checkpoint, dict) else None,
        "state_dict_location": state_name,
        "state_top_entries": {
            key: {
                "type": type(value).__name__,
                "child_keys": list(value)[:20] if isinstance(value, dict) else None,
            }
            for key, value in list(state.items())[:30]
        },
        "tensor_count": len(tensors),
        "parameter_numel": sum(value.numel() for value in tensors.values()),
        "parameter_bytes": sum(value.numel() * value.element_size() for value in tensors.values()),
        "top_prefix_counts": prefixes.most_common(20),
        "memory_module_tensor_count": len(selected),
        "memory_module_tensors": selected,
        "serialized_episode_state_keys": [
            key
            for key in tensors
            if (key.startswith("cog_mem_bank.") or key.startswith("per_mem_bank."))
            and not (key.endswith(".weight") or key.endswith(".bias"))
        ],
        "non_tensor_top_level": {
            key: type(value).__name__
            for key, value in checkpoint.items()
            if not isinstance(value, torch.Tensor) and key != state_name
        }
        if isinstance(checkpoint, dict)
        else {},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
