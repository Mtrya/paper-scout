"""Per-task instruction embedding with blank (gray) images: isolates the linguistic channel for the text ablation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dashdata as D  # noqa: E402
from extract_features import layer_indices, run_model  # noqa: E402


def main() -> int:
    model_id, out = sys.argv[1], Path(sys.argv[2])
    proc = AutoProcessor.from_pretrained(model_id)
    proc.image_processor.size = {"shortest_edge": 128 * 128, "longest_edge": 128 * 128}
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
    layers = layer_indices(model.config.text_config.num_hidden_layers)
    image_token_id = model.config.image_token_id or proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    gray = Image.fromarray(np.full((128, 128, 3), 127, np.uint8))
    res = {}
    for task, text in D.PROMPTS.items():
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        msgs_b = [msgs[0]] * 16          # same batch regime as the per-frame extraction
        prompt_b = proc.apply_chat_template(msgs_b, tokenize=False, add_generation_prompt=False)
        enc = proc(text=prompt_b, images=[[gray, gray]] * 16, return_tensors="pt", padding=True)
        b = dict(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                 mm_token_type_ids=enc["mm_token_type_ids"],
                 pixel_values=enc["pixel_values"], image_grid_thw=enc["image_grid_thw"])
        _, txt, _ = run_model(model, b, layers, image_token_id)
        res[task] = txt[0].astype(np.float32)
        print(task, res[task].shape, float(np.linalg.norm(res[task][-1])), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **res)
    (out.parent / "text_pure_layers.json").write_text(json.dumps({"model": model_id, "layers": layers}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
