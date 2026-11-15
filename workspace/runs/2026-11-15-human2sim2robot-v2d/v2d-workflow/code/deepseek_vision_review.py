#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
import os
import tomllib
import urllib.request
from pathlib import Path

DEFAULT_PROMPT = "你是 V2D 工作流的视觉输入审核员。随后两图分别是 tissue（每2秒采样）和 airplane（每4秒采样）的接触图，均从左到右、从上到下排列，没有时间戳或真值标签。独立观察图像，给出目标物体的 GroundingDINO 文本 prompt、建议参考时刻、可见交互、风险、置信度与理由。重点判断 VLM 能否替代人工选择目标和参考帧，以及哪些三维接触、尺度和物理性质不能由这些图像确定。不要声称逐帧审核，不要把二维清晰等同于三维重建正确。只输出合法 JSON，顶层为 {model_role,inputs:[{clip,decision,object_prompt,reference_time_seconds,observations,risks,confidence,rationale}],overall_conclusion}。"


def credential():
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    config_path = Path.home() / ".kimi-code" / "config.toml"
    config = tomllib.loads(config_path.read_text())
    provider = config["providers"]["deepseek"]
    if provider.get("api_key"):
        return provider["api_key"]
    env_name = provider.get("api_key_env", "DEEPSEEK_API_KEY")
    if os.environ.get(env_name):
        return os.environ[env_name]
    raise RuntimeError("No DeepSeek credential available")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", action="append", nargs=2, required=True, metavar=("LABEL", "PATH"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--stem", default="deepseek_official")
    args = parser.parse_args()
    content = [{"type": "text", "text": args.prompt}]
    input_metadata = []
    for label, filename in args.image:
        path = Path(filename)
        payload = path.read_bytes()
        content.append({"type": "text", "text": f"下一张图是 {label}。"})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(payload).decode(),
                "detail": "original",
            },
        })
        input_metadata.append({
            "label": label,
            "filename": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    body = json.dumps({
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 5000,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }).encode()
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={
            "Authorization": "Bearer " + credential(),
            "Content-Type": "application/json",
        },
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=240) as response:
        result = json.load(response)
    text = result["choices"][0]["message"]["content"]
    parsed = json.loads(text)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / (args.stem + "_response.json")).write_text(json.dumps({
        "model": result.get("model"),
        "created": result.get("created"),
        "finish_reason": result["choices"][0].get("finish_reason"),
        "usage": result.get("usage"),
        "prompt": args.prompt,
        "inputs": input_metadata,
        "content": text,
    }, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / (args.stem + "_review.json")).write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
