"""模型加载+单步推理冒烟(需要权重下载完成后)。"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
import sys
from types import SimpleNamespace

OPENVLA_ROOT = os.environ["OPENVLA_ROOT"]
sys.path.insert(0, OPENVLA_ROOT)
sys.path.insert(0, os.path.join(OPENVLA_ROOT, "experiments", "robot", "libero"))

from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_model, set_seed_everywhere

CKPT = os.environ["CKPT"]
set_seed_everywhere(0)
cfg = SimpleNamespace(
    model_family="openvla", pretrained_checkpoint=CKPT, load_in_8bit=False,
    load_in_4bit=False, center_crop=True,
)
t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
t0.record()
model = get_model(cfg)
t1.record(); torch.cuda.synchronize()
print("model load time:", t0.elapsed_time(t1) / 1000, "s", flush=True)
print("norm_stats keys:", list(model.norm_stats.keys())[:5], flush=True)
processor = get_processor(cfg)
print("processor OK", flush=True)

# 单步推理:随机图像 + 提示
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
from PIL import Image
image = Image.fromarray(img).convert("RGB")
prompt = "In: What action should the robot take to pick up the black bowl?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
t0.record()
action = model.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)
t1.record(); torch.cuda.synchronize()
print("inference time:", t0.elapsed_time(t1) / 1000, "s", flush=True)
print("action:", action, "shape:", action.shape, flush=True)
print("GPU mem:", torch.cuda.memory_allocated() / 1e9, "GB", flush=True)
print("SMOKE_MODEL_DONE", flush=True)
