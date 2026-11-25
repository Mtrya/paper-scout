"""PCM 读消融 harness:通过 MemoryController intervention 接口关闭 PCM 读路径。

教训:configs/model.yaml 的 content_memory_read_enabled 只进 mixer 的
self.config,而 dit.py 每次前向显式传 content_memory_read_enabled_runtime
(由 MemoryController 按窗口重设,默认恒 True),显式参数覆盖配置——直接翻
yaml 对推理零效果(首次尝试逐位复现了 PCM-on 的 rollout)。

正确入口是 MemoryController.configure_content_memory_intervention:
- 'readout_off' 只从第 4 窗起关读(作者自带工具;前 4 窗 = 训练视界内保持原状)
- full 模式在其上猴补丁 _set_content_memory_runtime,强制全部窗口读关

用法: python3 pcm_read_ablation.py full|w4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "code" / "motionmamba2"))
import motion_mamba2.model.checkpoint as ckpt

MODE = sys.argv[1] if len(sys.argv) > 1 else "full"
assert MODE in {"full", "w4"}, "mode must be full or w4"

_orig_load = ckpt.load_model


def patched_load(*args, **kwargs):
    model, client, _ = _orig_load(*args, **kwargs)
    controller = client.controller
    controller.configure_content_memory_intervention("readout_off")
    if MODE == "full":
        orig_runtime = controller._set_content_memory_runtime

        def forced_runtime():
            orig_runtime()
            for block in (controller.root_block, controller.body_block):
                block.content_memory_read_enabled_runtime = False

        controller._set_content_memory_runtime = forced_runtime
    blocks = [client.controller.root_block, client.controller.body_block]
    print(f"intervention installed: mode={MODE} "
          f"read_runtime={[getattr(b, 'content_memory_read_enabled_runtime', None) for b in blocks]} "
          f"strategy={controller.content_memory_intervention}", flush=True)
    return model, client, _


ckpt.load_model = patched_load

from benchmarks.evaluate import main

main(["--domain", "hiphi", "--task", "frames_988", "--limit", "48",
      "--paths", f"configs/paths-pcmoff-{MODE}.env"])
