"""SpatialClaw mechanism probe — a humble reconstruction of the action interface.

This script demonstrates the core design of SpatialClaw without requiring:
  - a real VLM backbone,
  - GPU perception servers (SAM3 / Depth-Anything-3),
  - a SLURM cluster,
  - a full Jupyter kernel.

Instead, it uses:
  - a persistent Python workspace (mock_kernel.PersistentKernel),
  - lightweight mock tools (mock_tools),
  - an AST safety sandbox (safety.check_code_safety),
  - a scripted agent that plays the role of the VLM.

The scenario: a single image with two objects (red car, blue bicycle). The
question asks which object is closer to the camera. The agent writes one code
cell per step, inspects intermediate evidence, revises if needed, and calls
ReturnAnswer() to finish.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

from mock_kernel import ExecutionResult, PersistentKernel
from mock_tools import InputImages, MockFrameImage, ToolsModule
from safety import check_code_safety


@dataclass
class AgentStep:
    purpose: str
    reasoning: str
    next_goal: str
    code: str


_return_answer_value: Optional[Any] = None


def ReturnAnswer(value: Any) -> Any:
    """The sentinel used by the agent to submit a final answer.

    When called, it stores the value in a module-level sentinel so the loop
    can detect submission, matching SpatialClaw's builtin-sentinel design.
    """
    global _return_answer_value
    _return_answer_value = value
    print(f"[ReturnAnswer] submitted: {value}")
    return value


def build_test_scene() -> InputImages:
    """Create a synthetic image with a red car and blue bicycle."""
    W, H = 320, 240
    img = Image.new("RGB", (W, H), (200, 210, 220))
    draw = ImageDraw.Draw(img)
    # Ground plane
    draw.rectangle([0, H * 0.6, W, H], fill=(120, 130, 120))
    # Red car on the left (closer/larger)
    car_box = [W * 0.15, H * 0.45, W * 0.40, H * 0.65]
    draw.rectangle(car_box, fill=(200, 60, 60), outline=(50, 20, 20), width=2)
    draw.text((car_box[0], car_box[1] - 12), "car", fill=(0, 0, 0))
    # Blue bicycle on the right (smaller/farther)
    bike_box = [W * 0.62, H * 0.50, W * 0.78, H * 0.68]
    draw.ellipse([bike_box[0], bike_box[1] + 10, bike_box[0] + 20, bike_box[3]],
                 outline=(30, 60, 180), width=3)
    draw.ellipse([bike_box[2] - 20, bike_box[1] + 10, bike_box[2], bike_box[3]],
                 outline=(30, 60, 180), width=3)
    draw.line([bike_box[0] + 10, bike_box[1] + 20,
               bike_box[2] - 10, bike_box[3] - 10],
              fill=(30, 60, 180), width=3)
    draw.text((bike_box[0], bike_box[1] - 12), "bicycle", fill=(0, 0, 0))
    return InputImages([MockFrameImage(img, frame_index=0)])


def inject_workspace(kernel: PersistentKernel, input_images: InputImages) -> None:
    """Pre-load the kernel with inputs, tools, show(), and ReturnAnswer."""
    tools = ToolsModule(input_images)
    kernel.inject("InputImages", input_images)
    kernel.inject("tools", tools)
    kernel.inject("np", np)
    kernel.inject("ReturnAnswer", ReturnAnswer)

    # show() simply prints a marker that the feedback loop can surface.
    def show(visual_input, label: str = ""):
        if isinstance(visual_input, list):
            print(f"[show() called with {len(visual_input)} image(s), label={label!r}]")
        else:
            print(f"[show() called with 1 image, label={label!r}]")
    kernel.inject("show", show)

    # Also patch matplotlib.pyplot.show to capture figures like SpatialClaw does.
    capture_code = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _plt
import io as _io
from PIL import Image as _PILImage
_original_show = _plt.show
def _patched_show(*args, **kwargs):
    figs = [_plt.figure(n) for n in _plt.get_fignums()]
    for fig in figs:
        buf = _io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        print(f"[matplotlib figure captured: {fig.get_axes()[0].get_title() if fig.get_axes() else 'untitled'}]")
        _plt.close(fig)
    return None
_plt.show = _patched_show
"""
    kernel.execute(capture_code)


def format_feedback(step_idx: int, result: ExecutionResult, show_images: List[str]) -> str:
    """Build the feedback text the agent receives after a step."""
    lines = [
        f"--- Step {step_idx + 1} feedback ---",
        f"Status: {'ERROR' if result.error else 'SUCCESS'}",
    ]
    if result.error:
        lines.append(f"Error: {result.error}")
    if result.stdout.strip():
        lines.append(f"stdout:\n{result.stdout.strip()}")
    if result.new_variables:
        lines.append("New variables:")
        for name, meta in result.new_variables.items():
            meta_str = ", ".join(f"{k}={v}" for k, v in meta.items())
            lines.append(f"  {name}: {meta_str}")
    if show_images:
        lines.append(f"show() images: {', '.join(show_images)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scripted agent cells for the demo scenario.
# In the real system these are generated by the VLM; here we author them to
# illustrate the kind of multi-step reasoning SpatialClaw enables.
# ---------------------------------------------------------------------------

AGENT_CELLS: List[AgentStep] = [
    AgentStep(
        purpose="Segment the two objects of interest.",
        reasoning="To answer which object is closer, I first need masks for the car and the bicycle.",
        next_goal="Verify the masks visually, then reconstruct depth.",
        code="""
seg_car = tools.SAM3.segment_image_by_text(InputImages[0], "red car", label="car")
seg_bike = tools.SAM3.segment_image_by_text(InputImages[0], "blue bicycle", label="bicycle")
print("car mask area:", tools.Mask.area(seg_car.get_mask(0, "car")))
print("bike mask area:", tools.Mask.area(seg_bike.get_mask(0, "bicycle")))
show([seg_car.visualize(0), seg_bike.visualize(0)])
""",
    ),
    AgentStep(
        purpose="Reconstruct the scene in 3D.",
        reasoning="Depth from pixels alone is ambiguous; reconstruct metric 3D geometry so I can measure distances in world space.",
        next_goal="Compute 3D centroids and compare camera distances.",
        code="""
recon = tools.Reconstruct.Reconstruct(InputImages)
print("reconstructed frame indices:", recon.frame_indices)
print("camera pose:")
print(recon.extrinsics[0])
""",
    ),
    AgentStep(
        purpose="Compute metric distances from camera to each object.",
        reasoning="Use the reconstructed point clouds and masks to get 3D centroids, then compare Euclidean distances from the camera origin.",
        next_goal="Cross-check the result with a visualization, then answer.",
        code="""
cam_pos = recon.extrinsics[0][:3, 3]
car_3d = seg_car.get_centroid_3d(recon, frame=0, object="car")
bike_3d = seg_bike.get_centroid_3d(recon, frame=0, object="bicycle")
print("car 3d centroid:", car_3d)
print("bike 3d centroid:", bike_3d)
d_car = tools.Geometry.euclidean_distance(cam_pos, car_3d)
d_bike = tools.Geometry.euclidean_distance(cam_pos, bike_3d)
print(f"distance camera->car: {d_car:.2f} m")
print(f"distance camera->bike: {d_bike:.2f} m")
bev = recon.render_bev(masks=seg_car, ego_trajectory=False)
show(bev)
""",
    ),
    AgentStep(
        purpose="Submit the final answer.",
        reasoning="The car centroid is closer to the camera than the bicycle centroid (lower depth value).",
        next_goal="Terminate the session.",
        code="""
answer = "car" if d_car < d_bike else "bicycle"
ReturnAnswer(answer)
""",
    ),
]


def run_probe():
    print("=" * 70)
    print("SpatialClaw Action-Interface Probe")
    print("Scenario: which object is closer to the camera — red car or blue bicycle?")
    print("=" * 70)

    input_images = build_test_scene()
    kernel = PersistentKernel()
    inject_workspace(kernel, input_images)

    # Save the input image for reference.
    out_dir = os.path.dirname(os.path.abspath(__file__))
    input_images[0].image.save(os.path.join(out_dir, "probe_input.jpg"))
    print(f"\n[SAVED] input image -> {os.path.join(out_dir, 'probe_input.jpg')}")

    final_answer = None
    for step_idx, step in enumerate(AGENT_CELLS):
        print(f"\n{'='*70}")
        print(f"Step {step_idx + 1}: {step.purpose}")
        print(f"Reasoning: {step.reasoning}")
        print(f"Next goal: {step.next_goal}")
        print("-" * 70)

        # 1. Safety check (AST sandbox).
        safety_error = check_code_safety(step.code)
        if safety_error:
            print(f"[SECURITY BLOCK] {safety_error}")
            break

        # 2. Execute in persistent kernel.
        result = kernel.execute(step.code)

        # 3. Detect ReturnAnswer via the module-level sentinel.
        global _return_answer_value
        if _return_answer_value is not None:
            final_answer = _return_answer_value
            _return_answer_value = None

        # 4. Collect show() images from stdout markers (mocked as text here).
        show_images = [line for line in result.stdout.splitlines() if "show()" in line or "matplotlib" in line]

        # 5. Feedback assembly.
        feedback = format_feedback(step_idx, result, show_images)
        print(feedback)

        if final_answer is not None:
            print(f"\n[FINAL ANSWER SUBMITTED] {final_answer}")
            break

    print("\n" + "=" * 70)
    if final_answer is None:
        print("Session ended without an answer.")
    else:
        print(f"Result: {final_answer} is closer to the camera.")
    print("=" * 70)


if __name__ == "__main__":
    run_probe()
