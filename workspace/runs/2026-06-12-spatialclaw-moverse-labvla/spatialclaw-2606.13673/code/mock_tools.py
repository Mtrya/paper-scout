"""Mock spatial tools for the SpatialClaw probe.

These are intentionally lightweight stand-ins for the real perception stack
(SAM3, Depth-Anything-3) so the probe can demonstrate the *action interface*
without needing GPU models or HTTP servers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class MockFrameImage:
    """A single frame with an absolute frame index, like SpatialClaw's FrameImage."""
    image: Image.Image
    frame_index: int

    def __getitem__(self, idx):
        # Allow FrameImage to behave like a list of one image.
        return self


class InputImages:
    """Minimal InputImages container exposing list access and frame_indices."""

    def __init__(self, frames: List[MockFrameImage]):
        self._frames = frames
        self.frame_indices = [f.frame_index for f in frames]

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return InputImages(self._frames[idx])
        return self._frames[idx]


@dataclass
class PerFrameMask:
    """Mock segmentation result with per-frame masks and object labels."""
    masks: np.ndarray  # (num_frames, num_objects, H, W) bool
    labels: List[str]
    frame_indices: List[int]
    frames: List[MockFrameImage] = field(default_factory=list)

    def get_mask(self, frame: int, object):
        fi = self.frame_indices.index(frame)
        if isinstance(object, int):
            return self.masks[fi, object]
        return self.masks[fi, self.labels.index(object)]

    def get_centroid_3d(self, recon, frame: int, object):
        """Return the 3D centroid of the masked points at a given frame."""
        mask = self.get_mask(frame, object)
        points = recon.points[frame]  # (H, W, 3)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        pts = points[ys, xs]
        return np.median(pts, axis=0)

    def visualize(self, frame: int):
        """Render a mask overlay for show()."""
        fi = self.frame_indices.index(frame)
        img = np.array(self.frames[fi].image).copy()
        mask = self.masks[fi]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for obj_idx in range(mask.shape[0]):
            ys, xs = np.where(mask[obj_idx])
            img[ys, xs] = (img[ys, xs] * 0.5 + 0.5 * np.array(colors[obj_idx % 3])).astype(np.uint8)
        return Image.fromarray(img)


@dataclass
class Reconstruction:
    """Mock 3D reconstruction result."""
    points: Dict[int, np.ndarray]  # frame_idx -> (H, W, 3)
    depth: Dict[int, np.ndarray]   # frame_idx -> (H, W)
    extrinsics: Dict[int, np.ndarray]  # frame_idx -> (4, 4)
    intrinsics: Dict[int, Dict[str, float]]
    frame_indices: List[int]

    def render_bev(self, masks=None, labels=None, ref_frame=None, ego_trajectory=True):
        """Draw a toy top-down view (X vs Z)."""
        size = 256
        img = Image.new("RGB", (size, size), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        # World bounds
        pts = np.concatenate([p.reshape(-1, 3) for p in self.points.values()])
        xs, zs = pts[:, 0], pts[:, 2]
        margin = 0.2
        x_min, x_max = xs.min() - margin, xs.max() + margin
        z_min, z_max = zs.min() - margin, zs.max() + margin

        def to_px(x, z):
            px = (x - x_min) / (x_max - x_min) * (size - 20) + 10
            pz = (z - z_min) / (z_max - z_min) * (size - 20) + 10
            return px, pz

        # Camera trajectory
        if ego_trajectory:
            cam_pts = [to_px(self.extrinsics[f][:3, 3][0], self.extrinsics[f][:3, 3][2])
                       for f in self.frame_indices]
            for i in range(len(cam_pts) - 1):
                draw.line([cam_pts[i], cam_pts[i + 1]], fill=(100, 100, 255), width=2)

        # Object masks
        if masks is not None:
            if isinstance(masks, PerFrameMask):
                labels = masks.labels
                for obj_idx, label in enumerate(labels):
                    for fi in masks.frame_indices:
                        c = masks.get_centroid_3d(self, fi, obj_idx)
                        if c is not None:
                            draw.ellipse([to_px(c[0], c[2])[0] - 4,
                                          to_px(c[0], c[2])[1] - 4,
                                          to_px(c[0], c[2])[0] + 4,
                                          to_px(c[0], c[2])[1] + 4],
                                         fill=(255, 0, 0))
                            draw.text(to_px(c[0], c[2]), label, fill=(0, 0, 0))
        return img


class SAM3Tool:
    """Mock SAM3: produces circular masks around fake object centers."""

    def __init__(self, input_images: InputImages):
        self.input_images = input_images

    def segment_image_by_text(self, image, prompt: str, label: str = None):
        # Extract a single frame index from the image argument.
        if isinstance(image, MockFrameImage):
            frame_image = image
            frame = image.frame_index
            img = image.image
        elif isinstance(image, InputImages):
            frame_image = image[0]
            frame = image.frame_indices[0]
            img = frame_image.image
        else:
            frame_image = MockFrameImage(image, 0)
            frame = 0
            img = image

        W, H = img.size
        # Fake object centers: prompt determines horizontal position.
        prompts = prompt.lower()
        if "car" in prompts or "red" in prompts:
            cx, cy = int(W * 0.35), int(H * 0.45)
        elif "bike" in prompts or "bicycle" in prompts or "blue" in prompts:
            cx, cy = int(W * 0.65), int(H * 0.55)
        elif "person" in prompts:
            cx, cy = int(W * 0.5), int(H * 0.5)
        else:
            cx, cy = int(W * 0.5), int(H * 0.5)

        Y, X = np.ogrid[:H, :W]
        mask = ((X - cx) ** 2 + (Y - cy) ** 2) < (min(H, W) * 0.18) ** 2
        masks = np.array([[mask]])  # (1 frame, 1 object, H, W)
        lbl = label or prompt
        return PerFrameMask(masks=masks, labels=[lbl], frame_indices=[frame], frames=[frame_image])


class ReconstructTool:
    """Mock Depth-Anything-3 reconstruction."""

    def __init__(self, input_images: InputImages):
        self.input_images = input_images

    def Reconstruct(self, frames):
        if isinstance(frames, InputImages):
            frame_list = [frames[i] for i in range(len(frames))]
        else:
            frame_list = frames if isinstance(frames, list) else [frames]

        frame_indices = [f.frame_index for f in frame_list]
        points: Dict[int, np.ndarray] = {}
        depth: Dict[int, np.ndarray] = {}
        extrinsics: Dict[int, np.ndarray] = {}
        intrinsics: Dict[int, Dict[str, float]] = {}

        H, W = frame_list[0].image.size[1], frame_list[0].image.size[0]
        fx, fy, cx, cy = W * 0.8, H * 0.8, W / 2.0, H / 2.0

        # Create a slanted ground plane so depth varies naturally.
        y_grid, x_grid = np.mgrid[0:H, 0:W]
        # Z increases toward bottom of image (deeper into scene).
        for i, f in enumerate(frame_list):
            z = 2.0 + 3.0 * (y_grid / H) + 0.5 * np.sin(x_grid / W * np.pi)
            # Back-project using pinhole model.
            X = (x_grid - cx) * z / fx
            Y = -(y_grid - cy) * z / fy  # OpenCV: Y points down
            pts = np.stack([X, Y, z], axis=-1).astype(np.float32)
            points[f.frame_index] = pts
            depth[f.frame_index] = z.astype(np.float32)

            # Camera moves forward 0.3m per frame along -Z.
            pose = np.eye(4, dtype=np.float64)
            pose[2, 3] = -0.3 * i  # world position along Z
            extrinsics[f.frame_index] = pose
            intrinsics[f.frame_index] = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

        return Reconstruction(
            points=points,
            depth=depth,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            frame_indices=frame_indices,
        )


class GeometryTool:
    """Static geometric helpers, mirroring tools.Geometry."""

    @staticmethod
    def euclidean_distance(p1, p2):
        return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))

    @staticmethod
    def normalized_to_pixel(coords, width, height):
        return [v / 1000.0 * d for v, d in zip(coords, [width, height] * (len(coords) // 2 + 1))][:len(coords)]


class MaskTool:
    """Static mask helpers, mirroring tools.Mask."""

    @staticmethod
    def centroid(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return (float("nan"), float("nan"))
        return (float(np.median(xs)), float(np.median(ys)))

    @staticmethod
    def area(mask):
        return int(mask.sum())


class ToolsModule:
    """Assembled namespace injected into the kernel as `tools`."""

    def __init__(self, input_images: InputImages):
        self.SAM3 = SAM3Tool(input_images)
        self.Reconstruct = ReconstructTool(input_images)
        self.Geometry = GeometryTool()
        self.Mask = MaskTool()
