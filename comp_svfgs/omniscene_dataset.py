import copy
import json
import math
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


def _ensure_hwc3(img: np.ndarray) -> np.ndarray:
    """Ensure image has 3 channels."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        color = img[..., :3].astype(np.float32)
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        img = color * alpha + 255.0 * (1.0 - alpha)
    return img.astype(np.uint8)


def _load_info(sensor_info: Dict) -> Tuple[str, np.ndarray]:
    """Extract image path and c2w matrix (in lidar/world coordinates)."""
    img_path = sensor_info["data_path"]
    c2w = np.asarray(sensor_info["sensor2lidar_transform"], dtype=np.float32)
    return img_path, c2w


def _build_opengl_matrix(c2w: np.ndarray) -> np.ndarray:
    """Convert dataset c2w to OpenGL-style c2w for NeRF/Blender JSON."""
    result = np.array(c2w, dtype=np.float32)
    result[:3, 1:3] *= -1.0  # flip Y/Z axes
    return result


@dataclass
class LoaderConfig:
    data_root: Path
    cache_root: Path
    stage: str = "val"
    resolution: Tuple[int, int] = (112, 200)  # (H, W)
    version: str = "interp_12Hz_trainval"
    dataset_prefix: str = "/datasets/nuScenes"
    near: float = 0.01
    far: float = 100.0


class OmniSceneLoader:
    camera_types: Sequence[str] = (
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    )

    demo_bins: Sequence[str] = (
        "scenee7ef871f77f44331aefdebc24ec034b7_bin010",
        "scenee7ef871f77f44331aefdebc24ec034b7_bin200",
        "scene30ae9c1092f6404a9e6aa0589e809780_bin100",
        "scene84e056bd8e994362a37cba45c0f75558_bin100",
        "scene717053dec2ef4baa913ba1e24c09edff_bin000",
        "scene82240fd6d5ba4375815f8a7fa1561361_bin050",
        "scene724957e51f464a9aa64a16458443786d_bin000",
        "scened3c39710e9da42f48b605824ce2a1927_bin050",
        "scene034256c9639044f98da7562ef3de3646_bin000",
        "scenee0b14a8e11994763acba690bbcc3f56a_bin080",
        "scene7e2d9f38f8eb409ea57b3864bb4ed098_bin150",
        "scene50ff554b3ecb4d208849d042b7643715_bin000",
    )

    def __init__(self, cfg: LoaderConfig):
        self.cfg = cfg
        self.data_root = Path(cfg.data_root)
        self.version_root = self.data_root / cfg.version
        self.cache_root = Path(cfg.cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.stage = cfg.stage
        self.resolution = cfg.resolution
        self.resolution_tag = f"cache_{self.resolution[0]}x{self.resolution[1]}"
        self.stage_tokens = self._load_tokens()

    def list_tokens(self) -> Sequence[str]:
        return self.stage_tokens

    def prepare_scene(self, token: str, force_rebuild: bool = False) -> Path:
        tokens = list(self.list_tokens())
        index = tokens.index(token)
        scene_name = f"{index+1:02d}_{token}"
        scene_dir = self.cache_root / "omniscene_cache" / self.resolution_tag / scene_name
        meta_path = scene_dir / "meta.json"
        expected_meta = {
            "token": token,
            "stage": self.stage,
            "resolution": {"height": self.resolution[0], "width": self.resolution[1]},
            "train_views": 6,
            "test_views": 18,
        }

        if scene_dir.exists():
            if force_rebuild:
                shutil.rmtree(scene_dir)
            elif meta_path.exists():
                try:
                    stored = json.loads(meta_path.read_text())
                    if stored == expected_meta:
                        return scene_dir
                except json.JSONDecodeError:
                    pass
                shutil.rmtree(scene_dir)
            else:
                shutil.rmtree(scene_dir)

        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "images_train").mkdir()
        (scene_dir / "images_test").mkdir()

        bin_info = self._load_bin_info(token)
        train_views = self._collect_train_views(bin_info)
        test_views = self._collect_test_views(bin_info)
        test_views.extend(train_views)  # append input views for evaluation

        camera_angle_x = self._calculate_camera_angle(train_views[0]["fx"])
        self._write_images_and_json(
            scene_dir / "images_train",
            scene_dir / "transforms_train.json",
            train_views,
            camera_angle_x,
        )
        self._write_images_and_json(
            scene_dir / "images_test",
            scene_dir / "transforms_test.json",
            test_views,
            camera_angle_x,
        )

        meta_path.write_text(json.dumps(expected_meta, indent=2))
        return scene_dir

    # Internal helpers ----------------------------------------------------- #
    def _load_tokens(self) -> Sequence[str]:
        stage = self.stage.lower()
        if stage == "train":
            tokens = self._read_bins("bins_train_3.2m.json")
        elif stage == "val":
            tokens = self._read_bins("bins_val_3.2m.json")[:30000:3000][:10]
        elif stage == "test":
            tokens = self._read_bins("bins_val_3.2m.json")[0::14][:2048]
        elif stage == "demo":
            tokens = list(self.demo_bins)
        else:
            raise ValueError(f"Unsupported stage: {self.stage}")
        if not tokens:
            raise RuntimeError(f"No bin tokens found for stage {self.stage}")
        return tokens

    def _read_bins(self, filename: str) -> List[str]:
        json_path = self.version_root / filename
        with open(json_path, "r") as f:
            data = json.load(f)
        return data["bins"]

    def _load_bin_info(self, token: str) -> Dict:
        info_path = self.version_root / "bin_infos_3.2m" / f"{token}.pkl"
        with open(info_path, "rb") as f:
            return pickle.load(f)

    def _resolve_path(self, img_path: str) -> Path:
        normalized = img_path.replace("\\", "/")
        prefix = self.cfg.dataset_prefix
        if prefix and prefix in normalized:
            normalized = normalized.replace(prefix, str(self.data_root))
        elif not normalized.startswith(str(self.data_root)):
            normalized = str(self.data_root / normalized.lstrip("/"))
        return Path(normalized)

    def _load_image_and_intrinsics(self, img_path: Path) -> Dict:
        base_path = str(img_path)
        param_path = (
            base_path.replace("samples", "samples_param_small")
            .replace("sweeps", "sweeps_param_small")
            .replace(".jpg", ".json")
            .replace(".png", ".json")
        )
        with open(param_path, "r") as f:
            intrinsics = np.asarray(json.load(f)["camera_intrinsic"], dtype=np.float32)

        rgb_path = base_path.replace("samples", "samples_small").replace("sweeps", "sweeps_small")
        image = Image.open(rgb_path).convert("RGB")
        target_h, target_w = self.resolution
        scale_w = target_w / image.width
        scale_h = target_h / image.height
        image = image.resize((target_w, target_h), Image.BILINEAR)
        np_img = _ensure_hwc3(np.array(image, dtype=np.uint8))

        fx = intrinsics[0, 0] * scale_w
        fy = intrinsics[1, 1] * scale_h
        cx = intrinsics[0, 2] * scale_w
        cy = intrinsics[1, 2] * scale_h

        return {
            "image": np_img,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": target_w,
            "height": target_h,
        }

    def _collect_train_views(self, bin_info: Dict) -> List[Dict]:
        views = []
        for cam in self.camera_types:
            info = copy.deepcopy(bin_info["sensor_info"][cam][0])
            img_path_raw, c2w = _load_info(info)
            img_path = self._resolve_path(img_path_raw)
            img_data = self._load_image_and_intrinsics(img_path)
            views.append(
                {
                    "image": img_data["image"],
                    "transform_matrix": _build_opengl_matrix(c2w),
                    "fx": img_data["fx"],
                    "fy": img_data["fy"],
                    "cx": img_data["cx"],
                    "cy": img_data["cy"],
                    "width": img_data["width"],
                    "height": img_data["height"],
                }
            )
        return views

    def _collect_test_views(self, bin_info: Dict) -> List[Dict]:
        views = []
        for cam in self.camera_types:
            for frame_id in (1, 2):
                info = copy.deepcopy(bin_info["sensor_info"][cam][frame_id])
                img_path_raw, c2w = _load_info(info)
                img_path = self._resolve_path(img_path_raw)
                img_data = self._load_image_and_intrinsics(img_path)
                views.append(
                    {
                        "image": img_data["image"],
                        "transform_matrix": _build_opengl_matrix(c2w),
                        "fx": img_data["fx"],
                        "fy": img_data["fy"],
                        "cx": img_data["cx"],
                        "cy": img_data["cy"],
                        "width": img_data["width"],
                        "height": img_data["height"],
                    }
                )
        return views

    def _calculate_camera_angle(self, fx: float) -> float:
        width = self.resolution[1]
        return float(2.0 * math.atan(width / (2.0 * fx)))

    def _write_images_and_json(
        self,
        image_dir: Path,
        json_path: Path,
        views: List[Dict],
        camera_angle_x: float,
    ) -> None:
        frames = []
        for idx, view in enumerate(views):
            file_name = f"{idx:06d}.png"
            file_stem = f"{image_dir.name}/{file_name[:-4]}"

            Image.fromarray(view["image"]).save(image_dir / file_name)
            frames.append(
                {
                    "file_path": file_stem,
                    "transform_matrix": view["transform_matrix"].tolist(),
                    "fl_x": float(view["fx"]),
                    "fl_y": float(view["fy"]),
                    "cx": float(view["cx"]),
                    "cy": float(view["cy"]),
                    "w": view["width"],
                    "h": view["height"],
                    "near": self.cfg.near,
                    "far": self.cfg.far,
                }
            )

        payload = {"camera_angle_x": camera_angle_x, "frames": frames}
        json_path.write_text(json.dumps(payload, indent=2))
