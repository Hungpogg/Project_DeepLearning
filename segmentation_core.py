"""Inference and exports matching unet-efficientnet_B3.ipynb (20 classes)."""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch


CLASSES = [
    ("unlabeled", (0, 0, 0)), ("road", (128, 64, 128)),
    ("sidewalk", (244, 35, 232)), ("building", (70, 70, 70)),
    ("wall", (102, 102, 156)), ("fence", (190, 153, 153)),
    ("pole", (153, 153, 153)), ("traffic light", (250, 170, 30)),
    ("traffic sign", (220, 220, 0)), ("vegetation", (107, 142, 35)),
    ("terrain", (152, 251, 152)), ("sky", (70, 130, 180)),
    ("person", (220, 20, 60)), ("rider", (255, 0, 0)),
    ("car", (0, 0, 142)), ("truck", (0, 0, 70)),
    ("bus", (0, 60, 100)), ("train", (0, 80, 100)),
    ("motorcycle", (0, 0, 230)), ("bicycle", (119, 11, 32)),
]
PALETTE = np.array([color for _, color in CLASSES], dtype=np.uint8)
NAMES = np.array([name for name, _ in CLASSES])
ENCODERS = tuple(f"efficientnet-b{i}" for i in range(8))
IMAGE_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Prediction:
    name: str
    image: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray
    ground_truth: np.ndarray | None
    source_size: tuple[int, int]


def mask_to_class(rgb: np.ndarray) -> np.ndarray:
    """Notebook's L1 nearest-palette mapping, chunked to bound memory."""
    flat = rgb.reshape(-1, 3).astype(np.int16)
    output = np.empty(len(flat), dtype=np.uint8)
    for start in range(0, len(flat), 8192):
        diff = np.abs(flat[start:start + 8192, None] - PALETTE.astype(np.int16))
        output[start:start + 8192] = diff.sum(axis=2).argmin(axis=1)
    return output.reshape(rgb.shape[:2])


def prepare_image(data: bytes, paired: bool) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int]]:
    with Image.open(io.BytesIO(data)) as opened:
        rgb = np.array(opened.convert("RGB"))
    height, width = rgb.shape[:2]
    gt = None
    if paired:
        if width != 2 * height:
            raise ValueError(
                f"Ảnh ghép cần tỷ lệ rộng:cao = 2:1; ảnh này là {width}×{height}. "
                "Nếu đây là ảnh RGB riêng, hãy đổi định dạng đầu vào."
            )
        half = width // 2
        # Decode palette before resize, matching the notebook dataset.
        gt = mask_to_class(rgb[:, half:])
        gt = cv2.resize(gt, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        rgb = rgb[:, :half]
    source_size = (rgb.shape[1], rgb.shape[0])
    image = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return image, gt, source_size


def image_tensor(image: np.ndarray) -> torch.Tensor:
    # Albumentations Normalize uses (image - mean*255) / (std*255).
    normalized = (image.astype(np.float32) - MEAN * 255.0) * (1.0 / (STD * 255.0))
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0)


def detect_encoder(state: Mapping) -> str:
    """Identify this project's EfficientNet variants from architecture shapes."""
    head = state.get("encoder._conv_head.weight")
    if not isinstance(head, torch.Tensor):
        raise ValueError("Không nhận diện được encoder. Hãy chọn thủ công B0–B7.")
    channels = int(head.shape[0])
    if channels == 1280:
        blocks = {int(m.group(1)) for k in state if (m := re.match(r"encoder\._blocks\.(\d+)\.", k))}
        variant = {16: 0, 23: 1}.get(len(blocks))
    else:
        variant = {1408: 2, 1536: 3, 1792: 4, 2048: 5, 2304: 6, 2560: 7}.get(channels)
    if variant is None:
        raise ValueError("Checkpoint không khớp kiến trúc EfficientNet B0–B7 của notebook.")
    return ENCODERS[variant]


def load_model(source: str | Path | bytes, encoder: str = "auto", device: str = "cpu"):
    """Strict load; never silently use partial/random weights or unsafe pickle."""
    file = io.BytesIO(source) if isinstance(source, bytes) else source
    checkpoint = torch.load(file, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint cần là model_state_dict hoặc dictionary chứa model_state_dict.")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Không tìm thấy trọng số trong checkpoint.")
    if all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    selected = detect_encoder(state) if encoder == "auto" else encoder
    if selected not in ENCODERS:
        raise ValueError("Encoder phải thuộc EfficientNet B0–B7.")
    model = smp.Unet(
        encoder_name=selected, encoder_weights=None, in_channels=3,
        classes=len(CLASSES), activation=None, decoder_use_norm="batchnorm",
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Trọng số không khớp UNet + {selected} / 20 lớp. "
            "Kiểm tra encoder và phiên bản segmentation-models-pytorch khi train."
        ) from exc
    model.to(device).eval()
    metadata = {"encoder": selected, "device": device, "input_size": [256, 256]}
    for key in ("epoch", "val_iou"):
        value = checkpoint.get(key)
        if isinstance(value, (int, float)):
            metadata[key] = value
    return model, metadata


@torch.inference_mode()
def predict(model, data: bytes, name: str, paired: bool, device: str) -> Prediction:
    image, gt, size = prepare_image(data, paired)
    logits = model(image_tensor(image).to(device))
    if tuple(logits.shape) != (1, len(CLASSES), IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Kích thước output không hợp lệ: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError("Model trả về NaN/Inf; hãy kiểm tra checkpoint.")
    confidence, labels = logits.softmax(dim=1).max(dim=1)
    return Prediction(name, image, labels[0].cpu().numpy().astype(np.uint8),
                      confidence[0].cpu().numpy(), gt, size)


def overlay(result: Prediction, alpha: float = 0.45, class_id: int | None = None) -> np.ndarray:
    blended = np.rint((1 - alpha) * result.image + alpha * PALETTE[result.labels]).astype(np.uint8)
    if class_id is not None:
        blended = np.where((result.labels == class_id)[..., None], blended, result.image)
    return blended


def class_statistics(result: Prediction) -> list[dict]:
    counts = np.bincount(result.labels.ravel(), minlength=len(CLASSES))
    rows = []
    for i, (name, _) in enumerate(CLASSES):
        row = {"ID": i, "Lớp": name, "Pixel dự đoán": int(counts[i]),
               "Diện tích (%)": round(float(counts[i]) / result.labels.size * 100, 2)}
        if result.ground_truth is not None:
            truth_mask = result.ground_truth == i
            row["Pixel ground truth"] = int(truth_mask.sum())
            intersection = int(((result.labels == i) & truth_mask).sum())
            union = int(counts[i]) + row["Pixel ground truth"] - intersection
            row["IoU (%)"] = round(100 * intersection / union, 2) if union else None
        else:
            row["IoU (%)"] = None
        rows.append(row)
    return rows


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def export_zip(results: list[Prediction], metadata: dict) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("run.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        archive.writestr("classes.json", json.dumps([
            {"id": i, "name": name, "rgb": color} for i, (name, color) in enumerate(CLASSES)
        ], indent=2))
        for index, result in enumerate(results, 1):
            stem = re.sub(r"[^\w.-]+", "_", Path(result.name).stem)[:80] or "image"
            prefix = f"{index:03d}_{stem}/"
            for name, data in {
                "input.png": result.image, "prediction_ids.png": result.labels,
                "prediction_color.png": PALETTE[result.labels], "overlay.png": overlay(result),
            }.items():
                archive.writestr(prefix + name, png_bytes(data))
            if result.ground_truth is not None:
                archive.writestr(prefix + "ground_truth_ids.png", png_bytes(result.ground_truth))
                archive.writestr(prefix + "ground_truth_color.png", png_bytes(PALETTE[result.ground_truth]))
                error = np.zeros_like(result.image)
                error[result.labels != result.ground_truth] = (239, 68, 68)
                archive.writestr(prefix + "disagreement.png", png_bytes(error))
            confidence = io.BytesIO()
            np.save(confidence, result.confidence, allow_pickle=False)
            archive.writestr(prefix + "confidence.npy", confidence.getvalue())
            stats = class_statistics(result)
            table = io.StringIO()
            writer = csv.DictWriter(table, fieldnames=list(stats[0]))
            writer.writeheader()
            writer.writerows(stats)
            archive.writestr(prefix + "class_pixels.csv", table.getvalue().encode("utf-8-sig"))
            archive.writestr(prefix + "image.json", json.dumps({
                "name": result.name, "source_size": result.source_size,
                "output_size": [IMAGE_SIZE, IMAGE_SIZE],
            }, ensure_ascii=False))
    return output.getvalue()
