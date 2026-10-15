'PNG 데이터, 상별 팔레트 변환 및 이미지별 IoU.'

from __future__ import annotations

import hashlib


import zipfile

from pathlib import Path, PurePosixPath

import numpy as np

from PIL import Image

CLASS_NAMES = ("Primary Si", "Al3Ni", "Eutectic Si", "Al")

PALETTE = np.array([(76, 178, 76), (25, 76, 153),
                    (204, 204, 204), (153, 127, 76)], dtype=np.uint8)

EXPECTED_COUNTS = {"train": 70, "valid": 20, "test": 10}

def read_image(path):
    with Image.open(path) as im:
        im.load()
        return np.array(im.convert("RGB"), dtype=np.uint8)

def decode_mask(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA mask, got {rgb.shape}")
    if rgb.shape[-1] == 4 and np.any(rgb[..., 3] != 255):
        raise ValueError("Non-opaque mask alpha; inspect before converting.")
    rgb = rgb[..., :3]
    result = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for label, color in enumerate(PALETTE):
        result[np.all(rgb == color, axis=-1)] = label
    unknown = result == 255
    if np.any(unknown):
        colors = np.unique(rgb[unknown], axis=0)[:10].tolist()
        raise ValueError(f"Unknown mask colors: {colors}; do not interpolate masks.")
    return result

def read_mask(path):
    with Image.open(path) as im:
        return decode_mask(np.array(im.convert("RGBA")))

def _check_labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be a 2D integer array.")
    if labels.size == 0 or labels.min() < 0 or labels.max() >= len(PALETTE):
        raise ValueError("Labels must be nonempty and in [0, 3].")
    return labels

def save_rgb_mask(labels, path):
    labels = _check_labels(labels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(PALETTE[labels]).save(path)

def discover_split(data_root, split):

    split = {"validation": "valid", "val": "valid"}.get(split, split)
    if split not in EXPECTED_COUNTS:
        raise ValueError(f"Unknown split: {split}")
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    images = sorted(p for p in root.rglob("*_image.png")
                    if p.parent.name == "images"
                    and split in p.relative_to(root).parts[:-2])
    if not images:
        raise FileNotFoundError(f"No {split}/.../images/*_image.png under {root}")
    records, seen = [], set()
    expected_masks = set()
    for image in images:
        if image.name in seen:
            raise ValueError(f"Duplicate image name in {split}: {image.name}")
        seen.add(image.name)
        mask = image.parent.parent / "masks" / image.name.replace("_image.png", "_mask.png")
        if split == "test":
            records.append((image, None))
        else:
            if not mask.is_file():
                raise FileNotFoundError(mask)
            records.append((image, mask))
            expected_masks.add(mask)
    if split != "test":
        actual_masks = {p for p in root.rglob("*_mask.png")
                        if p.parent.name == "masks"
                        and split in p.relative_to(root).parts[:-2]}
        if actual_masks != expected_masks:
            raise ValueError(f"Unmatched masks in {split}: {actual_masks - expected_masks}")
    return records

def image_iou(true, pred, absent_policy="ignore"):
    "Return one image's macro IoU, not a dataset-wide confusion-matrix IoU."
    true, pred = _check_labels(true), _check_labels(pred)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {true.shape} != {pred.shape}")
    if absent_policy not in ("ignore", "zero"):
        raise ValueError("absent_policy must be ignore or zero")
    cm = np.bincount(4 * true.astype(np.int64).ravel() + pred.ravel(),
                     minlength=16).reshape(4, 4)
    intersection = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    scores = [float(i / u) if u else None for i, u in zip(intersection, union)]
    selected = [v for v in scores if v is not None] if absent_policy == "ignore" else [v or 0.0 for v in scores]
    return {"miou": float(np.mean(selected)), "class_iou": scores,
            "intersection": intersection.tolist(), "union": union.tolist()}

def validate_data(data_root, strict_counts=True):

    root = Path(data_root)
    report = {"classes": list(CLASS_NAMES), "palette": PALETTE.tolist(), "splits": {}}
    for split, expected in EXPECTED_COUNTS.items():
        pairs = discover_split(root, split)
        if strict_counts and len(pairs) != expected:
            raise ValueError(f"{split}: expected {expected} images, got {len(pairs)}")
        counts = np.zeros(4, dtype=np.int64)
        sizes = set()
        absent = []
        for image_path, mask_path in pairs:
            image = read_image(image_path)
            sizes.add((image.shape[1], image.shape[0]))
            if mask_path is not None:
                mask = read_mask(mask_path)
                if image.shape[:2] != mask.shape:
                    raise ValueError(f"Image/mask dimensions differ: {image_path.name}")
                one = np.bincount(mask.ravel(), minlength=4)
                counts += one
                if (one == 0).any():
                    absent.append({"file": image_path.name,
                                   "absent_classes": [CLASS_NAMES[i] for i in np.flatnonzero(one == 0)]})
        report["splits"][split] = {"images": len(pairs), "sizes_wh": sorted(sizes),
                                    "pixel_counts": counts.tolist(),
                                    "pixel_fractions": (counts / counts.sum()).tolist() if counts.sum() else None,
                                    "images_with_absent_classes": absent}
    return report

def extract_archives(source_dir, destination):

    source_dir, destination = Path(source_dir), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for split in EXPECTED_COUNTS:
        archive = source_dir / f"{split}.zip"
        if not archive.is_file():
            raise FileNotFoundError(f"Upload train.zip, valid.zip, test.zip: missing {archive}")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        target = destination / split
        stamp = target / ".archive_sha256"
        if stamp.exists():
            if stamp.read_text().strip() == digest:
                continue
            raise ValueError(f"Changed archive: use a new destination instead of {target}")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            members = zf.infolist()
            for member in members:
                name = PurePosixPath(member.filename.replace("\\", "/"))
                if name.is_absolute() or ".." in name.parts or any(":" in part for part in name.parts):
                    raise ValueError(f"Unsafe ZIP path: {member.filename}")
                resolved = target.joinpath(*name.parts).resolve()
                if not resolved.is_relative_to(target.resolve()):
                    raise ValueError(f"ZIP path escapes destination: {member.filename}")
                if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError("ZIP symlinks are not accepted.")
            bad = zf.testzip()
            if bad is not None:
                raise ValueError(f"ZIP CRC failure: {bad}")
            zf.extractall(target)
        stamp.write_text(digest + "\n", encoding="utf-8")
    return destination
