'DINOv3 control 모델의 추가 학습과 추론.'
from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import psutil
from PIL import Image
from tqdm.auto import tqdm
import sympy
import sympy.core
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import transformers
import komap_specialist_data as kd

MODEL_FAMILY = "komap_pretrained_unet_multiclass_ram_v1"
CLASS_SLUGS = ("PrimarySi", "Al3Ni", "EutecticSi", "Al")


@dataclass
class Config:
    architecture: str = "dinov3"
    target_class: int = -1
    encoder: str = "efficientnet-b3"
    encoder_weights: str = "imagenet"
    patch: int = 480
    batch: int = 8
    patches_per_image: int = 64
    epochs: int = 20
    encoder_lr: float = 1e-5
    decoder_lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    freeze_encoder_epochs: int = 0
    eval_every: int = 2
    eval_stride: int = 240
    seed: int = 42
    dice_weight: float = 0.5
    random_crop_fraction: float = 0.5
    eutectic_crop_fraction: float = 0.25
    scale_probability: float = 0.3
    scale_min: float = 0.9
    scale_max: float = 1.1
    brightness_contrast_probability: float = 0.5
    brightness: float = 0.10
    contrast: float = 0.10
    gamma_probability: float = 0.3
    gamma_min: float = 0.9
    gamma_max: float = 1.1
    noise_probability: float = 0.1
    noise_std: float = 0.008
    use_ema: bool = True
    ema_decay: float = 0.99
    ema_every: int = 4
    absent_policy: str = "ignore"
    pretrained_model_id: str = ""
    hard_crop_fraction: float = 0.0
    ohem_fraction: float = 1.0
    validation_tta_rotations: int = 4

    def validate(self):
        if self.architecture != "dinov3":
            raise ValueError("Unsupported architecture")
        if (self.architecture == "specialist" and self.target_class not in (1, 2)) or (self.architecture != "specialist" and self.target_class != -1):
            raise ValueError("Only specialist models use target_class=1 or 2; others use -1")
        for name in ("patch", "batch", "patches_per_image", "epochs", "eval_every", "eval_stride", "ema_every", "warmup_epochs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.hard_crop_fraction <= 1 or not 0 < self.ohem_fraction <= 1:
            raise ValueError("Invalid hard_crop_fraction or ohem_fraction")
        if self.validation_tta_rotations not in (1,4):
            raise ValueError("validation_tta_rotations must be 1 or 4")
        if self.patch % 32 or self.eval_stride > self.patch or self.patch < 64:
            raise ValueError("patch must be a multiple of 32, stride <= patch")
        if self.absent_policy not in ("ignore", "zero"):
            raise ValueError("absent_policy must be ignore or zero")
        if not 0 <= self.freeze_encoder_epochs < self.epochs:
            raise ValueError("freeze_encoder_epochs must be less than epochs")
        if self.warmup_epochs >= self.epochs or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid warmup or seed")
        for name in ("random_crop_fraction", "eutectic_crop_fraction", "scale_probability", "brightness_contrast_probability", "gamma_probability", "noise_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"Invalid {name}")
        if self.random_crop_fraction + self.eutectic_crop_fraction > 1:
            raise ValueError("Crop fractions sum exceeds one")
        if not 0 < self.scale_min <= self.scale_max or not 0 < self.gamma_min <= self.gamma_max:
            raise ValueError("Invalid scale/gamma range")
        if not 0 < self.ema_decay < 1 or self.encoder_lr <= 0 or self.decoder_lr <= 0:
            raise ValueError("Invalid EMA/learning rate")
        if self.weight_decay < 0 or self.dice_weight < 0:
            raise ValueError("Invalid loss/weight decay")
        for value in asdict(self).values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("All numeric configuration values must be finite")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def save_checkpoint(path, state, local_run):
    path, local_run = Path(path), Path(local_run)
    local_run.mkdir(parents=True, exist_ok=True)
    local = local_run / path.name
    torch.save(state, local)
    destination = path.with_name(path.name + ".tmp")
    shutil.copyfile(local, destination)
    os.replace(destination, path)


def _decode_record(pair):
    ip, mp = pair
    image_bytes = ip.read_bytes()
    with Image.open(io.BytesIO(image_bytes)) as im:
        image = np.array(im.convert("RGB"), dtype=np.uint8)
    mask, mask_sha = None, None
    if mp is not None:
        mask_bytes = mp.read_bytes()
        with Image.open(io.BytesIO(mask_bytes)) as im:
            mask = kd.decode_mask(np.array(im.convert("RGBA")))
        mask_sha = hashlib.sha256(mask_bytes).hexdigest()
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask dimensions differ: {ip.name}")
        mask.setflags(write=False)
    image.setflags(write=False)
    return {"name": ip.name, "image": image, "mask": mask,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(), "mask_sha256": mask_sha}


def prepare_ram(project_root, data_root, strict_counts=True):
    'All PNGs are decoded exactly once per prepare call, including valid/test.'
    kd.extract_archives(Path(project_root) / "data", data_root)
    cv2.setNumThreads(0)
    cache, fingerprints = {}, []
    for split in ("train", "valid", "test"):
        pairs = kd.discover_split(data_root, split)
        if strict_counts and len(pairs) != kd.EXPECTED_COUNTS[split]:
            raise ValueError(f"{split}: expected {kd.EXPECTED_COUNTS[split]}, got {len(pairs)}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            cache[split] = list(tqdm(pool.map(_decode_record, pairs), total=len(pairs), desc=f"{split}: PNG → RAM"))
        for rec in cache[split]:
            fingerprints.append({"split": split, "image": rec["name"], "image_sha256": rec["image_sha256"],
                                 "mask_sha256": rec["mask_sha256"], "shape": list(rec["image"].shape[:2])})
    cache_bytes = sum(r["image"].nbytes + (r["mask"].nbytes if r["mask"] is not None else 0)
                      for values in cache.values() for r in values)
    sha = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
    report = {"data_sha256": sha, "files": fingerprints, "class_names": list(kd.CLASS_NAMES),
              "palette": kd.PALETTE.tolist(), "counts": {s: len(v) for s, v in cache.items()},
              "decoded_cache_bytes": cache_bytes, "train_class_pixels": sum(
                  (np.bincount(r["mask"].ravel(), minlength=4) for r in cache["train"]), np.zeros(4,dtype=np.int64)).tolist()}
    print(f"RAM 캐시 완료: {cache_bytes / 2**30:.3f} GiB; 사용 가능한 RAM {psutil.virtual_memory().available / 2**30:.2f} GiB", flush=True)
    print("클래스:", dict(enumerate(kd.CLASS_NAMES)))
    print("학습·검증·추론에서 PNG를 다시 디코딩하지 않습니다.")
    return cache, report


class RAMCrops(Dataset):
    def __init__(self, records, config, hard_coordinates=None):
        self.records, self.config = records, config
        self.hard_coordinates = hard_coordinates["positions"] if hard_coordinates is not None else None
        if config.hard_crop_fraction > 0 and self.hard_coordinates is None:
            raise ValueError("Prepare train hard coordinates first")
        if self.hard_coordinates is not None and len(self.hard_coordinates) != len(records):
            raise ValueError("Hard coordinates do not match train records")
        self.epoch_size = len(records) * config.patches_per_image
        self.coordinates = []
        rng = np.random.default_rng(config.seed)
        for rec in records:
            coords = {}

            for c in (0, 1, 2):
                index = np.flatnonzero(rec["mask"].ravel() == c)
                if index.size > 50000:
                    index = rng.choice(index, 50000, replace=False)
                coords[c] = index.astype(np.int32)
                coords[c].setflags(write=False)
            self.coordinates.append(coords)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, key):
        epoch, index = divmod(int(key), self.epoch_size)
        c = self.config
        rng = np.random.default_rng(np.random.SeedSequence([c.seed, epoch, index]))
        rec_id = index % len(self.records)
        rec = self.records[rec_id]
        image, mask = rec["image"], rec["mask"]
        h, w = mask.shape
        scale = float(rng.uniform(c.scale_min, c.scale_max)) if rng.random() < c.scale_probability else 1.0
        crop = max(32, int(round(c.patch / scale)))
        branch_rng = np.random.default_rng(np.random.SeedSequence([c.seed, epoch, index, 777]))
        groups = [v for v in self.hard_coordinates[rec_id].values() if v.size] if self.hard_coordinates is not None else []
        use_hard = c.hard_crop_fraction > 0 and bool(groups) and branch_rng.random() < c.hard_crop_fraction
        if use_hard:

            values = groups[int(branch_rng.integers(len(groups)))]
            pos = int(values[int(branch_rng.integers(values.size))])
            cy, cx = divmod(pos, w)
            jitter = max(1, crop // 4)
            y = int(np.clip(cy - crop // 2 + branch_rng.integers(-jitter, jitter + 1), 0, max(0, h - crop)))
            x = int(np.clip(cx - crop // 2 + branch_rng.integers(-jitter, jitter + 1), 0, max(0, w - crop)))
        else:
            choice = float(rng.random())
            target_class = None
            if c.random_crop_fraction <= choice < c.random_crop_fraction + c.eutectic_crop_fraction:
                target_class = 2
            elif choice >= c.random_crop_fraction + c.eutectic_crop_fraction:

                target_class = 1 if rng.random() < 0.7 else 0
            positions = self.coordinates[rec_id].get(target_class)
            if positions is not None and positions.size:
                pos = int(positions[rng.integers(positions.size)])
                cy, cx = divmod(pos, w)

                y = int(np.clip(cy - rng.integers(crop), 0, max(0, h - crop)))
                x = int(np.clip(cx - rng.integers(crop), 0, max(0, w - crop)))
            else:
                y, x = int(rng.integers(max(1, h - crop + 1))), int(rng.integers(max(1, w - crop + 1)))
        rgb = image[y:y + crop, x:x + crop].copy()
        label = mask[y:y + crop, x:x + crop].copy()
        pad_h, pad_w = crop - label.shape[0], crop - label.shape[1]
        if pad_h or pad_w:
            rgb = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
        if crop != c.patch:
            rgb = cv2.resize(rgb, (c.patch, c.patch), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (c.patch, c.patch), interpolation=cv2.INTER_NEAREST)

        k = int(rng.integers(4))
        rgb, label = np.rot90(rgb, k), np.rot90(label, k)
        if rng.random() < 0.5:
            rgb, label = np.fliplr(rgb), np.fliplr(label)
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))), torch.from_numpy(np.ascontiguousarray(label))


class EpochSampler(Sampler):
    def __init__(self, dataset, seed):
        self.size, self.seed, self.epoch = len(dataset), seed, 0

    def __len__(self):
        return self.size

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 987]))
        for index in rng.permutation(self.size):
            yield self.epoch * self.size + int(index)


def _worker_init(_):
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def make_loader(cache, c, num_workers=2, hard_coordinates=None):
    dataset = RAMCrops(cache["train"], c, hard_coordinates)
    sampler = EpochSampler(dataset, c.seed)
    kwargs = dict(batch_size=c.batch, sampler=sampler, num_workers=num_workers, pin_memory=True,
                  drop_last=False, worker_init_fn=_worker_init)
    if num_workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="fork")
    return DataLoader(dataset, **kwargs), sampler


class GPUInput:
    def __init__(self, device, normalize=True):
        self.device = device
        self.normalize = normalize
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def __call__(self, batch, config=None):
        x = batch.to(self.device, non_blocking=True).float().div_(255)
        if config is not None:
            c, b = config, x.shape[0]
            shape = (b, 1, 1, 1)
            active = (torch.rand(shape, device=self.device) < c.brightness_contrast_probability)
            contrast = 1 + (torch.rand(shape, device=self.device) * 2 - 1) * c.contrast * active
            brightness = (torch.rand(shape, device=self.device) * 2 - 1) * c.brightness * active
            x = ((x - 0.5) * contrast + 0.5 + brightness).clamp_(0, 1)
            active = (torch.rand(shape, device=self.device) < c.gamma_probability)
            gamma = torch.rand(shape, device=self.device) * (c.gamma_max - c.gamma_min) + c.gamma_min
            gamma = torch.where(active, gamma, torch.ones_like(gamma))
            x = x.pow(gamma)
            active = (torch.rand(shape, device=self.device) < c.noise_probability)
            x = (x + torch.randn_like(x) * (c.noise_std * active)).clamp_(0, 1)
        x = (x - self.mean) / self.std if self.normalize else x
        return x.contiguous(memory_format=torch.channels_last)


def build_model(c, pretrained, backbone_config=None):
    if c.architecture != "dinov3":
        raise ValueError("Only DINOv3 is used in this controlled comparison")
    return DinoFineDecoder(c.pretrained_model_id, pretrained, backbone_config)


def backbone_config_of(model):
    encoder = getattr(model, "encoder", None)
    return encoder.config.to_dict() if encoder is not None and hasattr(encoder, "config") else None


class DetailBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                         nn.GroupNorm(8, out_channels), nn.GELU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(8, out_channels), nn.GELU())


class DinoDetailPath(nn.Module):
    def __init__(self, feature_channels):
        super().__init__()
        self.stage1 = DetailBlock(3, 32, 2)
        self.stage2 = DetailBlock(32, 64, 2)
        self.stage3 = DetailBlock(64, 96, 2)
        self.context = nn.Sequential(nn.Conv2d(feature_channels * 4, 192, 1, bias=False), nn.GroupNorm(8, 192), nn.GELU())
        self.up8 = DetailBlock(192 + 96, 128)
        self.up4 = DetailBlock(128 + 64, 64)
        self.up2 = DetailBlock(64 + 32, 32)
        self.up1 = DetailBlock(32 + 3, 16)

    def forward(self, x, features):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        context = self.context(torch.cat(features, dim=1))
        up = self.up8(torch.cat([F.interpolate(context, size=s3.shape[-2:], mode="bilinear", align_corners=False), s3], dim=1))
        up = self.up4(torch.cat([F.interpolate(up, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], dim=1))
        up = self.up2(torch.cat([F.interpolate(up, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], dim=1))
        return self.up1(torch.cat([F.interpolate(up, size=x.shape[-2:], mode="bilinear", align_corners=False), x], dim=1))


class DinoFineDecoder(nn.Module):
    def __init__(self, model_id, pretrained, backbone_config=None):
        super().__init__()
        from transformers import AutoModel, DINOv3ViTConfig
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_id, token=os.environ.get("HF_TOKEN") or None)
        else:
            if backbone_config is None:
                raise ValueError("Saved DINO backbone configuration is missing")
            self.encoder = AutoModel.from_config(DINOv3ViTConfig.from_dict(backbone_config))
        if self.encoder.config.patch_size != 16 or self.encoder.config.hidden_size != 384:
            raise ValueError("This decoder requires the DINOv3 ViT-S/16 model")
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.decoder = DinoDetailPath(self.encoder.config.hidden_size)
        self.segmentation_head = nn.Conv2d(16, 4, 1)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            output = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        b, _, h, w = x.shape
        prefix = 1 + self.encoder.config.num_register_tokens
        features = [output.hidden_states[i][:, prefix:].transpose(1, 2).reshape(b, 384, h // 16, w // 16)
                    for i in (3, 6, 9, 12)]
        return self.segmentation_head(self.decoder(x, features))


def segmentation_loss(logits, target, dice_weight, target_class=-1, ohem_fraction=1.0):
    logits = logits.float()
    if target_class >= 0:
        valid = target != 255
        truth = (target == target_class).float()
        raw = logits[:, 0]
        bce = (F.binary_cross_entropy_with_logits(raw, truth, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
        p = raw.sigmoid() * valid
        intersection = (p * truth).sum(dim=(1, 2))
        denominator = p.sum(dim=(1, 2)) + (truth * valid).sum(dim=(1, 2))
        dice = 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)
        return bce + dice_weight * dice.mean()
    if ohem_fraction < 1.0:
        pixel_loss = F.cross_entropy(logits, target, ignore_index=255, reduction="none")
        valid_flat = (target != 255).flatten(1)
        flat = pixel_loss.flatten(1).masked_fill(~valid_flat, -torch.inf)
        maximum_kept = max(1, math.ceil(flat.shape[1] * ohem_fraction))
        hard = torch.topk(flat, maximum_kept, dim=1, largest=True, sorted=True).values
        kept = torch.ceil(valid_flat.sum(dim=1).float() * ohem_fraction).long().clamp_min(1)
        selected = (torch.arange(maximum_kept, device=logits.device)[None] < kept[:, None]) & torch.isfinite(hard)
        per_image_ce = torch.where(selected, hard, 0.0).sum(dim=1) / selected.sum(dim=1).clamp_min(1)
        ce = per_image_ce.mean()
    else:
        ce = F.cross_entropy(logits, target, ignore_index=255)
    valid = (target != 255).unsqueeze(1)
    one_hot = F.one_hot(target.clamp(max=3), 4).permute(0, 3, 1, 2).float() * valid
    probability = logits.softmax(dim=1) * valid
    intersection = (probability * one_hot).sum(dim=(2, 3))
    target_count = one_hot.sum(dim=(2, 3))
    denominator = probability.sum(dim=(2, 3)) + target_count
    loss = 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)
    present = target_count > 0
    per_image = (loss * present).sum(dim=1) / present.sum(dim=1).clamp_min(1)
    return ce + dice_weight * per_image.mean()


def _tile_starts(size, patch, stride):
    starts = list(range(0, max(1, size - patch + 1), stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


@torch.inference_mode()
def predict_probability(model, image, c, gpu_input, amp_dtype, tta_rotations=1):
    if tta_rotations not in (1, 4):
        raise ValueError("tta_rotations must be 1 or 4")
    model.eval()
    p = c.patch
    h, w = image.shape[:2]
    padded = cv2.copyMakeBorder(image, 0, max(0, p - h), 0, max(0, p - w), cv2.BORDER_REFLECT_101)
    ph, pw = padded.shape[:2]
    axis = np.arange(p, dtype=np.float32) - (p - 1) / 2
    weight = np.exp(-0.5 * ((axis[:, None] / (p / 8)) ** 2 + (axis[None, :] / (p / 8)) ** 2))
    weight = np.maximum(weight, 1e-6).astype(np.float32)
    total = np.zeros((1 if c.target_class >= 0 else 4, ph, pw), dtype=np.float32)
    denominator = np.zeros((ph, pw), dtype=np.float32)
    locations = [(y, x) for y in _tile_starts(ph, p, c.eval_stride) for x in _tile_starts(pw, p, c.eval_stride)]
    for start in range(0, len(locations), c.batch):
        chunk = locations[start:start + c.batch]
        array = np.stack([padded[y:y+p, x:x+p].transpose(2, 0, 1) for y, x in chunk])
        batch = gpu_input(torch.from_numpy(np.ascontiguousarray(array)))
        combined = None
        for k in range(tta_rotations):
            rotated = torch.rot90(batch, k, dims=(-2, -1)).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(rotated)
            prob = logits.float().sigmoid() if c.target_class >= 0 else logits.float().softmax(dim=1)
            prob = torch.rot90(prob, -k, dims=(-2, -1))
            combined = prob if combined is None else combined + prob
        result = combined.div_(tta_rotations).cpu().numpy()
        for probability, (y, x) in zip(result, chunk):
            total[:, y:y+p, x:x+p] += probability * weight[None]
            denominator[y:y+p, x:x+p] += weight
    result = total[:, :h, :w] / denominator[None, :h, :w]
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1.00001):
        raise FloatingPointError("Invalid prediction probabilities")
    return result


def evaluate_labels(name, target, pred):
    cm = np.bincount((4 * target.astype(np.int64) + pred).ravel(), minlength=16).reshape(4, 4)
    ignore, zero = kd.image_iou(target, pred, "ignore"), kd.image_iou(target, pred, "zero")
    row = {"image": name, "height": target.shape[0], "width": target.shape[1],
           "miou_ignore": ignore["miou"], "miou_zero": zero["miou"],
           "Al3Ni_to_EutecticSi_pixels": int(cm[1, 2]), "EutecticSi_to_Al3Ni_pixels": int(cm[2, 1])}
    for i, slug in enumerate(CLASS_SLUGS):
        row[slug + "_iou"] = ignore["class_iou"][i]
        tp = int(cm[i, i])
        fp, fn = int(cm[:, i].sum() - tp), int(cm[i].sum() - tp)
        row[slug + "_fp"] = fp
        row[slug + "_fn"] = fn
        row[slug + "_precision"] = tp / (tp + fp) if tp + fp else None
        row[slug + "_recall"] = tp / (tp + fn) if tp + fn else None
    return row, cm


def summarize(rows, cm):
    return {"images": len(rows), "miou_ignore": float(np.mean([r["miou_ignore"] for r in rows])),
            "miou_zero": float(np.mean([r["miou_zero"] for r in rows])),
            "class_iou_ignore": {slug: float(np.mean(v)) if (v := [r[slug + "_iou"] for r in rows if r[slug + "_iou"] is not None]) else None
                                 for slug in CLASS_SLUGS},
            "confusion_gt_rows_pred_columns": cm.tolist(), "class_names": list(kd.CLASS_NAMES),
            "aggregation": "mean of per-image mIoU, not pooled pixel IoU",
            "absent_class_note": "union==0: both ignore and zero are reported; official policy is unconfirmed"}


def evaluate(model, records, c, gpu_input, amp_dtype, fusion_context=None):
    rows, cm = [], np.zeros((4, 4), dtype=np.int64)
    for rec in records:
        probability = predict_probability(model, rec["image"], c, gpu_input, amp_dtype, c.validation_tta_rotations)
        if c.target_class >= 0:
            if fusion_context is None:
                raise ValueError("Specialist best selection requires fixed four-class baseline probabilities")
            baseline = fusion_context["probabilities"][rec["name"]].copy()
            baseline[c.target_class] = probability[0]
            probability = baseline * np.asarray(fusion_context["weights"], dtype=np.float32)[:, None, None]
        row, one = evaluate_labels(rec["name"], rec["mask"], probability.argmax(axis=0).astype(np.uint8))
        rows.append(row)
        cm += one
    return summarize(rows, cm), rows


def _identity(c, report):
    return {"family": MODEL_FAMILY, "config": asdict(c), "data_sha256": report["data_sha256"],
            "source_sha256": file_sha(__file__), "data_source_sha256": file_sha(kd.__file__),
            "torch_version": str(torch.__version__), "transformers_version": transformers.__version__, "palette": kd.PALETTE.tolist()}


def _require_identity(state, identity):
    if state.get("identity") != identity:
        raise ValueError("Existing run has different settings/code/data. Use a new RUN_NAME.")


def train(cache, report, run_dir, c, num_workers=2, stop_after_epoch=None, patience=6, min_epochs=20, log_every=100, resume=True,
          initial_checkpoint=None, fusion_context=None, hard_coordinates=None):
    c.validate()
    if initial_checkpoint is None:
        raise ValueError("Every experiment must start from the same existing DINOv3 checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("Colab 런타임을 GPU로 변경하세요. CPU 학습은 시작하지 않습니다.")
    if type(num_workers) is not int or not 0 <= num_workers <= 4:
        raise ValueError("num_workers must be 0..4")
    if stop_after_epoch is not None and (type(stop_after_epoch) is not int or not 1 <= stop_after_epoch <= c.epochs):
        raise ValueError("stop_after_epoch must be None or in 1..epochs")
    if type(patience) is not int or patience < 0 or log_every < 1:
        raise ValueError("Invalid patience/log_every")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(c.seed)
    np.random.seed(c.seed)
    torch.manual_seed(c.seed)
    torch.cuda.manual_seed_all(c.seed)
    torch.set_num_threads(2)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    local_run = Path("/content/komap_efficient_work") / run_dir.name
    local_run.mkdir(parents=True, exist_ok=True)
    identity = _identity(c, report)
    if initial_checkpoint is not None:
        identity["initial_checkpoint_sha256"] = file_sha(initial_checkpoint)
    if hard_coordinates is not None:
        identity["hard_mining_identity"] = hard_coordinates["identity"]
        identity["hard_coordinates_sha256"] = hard_coordinates["npz_sha256"]
    if fusion_context is not None:
        identity["fusion_context"] = fusion_context["identity"]
    identity_path, last_path, best_path = run_dir / "identity.json", run_dir / "last.pt", run_dir / "best.pt"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("RUN_NAME already has different settings/code/data. Use a new name.")
    if not resume and (last_path.exists() or best_path.exists()):
        raise ValueError("This run has checkpoints. Use a new RUN_NAME for fresh training.")
    if best_path.exists() and not last_path.exists():
        raise ValueError("best.pt exists but last.pt does not. Export best or use a new RUN_NAME.")
    state = torch.load(last_path, map_location="cpu", weights_only=False) if resume and last_path.exists() else None
    if state:
        _require_identity(state, identity)
    atomic_json(identity_path, identity)
    atomic_json(run_dir / "data_manifest.json", report)
    initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=False) if state is None else None
    saved_config = state.get("backbone_config") if state else initial.get("backbone_config")
    model = build_model(c, pretrained=False, backbone_config=saved_config).to(device=device, memory_format=torch.channels_last)
    if state:
        model.load_state_dict(state["model"], strict=True)
    else:
        if initial.get("config", {}).get("architecture") != "dinov3" or initial.get("identity", {}).get("data_sha256") != report["data_sha256"]:
            raise ValueError("Warm start requires the matching DINOv3 data and architecture")
        model.load_state_dict(initial["model"], strict=True)
    del initial
    backbone_config = backbone_config_of(model)
    encoder = model.encoders if c.architecture == "specialist" else model.encoder
    decoder_parameters = list(model.decoders.parameters()) + list(model.head.parameters()) if c.architecture == "specialist" else list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": c.encoder_lr},
        {"params": decoder_parameters, "lr": c.decoder_lr}
    ], weight_decay=c.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(c.ema_decay), use_buffers=False) if c.use_ema else None
    epoch_start, best_score, bad_checks, history, global_step = 1, -1.0, 0, [], 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if ema is not None:
            ema.load_state_dict(state["ema"])
        epoch_start, best_score = state["epoch"] + 1, state["best_score"]
        bad_checks, history, global_step = state["bad_checks"], state["history"], state["global_step"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"이어 학습: epoch {epoch_start}, best={best_score:.6f}")
    end_epoch = min(c.epochs, stop_after_epoch or c.epochs)
    if epoch_start > end_epoch or (patience and bad_checks >= patience and epoch_start > min_epochs):
        if not best_path.exists():
            raise RuntimeError("No best checkpoint exists")
        print("이번 실행 범위의 학습이 완료되어 best를 사용합니다.")
        return best_path
    loader, sampler = make_loader(cache, c, num_workers, hard_coordinates)
    gpu_input = GPUInput(device, normalize=c.architecture != "specialist")
    gpu = torch.cuda.get_device_properties(0)
    print(f"실제 장치: {gpu.name}, {gpu.total_memory / 2**30:.1f} GiB, AMP={amp_dtype}")
    print(f"patch={c.patch}, batch={c.batch}, crops={len(loader.dataset)}/epoch, updates={len(loader)}/epoch, workers={num_workers}")
    print("best 기준: 원본 해상도 회전4 TTA의 4클래스 이미지 평균 mIoU(" + c.absent_policy + ")")
    print(f"오류 영역 선택 확률={c.hard_crop_fraction:.0%}, CE에 포함할 어려운 픽셀 비율={c.ohem_fraction:.0%}")
    if state is None:

        initial_start = time.perf_counter()
        summary, valid_rows = evaluate(model, cache["valid"], c, gpu_input, amp_dtype)
        best_score = summary["miou_" + c.absent_policy]
        if not math.isfinite(best_score):
            raise FloatingPointError("Invalid starting validation score")
        initial_row = {"epoch":0,"loss":None,"encoder_lr":0.0,"decoder_lr":0.0,
                       "train_seconds":0.0,"data_wait_seconds":0.0,
                       "valid_seconds":time.perf_counter()-initial_start,"save_seconds":0.0,
                       "peak_gpu_gib":torch.cuda.max_memory_allocated()/2**30,
                       "valid_miou_ignore":summary["miou_ignore"],"valid_miou_zero":summary["miou_zero"],"best":best_score}
        history.append(initial_row)
        save_checkpoint(best_path,{"identity":identity,"config":asdict(c),"epoch":0,
                        "score":best_score,"model":model.state_dict(),"backbone_config":backbone_config,
                        "validation":summary,"uses_ema":False},local_run)
        atomic_json(run_dir/"initial_validation_summary.json",summary)
        atomic_json(run_dir/"best_validation_summary.json",summary)
        write_csv(run_dir/"best_validation_per_image.csv",valid_rows)
        write_csv(run_dir/"history.csv",history)
        save_checkpoint(last_path,{"identity":identity,"epoch":0,"model":model.state_dict(),
            "backbone_config":backbone_config,"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
            "ema":ema.state_dict() if ema is not None else None,"best_score":best_score,"bad_checks":0,
            "history":history,"global_step":0,"torch_rng":torch.get_rng_state(),
            "cuda_rng":torch.cuda.get_rng_state_all()},local_run)
        print(f"시작 best 보존: epoch 0, 회전4 TTA mIoU={best_score:.8f}",flush=True)
    for epoch in range(epoch_start, end_epoch + 1):
        sampler.epoch = epoch
        model.train()
        frozen = c.architecture == "dinov3" or epoch <= c.freeze_encoder_epochs
        for param in encoder.parameters():
            param.requires_grad_(not frozen)
        if frozen:
            encoder.eval()
        if epoch <= c.warmup_epochs:
            factor = epoch / c.warmup_epochs
        else:
            position = (epoch - c.warmup_epochs) / (c.epochs - c.warmup_epochs)
            factor = 0.02 + 0.98 * (1 + math.cos(math.pi * position)) / 2
        optimizer.param_groups[0]["lr"] = 0.0 if frozen else c.encoder_lr * factor
        optimizer.param_groups[1]["lr"] = c.decoder_lr * factor
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        wait_seconds, batches = 0.0, 0
        total_loss = torch.zeros((), device=device)
        previous = time.perf_counter()
        for step, (images, labels) in enumerate(loader, 1):
            now = time.perf_counter()
            wait_seconds += now - previous
            x = gpu_input(images, c)
            y = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(x)
                loss = segmentation_loss(logits, y, c.dice_weight, c.target_class, c.ohem_fraction)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach()
            batches += 1
            global_step += 1
            if ema is not None and (global_step % c.ema_every == 0 or step == len(loader)):
                ema.update_parameters(model)
            if step % log_every == 0:
                shown = float((total_loss / batches).item())
                if not math.isfinite(shown):
                    raise FloatingPointError("Non-finite loss; no checkpoint will be written")
                print(f"  epoch {epoch}: {step}/{len(loader)}, mean loss={shown:.4f}", flush=True)
            previous = time.perf_counter()
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - start_time
        mean_loss = float((total_loss / max(1, batches)).item())
        if not math.isfinite(mean_loss):
            raise FloatingPointError("Non-finite loss; no checkpoint will be written")
        row = {"epoch": epoch, "loss": mean_loss, "encoder_lr": optimizer.param_groups[0]["lr"],
               "decoder_lr": optimizer.param_groups[1]["lr"], "train_seconds": train_seconds,
               "data_wait_seconds": wait_seconds, "valid_seconds": 0.0, "save_seconds": 0.0,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
               "valid_miou_ignore": None, "valid_miou_zero": None, "best": best_score}
        do_eval = epoch == 1 or epoch % c.eval_every == 0 or epoch == end_epoch
        stopped = False
        if do_eval:
            evaluated_model = ema.module if ema is not None else model
            evaluation_start = time.perf_counter()
            summary, valid_rows = evaluate(evaluated_model, cache["valid"], c, gpu_input, amp_dtype, fusion_context)
            row["valid_seconds"] = time.perf_counter() - evaluation_start
            row["valid_miou_ignore"], row["valid_miou_zero"] = summary["miou_ignore"], summary["miou_zero"]
            score = summary["miou_" + c.absent_policy]
            if not math.isfinite(score):
                raise FloatingPointError("Invalid validation score")
            save_start = time.perf_counter()
            if score > best_score:
                best_score, bad_checks = score, 0
                save_checkpoint(best_path, {"identity": identity, "config": asdict(c), "epoch": epoch,
                                           "score": score, "model": evaluated_model.state_dict(),
                                           "backbone_config": backbone_config,
                                           "validation": summary, "uses_ema": c.use_ema}, local_run)
                write_csv(run_dir / "best_validation_per_image.csv", valid_rows)
                atomic_json(run_dir / "best_validation_summary.json", summary)
            else:
                bad_checks += 1
            row["best"] = best_score
            stopped = bool(patience and bad_checks >= patience and epoch >= min_epochs)
            history.append(row)
            last_state = {"identity": identity, "epoch": epoch, "model": model.state_dict(),
                          "backbone_config": backbone_config,
                          "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                          "ema": ema.state_dict() if ema is not None else None,
                          "best_score": best_score, "bad_checks": bad_checks, "history": history,
                          "global_step": global_step, "torch_rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all()}
            save_checkpoint(last_path, last_state, local_run)
            row["save_seconds"] = time.perf_counter() - save_start
            atomic_json(run_dir / "training_status.json", {"epoch": epoch, "best": best_score,
                        "early_stopped": stopped, "bad_validation_checks": bad_checks})
            print(f"Epoch {epoch}/{c.epochs}: loss={mean_loss:.4f}; valid mIoU={score:.4f}; best={best_score:.4f}; "
                  f"train={train_seconds:.1f}s; valid={row['valid_seconds']:.1f}s; GPU={row['peak_gpu_gib']:.2f}GiB", flush=True)
        else:
            history.append(row)
            print(f"Epoch {epoch}/{c.epochs}: loss={mean_loss:.4f}; train={train_seconds:.1f}s; best={best_score:.4f}", flush=True)
        write_csv(run_dir / "history.csv", history)
        if stopped:
            print(f"조기 종료: 검증 {bad_checks}회 동안 best 상승 없음. best 확률 내보내기로 진행합니다.")
            break
    del loader, model, optimizer, ema, x, y, logits, loss
    gc.collect()
    torch.cuda.empty_cache()
    return best_path


def export_best(cache, report, run_dir, tta_rotations=4, compare_plain=True, fusion_context=None):
    'Export native float32 probabilities and PNGs from RAM, using own best.'
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for probability export")
    run_dir = Path(run_dir)
    best_path = run_dir / "best.pt"
    state = torch.load(best_path, map_location="cpu", weights_only=False)
    c = Config(**state["config"])
    c.validate()
    core = _identity(c, report)
    if any(state["identity"].get(k) != v for k,v in core.items()):
        raise ValueError("Export code/settings/data differ from the checkpoint")
    if c.target_class >= 0:
        if fusion_context is None or state["identity"].get("fusion_context") != fusion_context["identity"]:
            raise ValueError("Use the same fixed baseline context for specialist validation")
    if tta_rotations not in (1,4):
        raise ValueError("tta_rotations must be 1 or 4")
    checkpoint_sha = file_sha(best_path)
    requested_modes = [1,4] if compare_plain and tta_rotations==4 else [tta_rotations]
    export_dir = run_dir / "exports" / f"best_e{state['epoch']:03d}_{checkpoint_sha[:12]}_tta{tta_rotations}_cmp{int(compare_plain)}"
    manifest_path = export_dir / "manifest.json"
    if manifest_path.exists():
        from komap_phase_fusion import load_export, read_probability
        m=json.loads(manifest_path.read_text(encoding="utf-8"))
        if m.get("checkpoint_sha256") != checkpoint_sha or m.get("complete") is not True:
            raise ValueError("Existing export is inconsistent")
        atomic_json(run_dir/"export_index.json",{"manifest":manifest_path.relative_to(run_dir).as_posix()})
        e=load_export(run_dir)
        for split in ("valid","test"):
            for rec in cache[split]: read_probability(e,split,rec)
        print("완료된 기존 내보내기를 사용합니다:",export_dir)
        return export_dir
    model = build_model(c, pretrained=False, backbone_config=state.get("backbone_config")).to("cuda", memory_format=torch.channels_last)
    model.load_state_dict(state["model"],strict=True)
    model.eval()
    gpu_input = GPUInput(torch.device("cuda"),normalize=c.architecture!="specialist")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    candidate_probability, candidate_rows, candidate_cm = {}, {}, {}
    for mode in requested_modes:
        probabilities, rows, cm = {}, [], np.zeros((4,4),dtype=np.int64)
        for rec in tqdm(cache["valid"],desc=f"best valid TTA={mode}"):
            p=predict_probability(model,rec["image"],c,gpu_input,amp_dtype,mode)
            probabilities[rec["name"]]=p
            score=p
            if c.target_class>=0:
                score=fusion_context["probabilities"][rec["name"]].copy()
                score[c.target_class]=p[0]
                score*=np.asarray(fusion_context["weights"],dtype=np.float32)[:,None,None]
            row,one=evaluate_labels(rec["name"],rec["mask"],score.argmax(axis=0).astype(np.uint8))
            rows.append(row)
            cm+=one
        candidate_probability[mode],candidate_rows[mode],candidate_cm[mode]=probabilities,rows,cm
    summaries={mode:summarize(candidate_rows[mode],candidate_cm[mode]) for mode in requested_modes}
    chosen=max(requested_modes,key=lambda mode:summaries[mode]["miou_"+c.absent_policy])
    selected=candidate_probability.pop(chosen)
    candidate_probability.clear()
    export_dir.mkdir(parents=True,exist_ok=True)
    files, test_masks = [], []
    try:
        for split in ("valid","test"):
            for rec in tqdm(cache[split],desc=split+" 확률 저장"):
                p=selected[rec["name"]] if split=="valid" else predict_probability(model,rec["image"],c,gpu_input,amp_dtype,chosen)
                destination=export_dir/split/(Path(rec["name"]).stem+".npz")
                destination.parent.mkdir(parents=True,exist_ok=True)
                tmp=destination.with_suffix(".npz.tmp")
                raw=p[0] if c.target_class>=0 else p
                with tmp.open("wb") as f:
                    if c.target_class>=0:
                        np.savez_compressed(f,probability=raw.astype(np.float32),target_class=np.int16(c.target_class))
                    else:
                        np.savez_compressed(f,probabilities=raw.astype(np.float32),class_ids=np.arange(4,dtype=np.int16))
                os.replace(tmp,destination)
                files.append({"split":split,"image":rec["name"],"image_sha256":rec["image_sha256"],
                              "mask_sha256":rec["mask_sha256"],"shape":list(raw.shape),
                              "probability_file":destination.relative_to(export_dir).as_posix(),"probability_sha256":file_sha(destination)})
                if c.target_class<0:
                    labels=p.argmax(axis=0).astype(np.uint8)
                    mask_path=export_dir/("valid_masks" if split=="valid" else "candidate_test_masks")/rec["name"].replace("_image.png","_mask.png")
                    kd.save_rgb_mask(labels,mask_path)
                    if split=="test":test_masks.append(mask_path)
        if test_masks:
            with zipfile.ZipFile(export_dir/"candidate_test_masks.zip","w",zipfile.ZIP_DEFLATED) as z:
                for path in test_masks:z.write(path,arcname=path.name)
        write_csv(export_dir/"validation_per_image.csv",candidate_rows[chosen])
        summary={"selected_tta_rotations":chosen,"plain_vs_tta":{str(k):v for k,v in summaries.items()},
                 "selected":summaries[chosen],"selection_note":"TTA selected on valid20; not independent test performance"}
        atomic_json(export_dir/"validation_summary.json",summary)
        manifest={"complete":True,"format_version":2,"model_family":MODEL_FAMILY,"target_class":c.target_class,
                  "class_names":list(kd.CLASS_NAMES),"palette":kd.PALETTE.tolist(),"config":asdict(c),
                  "checkpoint_epoch":state["epoch"],"checkpoint_sha256":checkpoint_sha,"identity":state["identity"],
                  "dtype":"float32","tta_rotations":chosen,"probability":"Gaussian weighted overlapping probabilities",
                  "calibration":"none","files":files}
        atomic_json(manifest_path,manifest)
        atomic_json(run_dir/"export_index.json",{"manifest":manifest_path.relative_to(run_dir).as_posix()})
    finally:
        del model, selected, state
        gc.collect()
        torch.cuda.empty_cache()
    print("선택 TTA:",chosen,"최종 valid mIoU:",summaries[chosen]["miou_"+c.absent_policy])
    print("Drive 확률·결과:",export_dir)
    return export_dir


def prepare_source_checkpoint(source_checkpoint, report, expected_sha_prefix="1bec54df97dd", expected_epoch=36):

    source_checkpoint = Path(source_checkpoint)
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"기존 64패치 best.pt가 없습니다: {source_checkpoint}")
    if not expected_sha_prefix or len(expected_sha_prefix) < 12:
        raise ValueError("기준 체크포인트의 SHA 접두사를 12자리 이상 지정하세요.")
    root = Path('/content/komap_hard_ablation_source')
    root.mkdir(parents=True, exist_ok=True)
    destination = root / (expected_sha_prefix + '.pt')
    if not destination.is_file() or not file_sha(destination).startswith(expected_sha_prefix):
        temporary = destination.with_suffix('.pt.tmp')
        shutil.copyfile(source_checkpoint, temporary)
        actual = file_sha(temporary)
        if not actual.startswith(expected_sha_prefix):
            temporary.unlink(missing_ok=True)
            raise ValueError("기준 best.pt가 바뀌었습니다. best_e036_1bec54df97dd에 해당하는 가중치를 SOURCE_CHECKPOINT로 지정하세요. 확률 NPZ만으로는 학습을 재개할 수 없습니다.")
        os.replace(temporary, destination)
    actual = file_sha(destination)
    state = torch.load(destination, map_location='cpu', weights_only=False)
    source_config = state.get('config', {})
    identity = state.get('identity', {})
    if identity.get('family') != MODEL_FAMILY or identity.get('palette') != kd.PALETTE.tolist():
        raise ValueError('기준 체크포인트의 모델 종류 또는 팔레트가 다릅니다.')
    if identity.get('data_sha256') != report['data_sha256']:
        raise ValueError('기준 모델과 현재 train/valid/test 이미지·정답 바이트가 다릅니다.')
    if source_config.get('architecture') != 'dinov3' or source_config.get('target_class') != -1:
        raise ValueError('기준 모델은 DINOv3 4클래스 모델이어야 합니다.')
    if source_config.get('patch') != 480 or source_config.get('patches_per_image') != 64:
        raise ValueError('이 비교 실험의 기준은 patch=480, patches_per_image=64 모델입니다.')
    if source_config.get('pretrained_model_id') != 'facebook/dinov3-vits16-pretrain-lvd1689m':
        raise ValueError('기준 DINOv3 모델 ID가 다릅니다.')
    if state.get('epoch') != expected_epoch or not state.get('backbone_config'):
        raise ValueError('기준 best 에포크 또는 백본 설정이 다릅니다.')
    result = {'path':str(destination),'sha256':actual,'source_epoch':state['epoch'],
              'source_config':source_config,'data_sha256':report['data_sha256']}
    del state
    gc.collect()
    print(f"공통 시작 모델: 원래 epoch {expected_epoch}, SHA={actual[:12]}; 새 실험은 0/20부터 시작합니다.")
    return result


def prepare_hard_coordinates(cache, report, source_info, c, run_dir):

    if c.hard_crop_fraction == 0:
        return None
    identity = {'source_checkpoint_sha256':source_info['sha256'],
                'data_sha256':report['data_sha256'],'source_sha256':file_sha(__file__),
                'patch':c.patch,'stride':c.eval_stride,'batch':c.batch,
                'mining_tta_rotations':1,'seed':c.seed,'maximum_positions_per_pair':20000,
                'pairs':[[1,2],[1,3]],'images':[r['name'] for r in cache['train']]}
    folder = Path(run_dir)/'hard_train_cache'
    folder.mkdir(parents=True,exist_ok=True)
    metadata_path, array_path = folder/'manifest.json', folder/'coordinates.npz'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('identity') != identity or metadata.get('complete') is not True:
            raise ValueError('오류 좌표 캐시의 설정이 다릅니다. 새 RUN_NAME을 사용하세요.')
        if not array_path.is_file() or file_sha(array_path) != metadata.get('npz_sha256'):
            raise ValueError('오류 좌표 캐시가 손상되었습니다.')
        with np.load(array_path,allow_pickle=False) as z:
            positions = [{pair:z[f'r{i}_{pair}'].copy() for pair in ('NiEu','NiAl')}
                         for i in range(len(cache['train']))]
        print('저장된 train 오류 좌표를 RAM에 불러왔습니다.')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('train 오류 위치를 찾으려면 Colab GPU가 필요합니다.')
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        initial = torch.load(source_info['path'],map_location='cpu',weights_only=False)
        model = build_model(c,pretrained=False,backbone_config=initial['backbone_config'])
        model.load_state_dict(initial['model'],strict=True)
        del initial
        model = model.to('cuda',memory_format=torch.channels_last)
        gpu_input = GPUInput(torch.device('cuda'),normalize=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        positions, statistics, arrays = [], [], {}
        try:
            for i,rec in enumerate(tqdm(cache['train'],desc='train70 오류 위치 → RAM (추론 1회전)')):
                probability = predict_probability(model,rec['image'],c,gpu_input,dtype,1)
                predicted, truth = probability.argmax(axis=0).astype(np.uint8), rec['mask']
                entry, row = {}, {'image':rec['name']}
                rng = np.random.default_rng(np.random.SeedSequence([c.seed,i,2026]))
                for pair,a,b in [('NiEu',1,2),('NiAl',1,3)]:
                    error = ((truth==a)&(predicted==b))|((truth==b)&(predicted==a))
                    indices = np.flatnonzero(error.ravel())
                    row[pair+'_error_pixels'] = int(indices.size)
                    if indices.size > 20000:
                        indices = rng.choice(indices,20000,replace=False)
                    entry[pair] = indices.astype(np.int32)
                    arrays[f'r{i}_{pair}'] = entry[pair]
                positions.append(entry);statistics.append(row)
                del probability,predicted
            temporary = array_path.with_suffix('.npz.tmp')
            with temporary.open('wb') as f:
                np.savez_compressed(f,**arrays)
            os.replace(temporary,array_path)
            metadata = {'complete':True,'identity':identity,'npz_sha256':file_sha(array_path),
                        'per_image':statistics,
                        'fallback_note':'해당 사진에 유효한 오류 좌표가 없으면 기존 패치 선택 방식으로 대체합니다.'}
            atomic_json(metadata_path,metadata)
            write_csv(folder/'train_error_counts.csv',statistics)
        finally:
            del model
            gc.collect();torch.cuda.empty_cache()
    for rec,entry in zip(cache['train'],positions):
        for values in entry.values():
            if values.ndim!=1 or not np.issubdtype(values.dtype,np.integer):
                raise ValueError('잘못된 오류 좌표 형식')
            if values.size and (values.min()<0 or values.max()>=rec['mask'].size):
                raise ValueError('오류 좌표가 원본 이미지 범위를 벗어났습니다.')
            values.setflags(write=False)
    usable = sum(any(v.size for v in entry.values()) for entry in positions)
    print(f'오류 좌표가 있는 train 사진: {usable}/{len(positions)}; 없으면 기존 선택 방식으로 대체합니다.')
    return {'positions':positions,'identity':identity,'npz_sha256':metadata['npz_sha256']}
