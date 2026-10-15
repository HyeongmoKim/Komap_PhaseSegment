'클래스 중심 crop, Eu 크기 균형 및 중첩 확률 추론.'
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
    decoder_lr: float = 5e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    freeze_encoder_epochs: int = 0
    eval_every: int = 1
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
    ohem_mix_weight: float = 0.0
    difficulty_ramp_epochs: int = 5
    select_on_fixed_blend: bool = True
    dino_trainable_blocks: int = 0
    hrnet_aux_weight: float = 0.4
    wide_context: bool = False
    context_adapter_lr: float = 3e-4
    copy_paste_probability: float = 0.0
    paste_ramp_epochs: int = 5
    paste_background_tolerance: float = 20.0
    zoom_probability: float = 0.0
    zoom_crop: int = 240
    eu_component_balanced: bool = False

    def validate(self):
        if self.architecture != 'dinov3' or self.wide_context or self.dino_trainable_blocks:
            raise ValueError('Zoom experiments use the unchanged frozen DINO detail decoder')
        if self.copy_paste_probability or self.hard_crop_fraction or self.ohem_mix_weight:
            raise ValueError('Zoom experiments keep the original sampler/loss without other repair methods')
        if not 0 <= self.zoom_probability <= 1:
            raise ValueError('zoom_probability must be 0..1')
        if type(self.zoom_crop) is not int or not 32 <= self.zoom_crop <= self.patch:
            raise ValueError('zoom_crop must be an integer in 32..patch')
        if type(self.eu_component_balanced) is not bool:
            raise ValueError('eu_component_balanced must be bool')
        if self.architecture not in ("dinov3","hrnet_ocr"):
            raise ValueError("Unsupported architecture")
        if (self.architecture == "specialist" and self.target_class not in (1, 2)) or (self.architecture != "specialist" and self.target_class != -1):
            raise ValueError("Only specialist models use target_class=1 or 2; others use -1")
        for name in ("patch", "batch", "patches_per_image", "epochs", "eval_every", "eval_stride", "ema_every", "warmup_epochs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.hard_crop_fraction <= 1 or not 0 < self.ohem_fraction <= 1:
            raise ValueError("Invalid hard_crop_fraction or ohem_fraction")
        if not 0 <= self.ohem_mix_weight <= 1:
            raise ValueError('ohem_mix_weight must be 0..1')
        if type(self.difficulty_ramp_epochs) is not int or self.difficulty_ramp_epochs < 1:
            raise ValueError('difficulty_ramp_epochs must be a positive integer')
        if self.ohem_mix_weight and self.ohem_fraction >= 1:
            raise ValueError('An active hard loss needs ohem_fraction < 1')
        if type(self.dino_trainable_blocks) is not int or self.dino_trainable_blocks not in (0,2):
            raise ValueError('DINO trainable blocks must be 0 or 2')
        if self.architecture == 'hrnet_ocr' and (self.dino_trainable_blocks or self.encoder != 'hrnet_w32.ms_in1k'):
            raise ValueError('HRNet experiment requires the fixed W32 encoder')
        if type(self.wide_context) is not bool:
            raise ValueError('wide_context must be bool')
        if self.wide_context and (self.architecture != 'dinov3' or self.dino_trainable_blocks):
            raise ValueError('Context experiment uses frozen DINO')
        if self.wide_context and self.copy_paste_probability:
            raise ValueError('Context and Copy-Paste are separate experiments')
        if not 0 <= self.copy_paste_probability <= .2 or self.context_adapter_lr <= 0:
            raise ValueError('Invalid paste probability or adapter LR')
        if type(self.paste_ramp_epochs) is not int or self.paste_ramp_epochs < 1:
            raise ValueError('Paste ramp must be a positive integer')
        if not 0 < self.paste_background_tolerance <= 255:
            raise ValueError('Invalid background tolerance')
        if not 0 <= self.hrnet_aux_weight <= 1:
            raise ValueError('Auxiliary loss weight must be 0..1')
        if self.select_on_fixed_blend is not True:
            raise ValueError('This package selects checkpoints by the fixed final blend')
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
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fields)
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
        self.copy_paste_pool,self.copy_paste_report=None,{}
        if config.copy_paste_probability:
            self.copy_paste_pool,self.copy_paste_report=build_paste_pool(records,config)
        self.epoch_size = len(records) * config.patches_per_image
        self.zoom_plan_epoch, self.zoom_plans = None, {}
        self.eu_component_pools = ([build_eu_component_pool(rec['mask']) for rec in records]
                                   if config.eu_component_balanced else None)
        pools = self.eu_component_pools or []
        self.zoom_sampling_report = eu_pool_summary(records if pools else [], pools, config)
        self.zoom_sampling_report['training_images'] = len(records)
        self.zoom_sampling_report['total_crops_per_epoch'] = self.epoch_size
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
        if self.zoom_plan_epoch != epoch:
            self.zoom_plan_epoch, self.zoom_plans = epoch, {}
        if rec_id not in self.zoom_plans:
            self.zoom_plans[rec_id] = zoom_plan_for_image(c, epoch, rec_id)
        is_zoom = bool(self.zoom_plans[rec_id][index // len(self.records)])
        nominal_crop = c.zoom_crop if is_zoom else c.patch
        crop = max(32, int(round(nominal_crop / scale)))
        branch_rng = np.random.default_rng(np.random.SeedSequence([c.seed, epoch, index, 777]))
        groups = [v for v in self.hard_coordinates[rec_id].values() if v.size] if self.hard_coordinates is not None else []
        ramp = min(1.0, epoch / c.difficulty_ramp_epochs)
        use_hard = c.hard_crop_fraction > 0 and bool(groups) and branch_rng.random() < c.hard_crop_fraction * ramp
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
                if target_class == 2 and self.eu_component_pools is not None:

                    component_rng = np.random.default_rng(np.random.SeedSequence([c.seed, epoch, index, 705]))
                    selected = draw_eu_component_pixel(self.eu_component_pools[rec_id], component_rng)
                    if selected is not None:
                        pos = selected
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
        pasted_class=0
        if self.copy_paste_pool is not None:
            paste_rng=np.random.default_rng(np.random.SeedSequence([c.seed,epoch,index,552]))
            ramp=min(1.0,epoch/c.paste_ramp_epochs)
            if paste_rng.random() < c.copy_paste_probability*ramp:
                rgb,label,pasted_class=paste_one_component(rgb,label,self.copy_paste_pool,
                    paste_rng,rec_id,c)
        if c.wide_context:
            wide=reflected_square(image,y-crop//2,x-crop//2,crop*2)
            wide=cv2.resize(wide,(c.patch,c.patch),interpolation=cv2.INTER_AREA)
            rgb=np.concatenate([rgb,wide],axis=2)

        k = int(rng.integers(4))
        rgb, label = np.rot90(rgb, k), np.rot90(label, k)
        if rng.random() < 0.5:
            rgb, label = np.fliplr(rgb), np.fliplr(label)
        return (torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))),
                torch.from_numpy(np.ascontiguousarray(label)),torch.tensor(pasted_class,dtype=torch.int8))


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
        if x.shape[1] not in (3,6):raise ValueError('RGB or paired RGB required')
        repeats=x.shape[1]//3
        x=(x-self.mean.repeat(1,repeats,1,1))/self.std.repeat(1,repeats,1,1) if self.normalize else x
        return x.contiguous(memory_format=torch.channels_last)


def build_model(c, pretrained, backbone_config=None):
    if c.architecture != 'dinov3' or c.wide_context or c.dino_trainable_blocks:
        raise ValueError('Only the frozen DINOv3 detail decoder is supported')
    return DinoFineDecoder(c.pretrained_model_id, pretrained, backbone_config, 0)


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
    def __init__(self, model_id, pretrained, backbone_config=None, trainable_blocks=0):
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
        self.trainable_blocks = trainable_blocks
        self.encoder.requires_grad_(False)
        if trainable_blocks:
            layers = getattr(self.encoder,'layer',None)
            if layers is None or len(layers)!=12:
                raise ValueError('Pinned DINOv3 encoder.layer must contain 12 blocks')
            for layer in layers[-trainable_blocks:]:layer.requires_grad_(True)
        self.encoder.eval()
        self.decoder = DinoDetailPath(self.encoder.config.hidden_size)
        self.segmentation_head = nn.Conv2d(16, 4, 1)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):

        enabled = torch.is_grad_enabled() and self.training and self.trainable_blocks > 0
        with torch.set_grad_enabled(enabled):
            output = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        b, _, h, w = x.shape
        prefix = 1 + self.encoder.config.num_register_tokens
        features = [output.hidden_states[i][:, prefix:].transpose(1, 2).reshape(b, 384, h // 16, w // 16)
                    for i in (3, 6, 9, 12)]
        return self.segmentation_head(self.decoder(x, features))


def segmentation_loss(logits, target, dice_weight, target_class=-1,
                      ohem_fraction=1.0, ohem_mix_weight=0.0):
    if target_class != -1:
        raise ValueError('Only four-class segmentation is supported')
    logits = logits.float()
    valid_flat = (target != 255).flatten(1)
    pixel_loss = F.cross_entropy(logits, target, ignore_index=255, reduction='none')

    full_ce = pixel_loss.sum() / valid_flat.sum().clamp_min(1)
    ce = full_ce
    if ohem_mix_weight > 0:
        flat = pixel_loss.flatten(1).masked_fill(~valid_flat, -torch.inf)
        maximum_kept = max(1, math.ceil(flat.shape[1] * ohem_fraction))
        hard = torch.topk(flat, maximum_kept, dim=1, largest=True, sorted=True).values
        kept = torch.ceil(valid_flat.sum(dim=1).float() * ohem_fraction).long().clamp_min(1)
        selected = ((torch.arange(maximum_kept, device=logits.device)[None] < kept[:, None])
                    & torch.isfinite(hard))
        hard_ce = (torch.where(selected, hard, 0.0).sum(dim=1)
                   / selected.sum(dim=1).clamp_min(1)).mean()
        ce = (1-ohem_mix_weight)*full_ce + ohem_mix_weight*hard_ce
    valid = (target != 255).unsqueeze(1)
    one_hot = F.one_hot(target.clamp(max=3), 4).permute(0, 3, 1, 2).float()*valid
    probability = logits.softmax(dim=1)*valid
    intersection = (probability*one_hot).sum(dim=(2, 3))
    target_count = one_hot.sum(dim=(2, 3))
    denominator = probability.sum(dim=(2, 3)) + target_count
    loss = 1-(2*intersection+1e-6)/(denominator+1e-6)
    present = target_count > 0
    per_image = (loss*present).sum(dim=1)/present.sum(dim=1).clamp_min(1)
    return ce + dice_weight*per_image.mean()


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
        views=[]
        for y,x in chunk:
            local=padded[y:y+p,x:x+p]
            if c.wide_context:
                wide=reflected_square(image,y-p//2,x-p//2,p*2)
                wide=cv2.resize(wide,(p,p),interpolation=cv2.INTER_AREA)
                local=np.concatenate([local,wide],axis=2)
            views.append(local.transpose(2,0,1))
        array=np.stack(views)
        batch = gpu_input(torch.from_numpy(np.ascontiguousarray(array)))
        combined = None
        for k in range(tta_rotations):
            rotated = torch.rot90(batch, k, dims=(-2, -1)).contiguous(memory_format=getattr(gpu_input,"memory_format",torch.channels_last))
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
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
    if fusion_context is None or not c.select_on_fixed_blend:
        raise ValueError('The fixed Eu-replaced specialist/DINO blend is required')
    rows, solo_rows = [], []
    cm, solo_cm = np.zeros((4,4), dtype=np.int64), np.zeros((4,4), dtype=np.int64)
    alpha = fusion_context['identity']['alpha']
    for rec in records:
        probability = predict_probability(model, rec['image'], c, gpu_input, amp_dtype,
                                          c.validation_tta_rotations)
        specialist = fusion_context['probabilities'][rec['name']]
        if specialist.shape != probability.shape:
            raise ValueError('Fixed specialist/native DINO sizes differ')
        combined = (1-alpha)*specialist + alpha*probability
        row, one = evaluate_labels(rec['name'], rec['mask'], combined.argmax(axis=0).astype(np.uint8))
        solo, one_solo = evaluate_labels(rec['name'], rec['mask'], probability.argmax(axis=0).astype(np.uint8))

        row['standalone_miou_ignore'] = solo['miou_ignore']
        row['standalone_miou_zero'] = solo['miou_zero']
        rows.append(row); solo_rows.append(solo)
        cm += one; solo_cm += one_solo
    summary = summarize(rows, cm)
    summary['standalone'] = summarize(solo_rows, solo_cm)
    summary['checkpoint_selection'] = 'fixed Eu-replaced specialist/DINO blend, alpha=0.5'
    return summary, rows


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
    if fusion_context is None:
        raise ValueError('Prepare fixed blend context before training')
    if c.architecture == 'dinov3' and initial_checkpoint is None:
        raise ValueError('DINO fine-tuning requires the previous GPU1 best')
    if c.architecture == 'hrnet_ocr' and initial_checkpoint is not None:
        raise ValueError('HRNet uses an ImageNet encoder and fresh OCR head, not DINO weights')
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
    initial = torch.load(initial_checkpoint,map_location='cpu',weights_only=False) if state is None and initial_checkpoint is not None else None
    saved_config = state.get('backbone_config') if state else initial.get('backbone_config') if initial else None
    download_pretrained = state is None and c.architecture == 'hrnet_ocr'
    model = build_model(c,pretrained=download_pretrained,backbone_config=saved_config).to(device=device,memory_format=torch.channels_last)
    if state:
        model.load_state_dict(state["model"], strict=True)
    elif initial is not None:
        if initial.get("config", {}).get("architecture") != "dinov3" or initial.get("identity", {}).get("data_sha256") != report["data_sha256"]:
            raise ValueError("Warm start requires the matching DINOv3 data and architecture")
        load_start_weights(model,initial["model"],c)
    del initial
    backbone_config = backbone_config_of(model)
    encoder = model.encoder
    configure_encoder_training(model,c,1)
    print('학습 가능한 encoder 파라미터:',sum(p.numel() for p in encoder.parameters() if p.requires_grad))
    decoder_parameters = list(model.decoders.parameters()) + list(model.head.parameters()) if c.architecture == "specialist" else list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
    context_parameters=list(model.decoder.wide_context_adapter.parameters()) if c.wide_context else []
    adapter_ids={id(p) for p in context_parameters}
    decoder_parameters=[p for p in decoder_parameters if id(p) not in adapter_ids]
    groups=[{'params':encoder.parameters(),'lr':c.encoder_lr},
            {'params':decoder_parameters,'lr':c.decoder_lr}]
    if context_parameters:groups.append({'params':context_parameters,'lr':c.context_adapter_lr})
    optimizer=torch.optim.AdamW(groups,weight_decay=c.weight_decay)
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
        del model, optimizer, ema, state
        gc.collect()
        torch.cuda.empty_cache()
        return best_path
    loader, sampler = make_loader(cache, c, num_workers, hard_coordinates)
    atomic_json(run_dir/'zoom_sampling_summary.json', loader.dataset.zoom_sampling_report)
    nzoom = int(math.floor(c.patches_per_image*c.zoom_probability+0.5))
    print(f"사진당 nominal 480 crop={c.patches_per_image-nzoom}, nominal {c.zoom_crop} crop={nzoom}; 출력={c.patch}; Eu 크기별 균형={c.eu_component_balanced}", flush=True)
    if c.copy_paste_probability:
        atomic_json(run_dir/'copy_paste_pool_summary.json',loader.dataset.copy_paste_report)
    gpu_input = GPUInput(device, normalize=c.architecture != "specialist")
    gpu = torch.cuda.get_device_properties(0)
    print(f"실제 장치: {gpu.name}, {gpu.total_memory / 2**30:.1f} GiB, AMP={amp_dtype}")
    print(f"patch={c.patch}, batch={c.batch}, crops={len(loader.dataset)}/epoch, updates={len(loader)}/epoch, workers={num_workers}")
    print("best 기준: Eu 교체 specialist + 현재 모델 50:50 결합 / 회전4 TTA / 이미지 평균 mIoU(" + c.absent_policy + ")")
    print(f"오류 패치 최대={c.hard_crop_fraction:.0%}, 어려운 상위={c.ohem_fraction:.0%}, hard CE 최대 가중치={c.ohem_mix_weight:.0%}, {c.difficulty_ramp_epochs}에포크 ramp")
    if state is None:

        initial_start = time.perf_counter()
        summary, valid_rows = evaluate(model, cache["valid"], c, gpu_input, amp_dtype, fusion_context)
        best_score = summary["miou_" + c.absent_policy]
        if not math.isfinite(best_score):
            raise FloatingPointError("Invalid starting validation score")
        initial_row = {"epoch":0,"loss":None,"encoder_lr":0.0,"decoder_lr":0.0,
                       "train_seconds":0.0,"data_wait_seconds":0.0,
                       "valid_seconds":time.perf_counter()-initial_start,"save_seconds":0.0,
                       "peak_gpu_gib":torch.cuda.max_memory_allocated()/2**30,
                       "valid_miou_ignore":summary["miou_ignore"],"valid_miou_zero":summary["miou_zero"],"best":best_score,
                       "standalone_miou_ignore":summary['standalone']['miou_ignore'],
                       "standalone_miou_zero":summary['standalone']['miou_zero'],
                       'context_lr':0.0,'paste_Ni':0,'paste_Eu':0,'paste_actual_fraction':0.0}
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
        print(f"시작 best 보존: epoch 0, 결합 mIoU={best_score:.8f}, 단독={summary['standalone']['miou_ignore']:.8f}",flush=True)
    for epoch in range(epoch_start, end_epoch + 1):
        sampler.epoch = epoch
        model.train()
        frozen = configure_encoder_training(model,c,epoch)
        if epoch <= c.warmup_epochs:
            factor = epoch / c.warmup_epochs
        else:
            position = (epoch - c.warmup_epochs) / (c.epochs - c.warmup_epochs)
            factor = 0.02 + 0.98 * (1 + math.cos(math.pi * position)) / 2
        optimizer.param_groups[0]["lr"] = 0.0 if frozen else c.encoder_lr * factor
        optimizer.param_groups[1]["lr"] = c.decoder_lr * factor
        if c.wide_context:optimizer.param_groups[2]['lr']=c.context_adapter_lr*factor
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        wait_seconds, batches = 0.0, 0
        paste_ni,paste_eu,patch_count=0,0,0
        total_loss = torch.zeros((), device=device)
        previous = time.perf_counter()
        for step, (images, labels, paste_class) in enumerate(loader, 1):
            paste_ni+=int((paste_class==1).sum());paste_eu+=int((paste_class==2).sum())
            patch_count+=int(images.shape[0])
            now = time.perf_counter()
            wait_seconds += now - previous
            x = gpu_input(images, c)
            y = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(x)
                ramp = min(1.0, epoch / c.difficulty_ramp_epochs)
                auxiliary = None
                if isinstance(logits,tuple):logits,auxiliary = logits
                loss = segmentation_loss(logits,y,c.dice_weight,c.target_class,
                                         c.ohem_fraction,c.ohem_mix_weight*ramp)
                if auxiliary is not None:
                    loss = loss + c.hrnet_aux_weight*segmentation_loss(auxiliary,y,0.0)
            scaler.scale(loss).backward()
            if epoch == 1 and step == 1 and c.architecture == 'dinov3' and c.dino_trainable_blocks:
                parameters = [p for layer in model.encoder.layer[-c.dino_trainable_blocks:] for p in layer.parameters()]
                if any(p.grad is None for p in parameters):
                    raise RuntimeError('Last DINO blocks did not receive gradients; training stopped')
                print('실제 첫 학습 배치: 마지막 2개 블록 gradient 연결 확인 완료',flush=True)
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
               "valid_miou_ignore": None, "valid_miou_zero": None, "best": best_score,
               "standalone_miou_ignore":None,"standalone_miou_zero":None,
               'context_lr':optimizer.param_groups[2]['lr'] if c.wide_context else 0.0,
               'paste_Ni':paste_ni,'paste_Eu':paste_eu,
               'paste_actual_fraction':(paste_ni+paste_eu)/max(1,patch_count)}
        do_eval = epoch == 1 or epoch % c.eval_every == 0 or epoch == end_epoch
        stopped = False
        if do_eval:
            evaluated_model = ema.module if ema is not None else model
            evaluation_start = time.perf_counter()
            summary, valid_rows = evaluate(evaluated_model, cache["valid"], c, gpu_input, amp_dtype, fusion_context)
            row["valid_seconds"] = time.perf_counter() - evaluation_start
            row['standalone_miou_ignore'] = summary['standalone']['miou_ignore']
            row['standalone_miou_zero'] = summary['standalone']['miou_zero']
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
            print(f"Epoch {epoch}/{c.epochs}: loss={mean_loss:.4f}; blend mIoU={score:.6f}; solo={summary['standalone']['miou_ignore']:.6f}; best={best_score:.6f}; "
                  f"train={train_seconds:.1f}s; valid={row['valid_seconds']:.1f}s; GPU={row['peak_gpu_gib']:.2f}GiB", flush=True)
        else:
            history.append(row)
            print(f"Epoch {epoch}/{c.epochs}: loss={mean_loss:.4f}; train={train_seconds:.1f}s; best={best_score:.4f}", flush=True)
        if c.copy_paste_probability:
            print(f'실제 적용 Copy-Paste: Ni={paste_ni}, Eu={paste_eu}, fraction={(paste_ni+paste_eu)/max(1,patch_count):.2%}',flush=True)
        write_csv(run_dir / "history.csv", history)
        if stopped:
            print(f"조기 종료: 검증 {bad_checks}회 동안 best 상승 없음. best 확률 내보내기로 진행합니다.")
            break

    evaluated_model = None
    del loader, model, optimizer, ema, x, y, logits, loss, state
    gc.collect()
    torch.cuda.empty_cache()
    return best_path


def prepare_source_checkpoint(source_checkpoint, report, expected_sha_prefix="1bec54df97dd", expected_epoch=36):

    source_checkpoint = Path(source_checkpoint)
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"기존 64패치 best.pt가 없습니다: {source_checkpoint}")
    if not expected_sha_prefix or len(expected_sha_prefix) < 12:
        raise ValueError("기준 체크포인트의 SHA 접두사를 12자리 이상 지정하세요.")
    root = Path('/content/komap_repair_v2_source')
    root.mkdir(parents=True, exist_ok=True)
    destination = root / (expected_sha_prefix + '.pt')
    if not destination.is_file() or not file_sha(destination).startswith(expected_sha_prefix):
        temporary = destination.with_suffix('.pt.tmp')
        shutil.copyfile(source_checkpoint, temporary)
        actual = file_sha(temporary)
        if not actual.startswith(expected_sha_prefix):
            temporary.unlink(missing_ok=True)
            raise ValueError("GPU1 best.pt와 내보내기의 SHA가 다릅니다. GPU1 best 내보내기를 먼저 완료하세요. 새 실험 중에는 기존 GPU1을 다시 학습하지 마세요.")
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
    print(f"공통 시작 모델: 이전 GPU1 best epoch {expected_epoch}, SHA={actual[:12]}; 새 실험은 epoch 0부터 시작합니다.")
    return result


def validate_export_dataset(export, cache, report):

    manifest = export['manifest']
    digest = manifest.get('identity', {}).get('data_sha256')
    expected = {(s, r['name']): r for s in ('valid', 'test') for r in cache[s]}
    if set(export['rows']) != set(expected):
        raise ValueError(f"Export image names/splits differ: {export['run_dir']}")
    for key, rec in expected.items():
        row = export['rows'][key]
        if row.get('image_sha256') != rec['image_sha256'] or row.get('mask_sha256') != rec['mask_sha256']:
            raise ValueError(f"Actual image/mask bytes differ: {export['run_dir']} / {key}")
    if digest == report['data_sha256']:
        return
    path = export['run_dir'] / 'data_manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f"Full train/valid manifest required for legacy data verification: {path}")
    stored = json.loads(path.read_text(encoding='utf-8'))
    rows = stored.get('files')
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Missing dataset file fingerprints: {path}")
    actual = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if actual != digest or stored.get('sha256', stored.get('data_sha256')) != digest:
        raise ValueError(f"Dataset manifest does not match export identity: {path}")

    if all('path' in row and 'sha256' in row for row in rows):
        wanted = {}
        for split in ('train', 'valid'):
            for rec in cache[split]:
                wanted[(split, rec['name'])] = rec['image_sha256']
                wanted[(split, rec['name'].replace('_image.png', '_mask.png'))] = rec['mask_sha256']
        found = {}
        for row in rows:
            key = (row['split'], Path(row['path']).name)
            if key in found:
                raise ValueError(f"Duplicate dataset fingerprint: {path} / {key}")
            found[key] = row['sha256']
        if found != wanted:
            bad = [key for key in set(found) | set(wanted) if found.get(key) != wanted.get(key)]
            raise ValueError(f"Actual train/valid names or bytes differ: {path}; examples={sorted(bad)[:5]}")
    else:
        fields = ('split', 'image', 'image_sha256', 'mask_sha256', 'shape')
        def normalize(values):
            result = {}
            for row in values:
                key = (row['split'], row['image'])
                if key in result: raise ValueError('Duplicate dataset image')
                result[key] = {field: row[field] for field in fields}
            return result
        if normalize(rows) != normalize(report['files']):
            raise ValueError(f"Actual dataset names/bytes/shapes differ: {path}")
    print('데이터 형식 호환 확인 완료 (실제 파일 SHA 일치):', export['run_dir'].name)

def prepare_fixed_blend_context(cache, report, baseline, eu_export, weights, alpha=.5):
    import komap_phase_fusion as pf
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (4,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Four positive finite specialist weights required')
    if len(baseline) != 4 or any(e['manifest'].get('target_class') != i for i,e in enumerate(baseline)):
        raise ValueError('Specialist class order must be 0,1,2,3')
    if alpha != .5 or eu_export['manifest'].get('target_class') != 2:
        raise ValueError('Use the fixed Eu replacement and alpha=0.5')
    for export in list(baseline)+[eu_export]:
        validate_export_dataset(export, cache, report)
    probabilities = {}
    for rec in cache['valid']:
        score = np.stack([pf.read_probability(e, 'valid', rec) for e in baseline])
        score[2] = pf.read_probability(eu_export, 'valid', rec)
        score *= weights[:,None,None]
        score /= np.maximum(score.sum(axis=0,keepdims=True), 1e-8)
        score.setflags(write=False)
        probabilities[rec['name']] = score
    return {'probabilities':probabilities,'identity':{
        'baseline':[e['manifest_sha256'] for e in baseline],
        'replace_Eu':eu_export['manifest_sha256'],'weights':weights.tolist(),
        'alpha':alpha,'normalization':'weighted sigmoid scores divided by channel sum'}}

def source_from_completed_control(project_root, report, source_run_name):
    import komap_phase_fusion as pf
    source = pf.load_export(Path(project_root)/'runs'/source_run_name)
    m = source['manifest']
    if m.get('target_class') != -1 or m.get('tta_rotations') != 4:
        raise ValueError('GPU1 completed four-class TTA4 export required')
    original_sha = m.get('identity',{}).get('initial_checkpoint_sha256','')
    if not original_sha.startswith('1bec54df97dd'):
        raise ValueError('Source must be the previous GPU1 control from the original e036 checkpoint')
    info = prepare_source_checkpoint(source['run_dir']/'best.pt',report,
                                     m['checkpoint_sha256'][:12],m['checkpoint_epoch'])
    if info['sha256'] != m['checkpoint_sha256']:
        raise ValueError('Source model and completed export checksum differ')
    info['source_export'] = source
    return info


def configure_encoder_training(model, c, epoch):
    if c.architecture == 'dinov3':
        model.encoder.requires_grad_(False)
        count = c.dino_trainable_blocks
        if count and epoch > c.freeze_encoder_epochs:
            layers = getattr(model.encoder,'layer',None)
            if layers is None or len(layers) != 12:
                raise ValueError('Pinned DINOv3 S/16 requires encoder.layer with 12 blocks')
            for layer in layers[-count:]:layer.requires_grad_(True)
        model.encoder.eval()
    else:
        model.encoder.requires_grad_(epoch > c.freeze_encoder_epochs)
        for module in model.encoder.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
    return not any(p.requires_grad for p in model.encoder.parameters())


def load_start_weights(model,weights,c):
    if not c.wide_context:
        model.load_state_dict(weights,strict=True)
        return
    missing,unexpected=model.load_state_dict(weights,strict=False)
    allowed={key for key in model.state_dict() if key.startswith('decoder.wide_context_adapter.')}
    if set(missing)!=allowed or unexpected:
        raise ValueError(f'Warm-start schema mismatch: missing={missing}, unexpected={unexpected}')
    print('기존 가중치 로드 완료; 새 문맥 adapter의 마지막 층은 0으로 초기화했습니다.')

"""Extra routines appended to the grounded Colab RAM implementation."""


EU_SIZE_BINS = ((1, 16), (17, 64), (65, 256), (257, None))
SCALE_MODES = ("native_1x", "zoom_2x", "mean_1x_2x")


def build_eu_component_pool(mask):

    from scipy import ndimage
    labels, count = ndimage.label(mask == 2, structure=np.ones((3, 3), np.uint8))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:].astype(np.int64)
    pixels = np.flatnonzero(labels.ravel()).astype(np.int32)
    ids = labels.ravel()[pixels]
    order = np.argsort(ids, kind="stable")
    pixels = pixels[order]
    offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
    groups = [np.flatnonzero((sizes >= lo) & (sizes <= hi if hi is not None else True)).astype(np.int32)
              for lo, hi in EU_SIZE_BINS]
    for array in [pixels, sizes, offsets] + groups:
        array.setflags(write=False)
    return {"pixels": pixels, "sizes": sizes, "offsets": offsets, "groups": groups}


def draw_eu_component_pixel(pool, rng):

    groups = [group for group in pool["groups"] if group.size]
    if not groups:
        return None
    group = groups[int(rng.integers(len(groups)))]
    component = int(group[int(rng.integers(group.size))])
    first, last = pool["offsets"][component:component + 2]
    return int(pool["pixels"][int(rng.integers(first, last))])


def zoom_plan_for_image(c, epoch, rec_id):

    count = int(math.floor(c.patches_per_image * c.zoom_probability + 0.5))
    rng = np.random.default_rng(np.random.SeedSequence([c.seed, epoch, rec_id, 903]))
    plan = np.zeros(c.patches_per_image, dtype=bool)
    plan[rng.permutation(c.patches_per_image)[:count]] = True
    return plan


def eu_pool_summary(records, pools, c):
    rows, totals = [], np.zeros(4, dtype=np.int64)
    for rec, pool in zip(records, pools):
        counts = [int(group.size) for group in pool["groups"]]
        totals += counts
        rows.append({"image": rec["name"], "components_by_bin": counts,
                     "eu_pixels": int(pool["pixels"].size)})
    zoom_per_image = int(math.floor(c.patches_per_image * c.zoom_probability + 0.5))
    return {"eu_component_balanced": c.eu_component_balanced,
            "connectivity": 8, "size_bins_native_pixels": [[lo, hi] for lo, hi in EU_SIZE_BINS],
            "sampling": "uniform nonempty bin -> uniform component -> uniform true pixel; Eu-targeted branch only",
            "components_by_bin": totals.tolist(), "per_image": rows,
            "zoom_crops_per_image": zoom_per_image,
            "native_crops_per_image": c.patches_per_image - zoom_per_image,
            "zoom_native_crop": c.zoom_crop, "output_patch": c.patch,
            "note": "Counts are for nominal crop sizes; the existing 0.9-1.1 jitter remains active."}


def checked_probability(probability, shape):
    p = np.asarray(probability, dtype=np.float32)
    if p.shape != (4, *shape) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1.00001:
        raise ValueError("Invalid four-class native probability map")
    if not np.allclose(p.sum(axis=0), 1, atol=1e-4):
        raise ValueError("Four-class probabilities must sum to one")
    return p


def probability_at_scale(model, image, c, gpu_input, amp_dtype, scale):

    if scale not in (1, 2):
        raise ValueError("Only the prespecified 1x and 2x scales are compared")
    h, w = image.shape[:2]
    if scale == 1:
        return checked_probability(predict_probability(model, image, c, gpu_input, amp_dtype, 4), (h, w))
    enlarged = cv2.resize(image, (2*w, 2*h), interpolation=cv2.INTER_LINEAR)
    large = predict_probability(model, enlarged, c, gpu_input, amp_dtype, 4)
    restored = np.stack([cv2.resize(channel, (w, h), interpolation=cv2.INTER_LINEAR)
                         for channel in large]).astype(np.float32)
    restored /= np.maximum(restored.sum(axis=0, keepdims=True), 1e-12)
    return checked_probability(restored, (h, w))


def eu_component_statistics(target, prediction):

    from scipy import ndimage
    labels, count = ndimage.label(target == 2, structure=np.ones((3, 3), np.uint8))
    size = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    correct = np.bincount(labels[(target == 2) & (prediction == 2)], minlength=count + 1)[1:]
    result = {}
    for lo, hi in EU_SIZE_BINS:
        suffix = f"{lo}_{hi}" if hi is not None else f"{lo}_plus"
        selected = (size >= lo) & (size <= hi if hi is not None else True)
        result[f"Eu_{suffix}_gt_pixels"] = int(size[selected].sum())
        result[f"Eu_{suffix}_correct_pixels"] = int(correct[selected].sum())
        result[f"Eu_{suffix}_components"] = int(selected.sum())
        result[f"Eu_{suffix}_missed_components"] = int((correct[selected] == 0).sum())
    return result


def score_native_maps(records, probabilities, fusion_context):
    rows, solo_rows = [], []
    cm, solo_cm = np.zeros((4, 4), np.int64), np.zeros((4, 4), np.int64)
    alpha = fusion_context["identity"]["alpha"]
    for rec in records:
        p = checked_probability(probabilities[rec["name"]], rec["image"].shape[:2])
        fixed = fusion_context["probabilities"][rec["name"]]
        blend = checked_probability((1-alpha)*fixed + alpha*p, rec["image"].shape[:2])
        pred = blend.argmax(axis=0).astype(np.uint8)
        row, one = evaluate_labels(rec["name"], rec["mask"], pred)
        solo, one_solo = evaluate_labels(rec["name"], rec["mask"], p.argmax(axis=0).astype(np.uint8))
        row["standalone_miou_ignore"] = solo["miou_ignore"]
        row["standalone_miou_zero"] = solo["miou_zero"]
        row.update(eu_component_statistics(rec["mask"], pred))
        rows.append(row); solo_rows.append(solo)
        cm += one; solo_cm += one_solo
    summary = summarize(rows, cm)
    summary["standalone"] = summarize(solo_rows, solo_cm)
    diagnostics = {}
    for lo, hi in EU_SIZE_BINS:
        suffix = f"{lo}_{hi}" if hi is not None else f"{lo}_plus"
        counts = {field: sum(row[f"Eu_{suffix}_{field}"] for row in rows)
                  for field in ("gt_pixels", "correct_pixels", "components", "missed_components")}
        counts["pixel_recall"] = counts["correct_pixels"] / counts["gt_pixels"] if counts["gt_pixels"] else None
        counts["completely_missed_fraction"] = counts["missed_components"] / counts["components"] if counts["components"] else None
        diagnostics[suffix] = counts
    summary["eu_component_diagnostics"] = diagnostics
    summary["checkpoint_selection"] = "fixed Eu-replaced specialist/DINO blend; alpha=0.5"
    return summary, rows


def choose_scale(summaries, absent_policy):

    chosen = SCALE_MODES[0]
    for mode in SCALE_MODES[1:]:
        if summaries[mode]["miou_" + absent_policy] > summaries[chosen]["miou_" + absent_policy] + 1e-8:
            chosen = mode
    return chosen


def compare_scale_candidates(model, records, c, gpu_input, amp_dtype, fusion_context):
    maps = {mode: {} for mode in SCALE_MODES}
    for rec in tqdm(records, desc="valid20: 회전4 TTA, 1x / 2x"):
        native = probability_at_scale(model, rec["image"], c, gpu_input, amp_dtype, 1)
        zoom = probability_at_scale(model, rec["image"], c, gpu_input, amp_dtype, 2)
        maps["native_1x"][rec["name"]] = native
        maps["zoom_2x"][rec["name"]] = zoom
        maps["mean_1x_2x"][rec["name"]] = checked_probability((native+zoom)*0.5, rec["image"].shape[:2])
    summaries, rows_by_mode = {}, {}
    for mode in SCALE_MODES:
        summaries[mode], rows_by_mode[mode] = score_native_maps(records, maps[mode], fusion_context)
    chosen = choose_scale(summaries, c.absent_policy)
    original = summaries["native_1x"]["miou_" + c.absent_policy]
    comparison = []
    for mode in SCALE_MODES:
        s = summaries[mode]
        tiny = s["eu_component_diagnostics"]["1_16"]
        comparison.append({"candidate": mode, "selected": int(mode == chosen),
            "blend_miou_ignore": s["miou_ignore"], "blend_miou_zero": s["miou_zero"],
            "delta_vs_native": s["miou_" + c.absent_policy]-original,
            "solo_miou_ignore": s["standalone"]["miou_ignore"],
            "Al3Ni_iou": s["class_iou_ignore"]["Al3Ni"],
            "EutecticSi_iou": s["class_iou_ignore"]["EutecticSi"],
            "Eu_1_16_pixel_recall": tiny["pixel_recall"],
            "Eu_1_16_completely_missed_fraction": tiny["completely_missed_fraction"]})
    selected = maps.pop(chosen)
    maps.clear()
    return chosen, selected, summaries, rows_by_mode, comparison


def test_specialist_probability(rec, baseline, eu_export, weights):
    import komap_phase_fusion as pf
    score = np.stack([pf.read_probability(e, "test", rec) for e in baseline])
    score[2] = pf.read_probability(eu_export, "test", rec)
    score *= np.asarray(weights, np.float32)[:, None, None]
    score /= np.maximum(score.sum(axis=0, keepdims=True), 1e-8)
    return checked_probability(score, rec["image"].shape[:2])


def write_probability_pair(root, split, rec, probability, specialist, alpha):

    root = Path(root)
    p = checked_probability(probability, rec["image"].shape[:2])
    combined = checked_probability((1-alpha)*specialist + alpha*p, rec["image"].shape[:2])
    stem = Path(rec["name"]).stem
    path = root/split/(stem+".npz")
    blend_path = root/("blend_"+split)/(stem+".npz")
    for output, array in ((path, p), (blend_path, combined)):
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as f:
            np.savez_compressed(f, probabilities=array, class_ids=np.arange(4, dtype=np.int16))
    mask_name = rec["name"].replace("_image.png", "_mask.png")
    raw_mask = root/("raw_"+split+"_masks")/mask_name
    final_mask = root/("selected_"+split+"_masks")/mask_name
    kd.save_rgb_mask(p.argmax(axis=0).astype(np.uint8), raw_mask)
    kd.save_rgb_mask(combined.argmax(axis=0).astype(np.uint8), final_mask)
    return {"split": split, "image": rec["name"], "image_sha256": rec["image_sha256"],
            "mask_sha256": rec["mask_sha256"], "shape": list(p.shape),
            "probability_file": path.relative_to(root).as_posix(), "probability_sha256": file_sha(path),
            "final_blend_probability_file": blend_path.relative_to(root).as_posix(),
            "final_blend_probability_sha256": file_sha(blend_path),
            "raw_mask_file": raw_mask.relative_to(root).as_posix(),
            "selected_mask_file": final_mask.relative_to(root).as_posix()}


def zip_final_masks(root, split):
    root = Path(root)
    paths = sorted((root/("selected_"+split+"_masks")).glob("*.png"))
    if len(paths) != kd.EXPECTED_COUNTS[split]:
        raise ValueError("Incomplete native final mask collection")
    with zipfile.ZipFile(root/("selected_"+split+"_masks.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, arcname=path.name)


def verify_scale_export(root, manifest, cache):
    import komap_phase_fusion as pf
    root = Path(root)
    if manifest.get("complete") is not True:
        raise ValueError("Incomplete scale comparison export")
    for item in manifest["artifacts"]:
        path = pf.contained_path(root, item["path"])
        if file_sha(path) != item["sha256"]:
            raise ValueError(f"Export artifact changed: {path}")
    export = {"directory": root, "manifest": manifest,
              "rows": {(r["split"], r["image"]): r for r in manifest["files"]}}
    expected = {(s, r["name"]) for s in ("valid", "test") for r in cache[s]}
    if len(export["rows"]) != len(manifest["files"]) or set(export["rows"]) != expected:
        raise ValueError("Export native image list differs")
    for split in ("valid", "test"):
        for rec in cache[split]:
            pf.read_probability(export, split, rec)


def report_scale_export(export_dir):
    export_dir = Path(export_dir)
    result = json.loads((export_dir/"validation_summary.json").read_text(encoding="utf-8"))
    chosen, policy = result["selected_scale_mode"], result["absent_policy"]
    score = result["selected"]["miou_"+policy]
    original = result["candidates"]["native_1x"]["miou_"+policy]
    print(f"선택: {chosen}; 최종 결합 valid mIoU={score:.8f}; 같은 best 1x 대비={score-original:+.8f}")
    print("Drive 확률·결과:", export_dir)
    print("비교 CSV:", export_dir/"scale_comparison.csv")
    print("최종 test 결합 마스크:", export_dir/"selected_test_masks.zip")
    print("분석용 파일:", export_dir/"analysis_bundle.zip")


def export_scale_comparison(cache, report, run_dir, checkpoint_path, fusion_context,
                            baseline, eu_export, weights, source_info, trained=False):

    if not torch.cuda.is_available():
        raise RuntimeError("Colab GPU required for probability export")
    run_dir, checkpoint_path = Path(run_dir), Path(checkpoint_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    sha = file_sha(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    c = Config(**state["config"])
    if not trained:


        c.select_on_fixed_blend = True
        c.validation_tta_rotations = 4
    c.validate()
    if c.architecture != "dinov3" or c.wide_context or c.dino_trainable_blocks:
        raise ValueError("This comparison requires the unchanged frozen DINO detail decoder")
    identity = state["identity"]
    if identity.get("family") != MODEL_FAMILY or identity.get("data_sha256") != report["data_sha256"]:
        raise ValueError("Checkpoint dataset/model identity differs")
    expected_fusion = {"baseline": [e["manifest_sha256"] for e in baseline],
        "replace_Eu": eu_export["manifest_sha256"], "weights": np.asarray(weights, np.float32).tolist(),
        "alpha": 0.5, "normalization": "weighted sigmoid scores divided by channel sum"}
    if fusion_context["identity"] != expected_fusion:
        raise ValueError("Valid/test must use the same fixed specialist blend")
    if trained:
        if any(identity.get(k) != v for k, v in _identity(c, report).items()):
            raise ValueError("Training export code/settings/data differ from the saved best")
        if identity.get("initial_checkpoint_sha256") != source_info["sha256"] or identity.get("fusion_context") != expected_fusion:
            raise ValueError("Training source or blend changed")
    elif sha != source_info["sha256"] or state["epoch"] != source_info["source_epoch"]:
        raise ValueError("Inference requires the confirmed unchanged source best")
    metadata = {"experiment": "mixed_zoom_training" if trained else "existing_best_scale_inference",
        "checkpoint_sha256": sha, "checkpoint_epoch": state["epoch"],
        "common_source_sha256": source_info["sha256"], "data_sha256": report["data_sha256"],
        "fusion_context": expected_fusion, "scales": [1, 2], "mean_weights": [0.5, 0.5],
        "tta_rotations": 4, "output_resolution": "original image height/width",
        "validation_selection": "mean per-image 4-class fixed-blend mIoU; valid20 only",
        "absent_policy": c.absent_policy, "config": asdict(c),
        "saved_checkpoint_config": state["config"],
        "export_code_sha256": file_sha(__file__), "data_code_sha256": file_sha(kd.__file__)}
    if not trained:
        identity_path = run_dir/"identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("This inference RUN_NAME has different settings. Use a new RUN_NAME.")
        atomic_json(identity_path, metadata)
        atomic_json(run_dir/"data_manifest.json", report)
    export_key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:10]
    export_dir = run_dir/"exports"/f"best_e{state['epoch']:03d}_{sha[:12]}_ms1x2x_tta4_{export_key}"
    manifest_path = export_dir/"manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("scale_export_identity") != metadata:
            raise ValueError("Existing scale export has different settings")
        verify_scale_export(export_dir, manifest, cache)
        atomic_json(run_dir/"export_index.json", {"manifest": manifest_path.relative_to(run_dir).as_posix()})
        del state
        print("완료된 기존 내보내기를 사용합니다.")
        report_scale_export(export_dir)
        return export_dir
    if export_dir.exists():
        raise ValueError("Unexpected incomplete export directory. Use a new RUN_NAME.")
    local_root = Path("/content/komap_nieu_zoom_exports")/run_dir.name/export_dir.name
    if local_root.exists():
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    model = build_model(c, pretrained=False, backbone_config=state.get("backbone_config")).to(
        "cuda", memory_format=torch.channels_last)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    gpu_input = GPUInput(torch.device("cuda"))
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print("PNG 재디코딩 없이 RAM 이미지로 1x / 2x를 추론합니다. 2x는 타일 수와 추론 시간이 늘어납니다.")
    selected = None
    try:
        chosen, selected, summaries, rows_by_mode, comparison = compare_scale_candidates(
            model, cache["valid"], c, gpu_input, amp_dtype, fusion_context)
        files = []
        for split in ("valid", "test"):
            for rec in tqdm(cache[split], desc=split+" 원본 크기 확률·결합 마스크"):
                if split == "valid":
                    p = selected[rec["name"]]
                    specialist = fusion_context["probabilities"][rec["name"]]
                else:
                    specialist = test_specialist_probability(rec, baseline, eu_export, weights)
                    if chosen == "native_1x":
                        p = probability_at_scale(model, rec["image"], c, gpu_input, amp_dtype, 1)
                    elif chosen == "zoom_2x":
                        p = probability_at_scale(model, rec["image"], c, gpu_input, amp_dtype, 2)
                    else:
                        p = (probability_at_scale(model, rec["image"], c, gpu_input, amp_dtype, 1)
                             + probability_at_scale(model, rec["image"], c, gpu_input, amp_dtype, 2))*0.5
                files.append(write_probability_pair(local_root, split, rec, p, specialist, 0.5))
            zip_final_masks(local_root, split)
        write_csv(local_root/"scale_comparison.csv", comparison)
        all_rows = [{"scale_mode": mode, **row} for mode in SCALE_MODES for row in rows_by_mode[mode]]
        write_csv(local_root/"validation_scale_per_image.csv", all_rows)
        write_csv(local_root/"validation_selected_per_image.csv", rows_by_mode[chosen])
        result = {"selected_scale_mode": chosen, "selected_tta_rotations": 4,
            "absent_policy": c.absent_policy, "selected": summaries[chosen], "candidates": summaries,
            "selection_note": "Selected on valid20, not an independent test score; test10 has no GT."}
        atomic_json(local_root/"validation_summary.json", result)
        atomic_json(local_root/"export_settings.json", {**metadata, "selected_scale_mode": chosen})
        atomic_json(local_root/"checkpoint_selection.json", {
            "trained": trained, "checkpoint_epoch": state["epoch"],
            "training_selection": state.get("validation"),
            "final_selection": "same saved best; compare native1x, zoom2x, equal probability mean on fixed blend",
            "epoch_zero_protection": "training best includes the unmodified source evaluated at native1x" if trained else "source weights unchanged"})
        with zipfile.ZipFile(local_root/"analysis_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("scale_comparison.csv", "validation_scale_per_image.csv", "validation_selected_per_image.csv",
                         "validation_summary.json", "export_settings.json", "checkpoint_selection.json"):
                z.write(local_root/name, arcname=name)
            for path in sorted((local_root/"selected_valid_masks").glob("*.png")):
                z.write(path, arcname="selected_valid_masks/"+path.name)
        artifacts = [{"path": path.relative_to(local_root).as_posix(), "sha256": file_sha(path)}
                     for path in sorted(local_root.rglob("*")) if path.is_file()]
        manifest = {"complete": True, "format_version": 2, "model_family": MODEL_FAMILY,
            "target_class": -1, "class_names": list(kd.CLASS_NAMES), "palette": kd.PALETTE.tolist(),
            "config": asdict(c), "checkpoint_epoch": state["epoch"], "checkpoint_sha256": sha,
            "identity": identity, "dtype": "float32", "tta_rotations": 4,
            "probability": "raw DINO probabilities; selected image scale restored to native size",
            "inference_scale_mode": chosen, "scale_export_identity": metadata,
            "final_blend": expected_fusion, "calibration": "none", "files": files, "artifacts": artifacts}
        atomic_json(local_root/"manifest.json", manifest)
        export_dir.parent.mkdir(parents=True, exist_ok=True)
        pending = export_dir.with_name(export_dir.name+".pending")
        if pending.exists():
            shutil.rmtree(pending)
        shutil.copytree(local_root, pending)
        verify_scale_export(pending, manifest, cache)
        os.replace(pending, export_dir)
        atomic_json(run_dir/"export_index.json", {"manifest": manifest_path.relative_to(run_dir).as_posix()})
    finally:
        del model, selected, state
        gc.collect()
        torch.cuda.empty_cache()
    report_scale_export(export_dir)
    return export_dir
