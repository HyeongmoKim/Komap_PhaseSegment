'YOLO11 backbone과 semantic decoder의 학습·추론.'
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm.auto import tqdm
import komap_specialist_data as kd
import komap_phase_fusion as pf
import komap_nieu_zoom_ram as core

PROBABILITY_FAMILY = core.MODEL_FAMILY
EXPERIMENT_FAMILY = "komap_metal_paper_ideas_semantic_v1"
MIX_WEIGHTS = (0.0, 0.25, 0.5, 1.0)


@dataclass
class Config(core.Config):
    architecture: str = "mpu_inspired"
    encoder: str = "resnet34"
    encoder_lr: float = 3e-5
    decoder_lr: float = 3e-4
    epochs: int = 60
    warmup_epochs: int = 3
    freeze_encoder_epochs: int = 1
    patches_per_image: int = 32
    eval_every: int = 2
    zoom_probability: float = 0.5
    zoom_crop: int = 240
    dropout: float = 0.1
    boundary_weight: float = 0.1
    boundary_ramp_epochs: int = 5
    denoise_max_strength: float = 0.25
    freeze_encoder_bn: bool = True
    yolo_preset: str = "yolo11s"
    fixed_new_model_weight: float = 0.25

    def validate(self):
        if self.architecture not in ("dnu_inspired", "mpu_inspired", "yolo11_edge_semantic", "sem_ebsd_fusion"):
            raise ValueError("Unknown paper-idea adaptation")

        shared = {k: getattr(self, k) for k in core.Config.__dataclass_fields__}
        shared.update(architecture="dinov3", wide_context=False, dino_trainable_blocks=0)
        core.Config(**shared).validate()
        if self.encoder != "resnet34" or self.yolo_preset != "yolo11s":
            raise ValueError("This version uses fixed ResNet34 or YOLO11s backbones")
        if not 0 <= self.dropout < 1 or not 0 <= self.boundary_weight <= 1:
            raise ValueError("Invalid dropout/boundary weight")
        if type(self.boundary_ramp_epochs) is not int or self.boundary_ramp_epochs < 1:
            raise ValueError("boundary_ramp_epochs must be a positive integer")
        if not 0 <= self.denoise_max_strength <= 0.5 or type(self.freeze_encoder_bn) is not bool:
            raise ValueError("Invalid denoising/encoder normalization setting")
        if self.fixed_new_model_weight != 0.25 or self.validation_tta_rotations != 4:
            raise ValueError("Training criterion is fixed 25% new model and rotation4 TTA")


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        groups = math.gcd(cout, 8)
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.GELU())


class GentleDenoise(nn.Module):

    def __init__(self, maximum):
        super().__init__()
        axis = torch.arange(-1, 2, dtype=torch.float32)
        kernel = torch.exp(-(axis[:, None]**2 + axis[None, :]**2)/(2*0.8**2))
        kernel /= kernel.sum()
        self.register_buffer("kernel", kernel[None, None].repeat(3, 1, 1, 1))
        self.gate = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 3, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -3.0)
        self.maximum = maximum

    def forward(self, x):
        smooth = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), self.kernel.to(x.dtype), groups=3)
        amount = self.maximum * self.gate(x).sigmoid()
        return x + amount*(smooth-x)


class UnetAdaptation(nn.Module):
    def __init__(self, c, pretrained):
        super().__init__()
        import segmentation_models_pytorch as smp
        unet = smp.Unet(encoder_name=c.encoder, encoder_weights="imagenet" if pretrained else None,
                        classes=4, in_channels=3, decoder_interpolation="bilinear")
        self.encoder, self.decoder, self.segmentation_head = unet.encoder, unet.decoder, unet.segmentation_head
        self.architecture, self.dropout = c.architecture, nn.Dropout2d(c.dropout)
        self.input_filter = GentleDenoise(c.denoise_max_strength) if c.architecture == "dnu_inspired" else nn.Identity()
        self.boundary_head = nn.Sequential(ConvBlock(16, 16), nn.Conv2d(16, 1, 1)) if c.architecture == "mpu_inspired" else None
        self.physical_path = ConvBlock(2, 16) if c.architecture == "sem_ebsd_fusion" else None
        self.physical_fuse = ConvBlock(32, 16) if self.physical_path is not None else None

    def forward_tasks(self, x):
        rgb = self.input_filter(x[:, :3])
        features = self.encoder(rgb)
        decoded = self.decoder(features)
        if self.physical_path is not None:
            if x.shape[1] != 5:
                raise ValueError("SEM+EBSD model needs RGB + measured KAM + pattern quality")
            decoded = self.physical_fuse(torch.cat([decoded, self.physical_path(x[:, 3:])], dim=1))
        logits = self.segmentation_head(self.dropout(decoded))
        boundary = self.boundary_head(decoded) if self.boundary_head is not None else None
        return logits, boundary

    def forward(self, x):
        return self.forward_tasks(x)[0]


class CascadedGroupAttention(nn.Module):

    def __init__(self, channels=128, groups=4):
        super().__init__()
        if channels % groups:
            raise ValueError("Attention channels must divide into groups")
        self.groups, self.width = groups, channels//groups
        self.qkv = nn.ModuleList(nn.Conv2d(self.width, 3*self.width, 1) for _ in range(groups))
        self.local_q = nn.ModuleList(nn.Conv2d(self.width, self.width, 3, padding=1, groups=self.width) for _ in range(groups))
        self.projection = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.projection.weight); nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        outputs, previous = [], None
        for index, chunk in enumerate(x.chunk(self.groups, dim=1)):
            q, k, v = self.qkv[index](chunk if previous is None else chunk+previous).chunk(3, dim=1)
            q = self.local_q[index](q).flatten(2)
            k, v = k.flatten(2), v.flatten(2)
            attention = ((q.transpose(1, 2).float() @ k.float())/math.sqrt(self.width)).softmax(dim=-1)
            previous = (v.float() @ attention.transpose(1, 2)).to(x.dtype).reshape_as(chunk)
            outputs.append(previous)
        return x + self.projection(torch.cat(outputs, dim=1))


class LearnedPointUp(nn.Module):

    def __init__(self, channels, groups=4):
        super().__init__()
        if channels % groups:
            raise ValueError("Upsampling channels must divide into groups")
        self.groups = groups
        self.offset = nn.Conv2d(channels, 2*groups, 1)
        nn.init.zeros_(self.offset.weight); nn.init.zeros_(self.offset.bias)

    def forward(self, x, size=None):
        b, c, h, w = x.shape
        oh, ow = (2*h, 2*w) if size is None else size
        delta = F.interpolate(self.offset(x).float(), size=(oh, ow), mode="bilinear", align_corners=False)
        delta = 0.25*delta.tanh().reshape(b, self.groups, 2, oh, ow)
        yy = (torch.arange(oh, device=x.device, dtype=torch.float32)+0.5)*h/oh - 0.5
        xx = (torch.arange(ow, device=x.device, dtype=torch.float32)+0.5)*w/ow - 0.5
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        grid_x = ((gx[None, None]+delta[:, :, 0]+0.5)*2/w-1)
        grid_y = ((gy[None, None]+delta[:, :, 1]+0.5)*2/h-1)
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(b*self.groups, oh, ow, 2)

        sampled = F.grid_sample(x.float().reshape(b*self.groups, c//self.groups, h, w), grid,
                                mode="bilinear", padding_mode="border", align_corners=False)
        return sampled.reshape(b, c, oh, ow).to(x.dtype)


class YoloEdgeSemantic(nn.Module):

    def __init__(self, c, pretrained):
        super().__init__()
        if pretrained:
            from ultralytics import YOLO
            donor = YOLO(c.yolo_preset+".pt").model
        else:
            from ultralytics.nn.tasks import DetectionModel
            donor = DetectionModel(cfg=c.yolo_preset+".yaml", nc=80, verbose=False)
        layers = list(donor.model[:11])
        if len(layers) != 11 or any(layer.f != -1 for layer in layers):
            raise ValueError("Pinned YOLO11 backbone graph changed")
        self.encoder = nn.ModuleList(layers)
        self.feature_indices = (2, 4, 6, 10)
        channels = [self.encoder[index].cv2.conv.out_channels for index in self.feature_indices]
        self.lateral = nn.ModuleList(nn.Conv2d(cin, 128, 1) for cin in channels)
        self.attention = CascadedGroupAttention(128)
        self.upsamplers = nn.ModuleList(LearnedPointUp(128) for _ in range(3))
        self.pyramid = nn.ModuleList(ConvBlock(256, 128) for _ in range(3))
        self.reduce = nn.Conv2d(128, 64, 1)
        self.full_up = LearnedPointUp(64)
        self.edge_path = ConvBlock(4, 16)
        self.fuse = ConvBlock(80, 32)
        self.segmentation_head = nn.Conv2d(32, 4, 1)
        self.dropout = nn.Dropout2d(c.dropout)
        sobel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer("sobel", torch.stack([sobel, sobel.T])[:, None])
        del donor

    def forward_tasks(self, x):
        collected, y = [], x
        for index, layer in enumerate(self.encoder):
            y = layer(y)
            if index in self.feature_indices:
                collected.append(y)
        lateral = [projection(feature) for projection, feature in zip(self.lateral, collected)]
        y = self.attention(lateral[-1])
        for index, skip in enumerate(reversed(lateral[:-1])):
            y = self.pyramid[index](torch.cat([self.upsamplers[index](y, skip.shape[-2:]), skip], dim=1))
        y = self.full_up(self.reduce(y), x.shape[-2:])
        edge_xy = F.conv2d(F.pad(x.mean(dim=1, keepdim=True), (1, 1, 1, 1), mode="reflect"), self.sobel.to(x.dtype))
        magnitude = ((edge_xy.float().square().sum(dim=1, keepdim=True)+1e-8).sqrt()/4).clamp(0, 1).to(x.dtype)
        detail = self.edge_path(torch.cat([x, magnitude], dim=1))
        logits = self.segmentation_head(self.dropout(self.fuse(torch.cat([y, detail], dim=1))))
        return logits, None

    def forward(self, x):
        return self.forward_tasks(x)[0]


def build_model(c, pretrained):
    c.validate()
    return YoloEdgeSemantic(c, pretrained) if c.architecture == "yolo11_edge_semantic" else UnetAdaptation(c, pretrained)


class GPUInput:
    def __init__(self, device, c):
        self.device, self.c = device, c
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def __call__(self, batch, config=None):
        raw = batch.to(self.device, non_blocking=True).float()
        x, physical = raw[:, :3]/255, raw[:, 3:]
        if config is not None:
            c, shape = config, (x.shape[0], 1, 1, 1)
            active = torch.rand(shape, device=self.device) < c.brightness_contrast_probability
            contrast = 1+(torch.rand(shape, device=self.device)*2-1)*c.contrast*active
            brightness = (torch.rand(shape, device=self.device)*2-1)*c.brightness*active
            x = ((x-0.5)*contrast+0.5+brightness).clamp(0, 1)
            active = torch.rand(shape, device=self.device) < c.gamma_probability
            gamma = c.gamma_min+torch.rand(shape, device=self.device)*(c.gamma_max-c.gamma_min)
            x = x.pow(torch.where(active, gamma, torch.ones_like(gamma)))
            active = torch.rand(shape, device=self.device) < c.noise_probability
            x = (x+torch.randn_like(x)*c.noise_std*active).clamp(0, 1)
        if self.c.architecture != "yolo11_edge_semantic":
            x = (x-self.mean)/self.std
        if self.c.architecture == "sem_ebsd_fusion":
            if physical.shape[1] != 2:
                raise ValueError("Physical input channels missing")
            x = torch.cat([x, physical], dim=1)
        elif raw.shape[1] != 3:
            raise ValueError("This notebook expects only RGB SEM input")
        return x.contiguous(memory_format=torch.channels_last)


def boundary_labels_numpy(labels):
    labels = np.asarray(labels)
    if labels.ndim not in (2, 3):
        raise ValueError("Expected H,W or B,H,W class labels")
    edge = np.zeros(labels.shape, dtype=bool)
    valid = labels != 255
    horizontal = (labels[..., :, 1:] != labels[..., :, :-1]) & valid[..., :, 1:] & valid[..., :, :-1]
    vertical = (labels[..., 1:, :] != labels[..., :-1, :]) & valid[..., 1:, :] & valid[..., :-1, :]
    edge[..., :, 1:] |= horizontal; edge[..., :, :-1] |= horizontal
    edge[..., 1:, :] |= vertical; edge[..., :-1, :] |= vertical
    return edge


def boundary_target(labels):
    valid = labels != 255
    edge = torch.zeros_like(labels, dtype=torch.bool)
    horizontal = (labels[:, :, 1:] != labels[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    vertical = (labels[:, 1:, :] != labels[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    edge[:, :, 1:] |= horizontal; edge[:, :, :-1] |= horizontal
    edge[:, 1:, :] |= vertical; edge[:, :-1, :] |= vertical
    return edge[:, None].float(), valid[:, None]


def losses(logits, boundary, labels, c, epoch):
    main = core.segmentation_loss(logits, labels, c.dice_weight)
    auxiliary = main.new_zeros(())
    weight = 0.0
    if boundary is not None:
        target, valid = boundary_target(labels)
        positive = (target*valid).sum()
        negative = valid.sum()-positive
        positive_weight = (negative/positive.clamp_min(1)).clamp(1, 10).detach()
        bce = F.binary_cross_entropy_with_logits(boundary.float(), target, pos_weight=positive_weight, reduction="none")
        bce = (bce*valid).sum()/valid.sum().clamp_min(1)
        p = boundary.float().sigmoid()*valid
        dice = 1-(2*(p*target).sum()+1e-6)/(p.sum()+(target*valid).sum()+1e-6)
        auxiliary = bce+0.25*dice
        weight = c.boundary_weight*min(1.0, epoch/c.boundary_ramp_epochs)
    return main+weight*auxiliary, main.detach(), auxiliary.detach(), weight


def prepare_current_baseline(cache, report, source_info, specialists, eu_export, weights):

    source = source_info["source_export"]
    core.validate_export_dataset(source, cache, report)
    specialist_context = core.prepare_fixed_blend_context(cache, report, specialists, eu_export, weights, 0.5)
    probabilities = {"valid": {}, "test": {}}
    for split in ("valid", "test"):
        for rec in tqdm(cache[split], desc=split+": 기존 최종 조합 확률 → RAM"):
            if split == "valid":
                fixed = specialist_context["probabilities"][rec["name"]]
            else:
                fixed = core.test_specialist_probability(rec, specialists, eu_export, weights)
            dino = pf.read_probability(source, split, rec)
            p = core.checked_probability(0.5*fixed+0.5*dino, rec["image"].shape[:2])
            p.setflags(write=False)
            probabilities[split][rec["name"]] = p
    identity = {"specialists": specialist_context["identity"], "source_export_sha256": source["manifest_sha256"],
                "source_checkpoint_sha256": source_info["sha256"], "source_weight": 0.5,
                "description": "Eu-replaced specialists 50% + previous control DINO best 50%"}
    summary, rows = score_maps(cache["valid"], probabilities["valid"])
    print(f"기존 최종 조합 valid mIoU={summary['miou_ignore']:.8f}")
    return {"identity": identity, "probabilities": probabilities, "validation": summary, "rows": rows}


def score_maps(records, probabilities):
    rows, cm = [], np.zeros((4, 4), np.int64)
    for rec in records:
        p = core.checked_probability(probabilities[rec["name"]], rec["image"].shape[:2])
        prediction = p.argmax(axis=0).astype(np.uint8)
        row, one = core.evaluate_labels(rec["name"], rec["mask"], prediction)
        row.update(core.eu_component_statistics(rec["mask"], prediction))
        rows.append(row); cm += one
    summary = core.summarize(rows, cm)
    diagnostics = {}
    for lo, hi in core.EU_SIZE_BINS:
        suffix = f"{lo}_{hi}" if hi is not None else f"{lo}_plus"
        counts = {name:sum(row[f"Eu_{suffix}_{name}"] for row in rows)
                  for name in ("gt_pixels", "correct_pixels", "components", "missed_components")}
        counts["pixel_recall"] = counts["correct_pixels"]/counts["gt_pixels"] if counts["gt_pixels"] else None
        counts["completely_missed_fraction"] = counts["missed_components"]/counts["components"] if counts["components"] else None
        diagnostics[suffix] = counts
    summary["eu_component_diagnostics"] = diagnostics
    return summary, rows


def evaluate(model, records, c, gpu_input, amp_dtype, baseline_context):
    solo_maps, fixed_maps = {}, {}
    for rec in records:
        p = core.predict_probability(model, rec["image"], c, gpu_input, amp_dtype, 4)
        p = core.checked_probability(p, rec["image"].shape[:2])
        solo_maps[rec["name"]] = p
        old = baseline_context["probabilities"]["valid"][rec["name"]]
        fixed_maps[rec["name"]] = (1-c.fixed_new_model_weight)*old+c.fixed_new_model_weight*p
    solo, solo_rows = score_maps(records, solo_maps)
    fixed, fixed_rows = score_maps(records, fixed_maps)
    return {"solo":solo, "fixed25":fixed}, {"solo":solo_rows, "fixed25":fixed_rows}


def version_metadata(c):
    import importlib.metadata as metadata
    result = {"torch": str(torch.__version__), "segmentation_models_pytorch": metadata.version("segmentation_models_pytorch")}
    if c.architecture == "yolo11_edge_semantic":
        result["ultralytics"] = metadata.version("ultralytics")
    return result


def identity_of(c, report, baseline_context, physical_report=None):
    return {"family": EXPERIMENT_FAMILY, "config":asdict(c), "data_sha256":report["data_sha256"],
            "source_sha256":core.file_sha(__file__), "core_source_sha256":core.file_sha(core.__file__),
            "data_source_sha256":core.file_sha(kd.__file__), "palette":kd.PALETTE.tolist(),
            "baseline_context":baseline_context["identity"], "physical_data":physical_report,
            "versions":version_metadata(c)}


def configure_encoder(model, c, epoch):
    frozen = epoch <= c.freeze_encoder_epochs
    model.encoder.requires_grad_(not frozen)
    if frozen:
        model.encoder.eval()
    if c.freeze_encoder_bn:
        for module in model.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    return frozen


def train(cache, report, run_dir, c, baseline_context, num_workers=2, resume=True,
          patience=10, min_epochs=20, stop_after_epoch=None, physical_report=None):
    c.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("Colab GPU 런타임으로 변경하세요.")
    if type(num_workers) is not int or not 0 <= num_workers <= 4 or type(patience) is not int or patience < 0:
        raise ValueError("Invalid workers/patience")
    if type(min_epochs) is not int or not 1 <= min_epochs <= c.epochs:
        raise ValueError("MIN_EPOCHS must be in 1..EPOCHS")
    if stop_after_epoch is not None and (type(stop_after_epoch) is not int or not 1 <= stop_after_epoch <= c.epochs):
        raise ValueError("Invalid STOP_AFTER_EPOCH")
    if (c.architecture == "sem_ebsd_fusion") != (physical_report is not None):
        raise ValueError("Measured physical channels are required only for SEM+EBSD")
    random.seed(c.seed); np.random.seed(c.seed); torch.manual_seed(c.seed); torch.cuda.manual_seed_all(c.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    local_run = Path("/content/komap_metal_methods_training")/run_dir.name
    identity = identity_of(c, report, baseline_context, physical_report)
    identity_path, last_path = run_dir/"identity.json", run_dir/"last.pt"
    best_path, solo_path = run_dir/"best.pt", run_dir/"best_solo.pt"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("설정·코드·데이터·기존 조합이 바뀌었습니다. 새 RUN_NAME을 사용하세요.")
    if (best_path.exists() or solo_path.exists()) and not last_path.exists():
        raise ValueError("best exists without last.pt; export it or use a new RUN_NAME")
    if not resume and any(path.exists() for path in (last_path, best_path, solo_path)):
        raise ValueError("이 run에 저장된 학습이 있습니다. 새 RUN_NAME을 사용하세요.")
    state = torch.load(last_path, map_location="cpu", weights_only=False) if resume and last_path.exists() else None
    if state is not None and state["identity"] != identity:
        raise ValueError("Resume identity differs")
    core.atomic_json(identity_path, identity)
    core.atomic_json(run_dir/"data_manifest.json", report)
    core.atomic_json(run_dir/"baseline_validation_summary.json", baseline_context["validation"])
    core.write_csv(run_dir/"baseline_validation_per_image.csv", baseline_context["rows"])
    epoch_start = state["epoch"]+1 if state is not None else 1
    end_epoch = min(c.epochs, stop_after_epoch or c.epochs)
    if state is not None and (epoch_start > end_epoch or (patience and state["bad_checks"] >= patience and epoch_start > min_epochs)):
        print("저장된 학습 범위가 완료되어 best 내보내기로 진행합니다.")
        del state
        return best_path, solo_path
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = build_model(c, pretrained=state is None).to(device=device, memory_format=torch.channels_last)
    if state is not None:model.load_state_dict(state["model"], strict=True)

    model.eval()
    channels = 5 if c.architecture == "sem_ebsd_fusion" else 3
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
        smoke = model(torch.zeros((1, channels, c.patch, c.patch), device=device))
    if tuple(smoke.shape) != (1, 4, c.patch, c.patch) or not torch.isfinite(smoke).all():
        raise RuntimeError("Model output shape/values invalid; training was not started")
    del smoke
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    head_parameters = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([{"params":model.encoder.parameters(), "lr":c.encoder_lr},
                                  {"params":head_parameters, "lr":c.decoder_lr}], weight_decay=c.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(c.ema_decay), use_buffers=False) if c.use_ema else None
    history, global_step, bad_checks = [], 0, 0
    best_fixed, best_solo = -1.0, -1.0
    gpu_input = GPUInput(device, c)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        if ema is not None:ema.load_state_dict(state["ema"])
        history, global_step, bad_checks = state["history"], state["global_step"], state["bad_checks"]
        best_fixed, best_solo = state["best_fixed"], state["best_solo"]
        torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"이어 학습 epoch {epoch_start}; fixed25 best={best_fixed:.6f}; solo best={best_solo:.6f}")
    else:
        summaries, rows = evaluate(model, cache["valid"], c, gpu_input, amp_dtype, baseline_context)
        best_fixed, best_solo = summaries["fixed25"]["miou_"+c.absent_policy], summaries["solo"]["miou_"+c.absent_policy]
        payload = {"identity":identity, "config":asdict(c), "epoch":0, "model":model.state_dict(),
                   "validation":summaries, "uses_ema":False}
        core.save_checkpoint(best_path, {**payload,"selection":"fixed25","score":best_fixed}, local_run)
        core.save_checkpoint(solo_path, {**payload,"selection":"solo","score":best_solo}, local_run)
        core.atomic_json(run_dir/"initial_validation_summary.json",summaries)
        core.write_csv(run_dir/"best_fixed25_per_image.csv",rows["fixed25"])
        core.write_csv(run_dir/"best_solo_per_image.csv",rows["solo"])
        history.append({"epoch":0,"loss":None,"main_loss":None,"boundary_loss":None,"boundary_weight":0.0,
                        "fixed25_miou":best_fixed,"solo_miou":best_solo,"best_fixed25":best_fixed,"best_solo":best_solo,
                        "encoder_lr":0.0,"decoder_lr":0.0,"train_seconds":0.0,"valid_seconds":0.0,"peak_gpu_gib":0.0})
        print(f"새 모델 epoch0: fixed25={best_fixed:.6f}, solo={best_solo:.6f}; 기존 최종 조합은 별도로 보존합니다.")
    def save_last(epoch):
        core.save_checkpoint(last_path, {"identity":identity,"epoch":epoch,"model":model.state_dict(),
            "optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),"ema":ema.state_dict() if ema is not None else None,
            "history":history,"global_step":global_step,"bad_checks":bad_checks,"best_fixed":best_fixed,"best_solo":best_solo,
            "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()}, local_run)
    if state is None:save_last(0)
    saved_epoch = epoch_start-1
    del state
    loader, sampler = core.make_loader(cache, c, num_workers)
    core.atomic_json(run_dir/"sampling_summary.json", loader.dataset.zoom_sampling_report)
    props = torch.cuda.get_device_properties(0)
    print(f"실제 장치: {props.name}, {props.total_memory/2**30:.1f} GiB, AMP={amp_dtype}")
    print(f"모델={c.architecture}; patch={c.patch}; batch={c.batch}; crops={len(loader.dataset)}/epoch; updates={len(loader)}/epoch")
    print("best.pt: 기존 최종 조합75% + 새 모델25%; best_solo.pt: 새 모델 단독; 모두 원본 해상도 회전4 TTA")
    print("최종 후보에 기존 최종 조합100%를 포함합니다. 기존 결과가 더 좋으면 기존 마스크를 선택합니다.")
    evaluated_model = None
    for epoch in range(epoch_start, end_epoch+1):
        sampler.epoch = epoch
        model.train()
        frozen = configure_encoder(model, c, epoch)
        factor = epoch/c.warmup_epochs if epoch <= c.warmup_epochs else (
            0.02+0.98*(1+math.cos(math.pi*(epoch-c.warmup_epochs)/(c.epochs-c.warmup_epochs)))/2)
        optimizer.param_groups[0]["lr"] = 0.0 if frozen else c.encoder_lr*factor
        optimizer.param_groups[1]["lr"] = c.decoder_lr*factor
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        started = time.perf_counter()
        totals = torch.zeros(3, device=device)
        batches, actual_weight = 0, 0.0
        for step, (images, labels, _) in enumerate(loader, 1):
            x = gpu_input(images, c); y = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits, boundary = model.forward_tasks(x)
                loss, main, auxiliary, actual_weight = losses(logits, boundary, y, c, epoch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            scaler.step(optimizer); scaler.update()
            global_step += 1; batches += 1
            if ema is not None and (global_step % c.ema_every == 0 or step == len(loader)):
                ema.update_parameters(model)
            totals += torch.stack([loss.detach(), main, auxiliary])
            if step % 100 == 0:
                shown = float((totals[0]/batches).item())
                if not math.isfinite(shown):raise FloatingPointError("Non-finite training loss")
                print(f"  epoch {epoch}: {step}/{len(loader)}, mean loss={shown:.4f}", flush=True)
        torch.cuda.synchronize()
        means = (totals/max(1,batches)).cpu().tolist()
        if not all(math.isfinite(value) for value in means):raise FloatingPointError("Non-finite epoch loss")
        row = {"epoch":epoch,"loss":means[0],"main_loss":means[1],"boundary_loss":means[2],"boundary_weight":actual_weight,
               "fixed25_miou":None,"solo_miou":None,"best_fixed25":best_fixed,"best_solo":best_solo,
               "encoder_lr":optimizer.param_groups[0]["lr"],"decoder_lr":optimizer.param_groups[1]["lr"],
               "train_seconds":time.perf_counter()-started,"valid_seconds":0.0,
               "peak_gpu_gib":torch.cuda.max_memory_allocated()/2**30}
        do_eval = epoch == 1 or epoch % c.eval_every == 0 or epoch == end_epoch
        stopped = False
        if do_eval:
            evaluated_model = ema.module if ema is not None else model
            valid_start = time.perf_counter()
            summaries, rows = evaluate(evaluated_model, cache["valid"], c, gpu_input, amp_dtype, baseline_context)
            row["valid_seconds"] = time.perf_counter()-valid_start
            fixed, solo = summaries["fixed25"]["miou_"+c.absent_policy], summaries["solo"]["miou_"+c.absent_policy]
            if not all(math.isfinite(v) for v in (fixed,solo)):raise FloatingPointError("Invalid validation score")
            row.update(fixed25_miou=fixed, solo_miou=solo)
            payload = {"identity":identity,"config":asdict(c),"epoch":epoch,"model":evaluated_model.state_dict(),
                       "validation":summaries,"uses_ema":c.use_ema}
            improved_fixed, improved_solo = fixed > best_fixed, solo > best_solo
            if improved_fixed:
                best_fixed = fixed
                core.save_checkpoint(best_path,{**payload,"selection":"fixed25","score":fixed},local_run)
                core.write_csv(run_dir/"best_fixed25_per_image.csv", rows["fixed25"])
            if improved_solo:
                best_solo = solo
                core.save_checkpoint(solo_path,{**payload,"selection":"solo","score":solo},local_run)
                core.write_csv(run_dir/"best_solo_per_image.csv", rows["solo"])
            bad_checks = 0 if improved_fixed or improved_solo else bad_checks+1
            stopped = bool(patience and bad_checks >= patience and epoch >= min_epochs)
            row.update(best_fixed25=best_fixed, best_solo=best_solo)
            history.append(row)
            save_last(epoch)
            saved_epoch = epoch
            print(f"Epoch {epoch}/{c.epochs}: loss={means[0]:.4f}; fixed25={fixed:.6f}; solo={solo:.6f}; "
                  f"best25={best_fixed:.6f}; best_solo={best_solo:.6f}; train={row['train_seconds']:.1f}s; "
                  f"valid={row['valid_seconds']:.1f}s; GPU={row['peak_gpu_gib']:.2f}GiB",flush=True)
        else:
            history.append(row)
            print(f"Epoch {epoch}/{c.epochs}: loss={means[0]:.4f}; train={row['train_seconds']:.1f}s",flush=True)
        core.write_csv(run_dir/"history.csv", history)
        core.atomic_json(run_dir/"training_status.json", {"epoch":epoch,"best_fixed25":best_fixed,
            "best_solo":best_solo,"bad_validation_checks":bad_checks,"early_stopped":stopped,
            "last_checkpoint_epoch":saved_epoch})
        if stopped:
            print("조기 종료: 고정 결합·단독 점수 모두 개선 없음. best 비교·자동 내보내기로 진행합니다.")
            break
    evaluated_model = None
    payload = None
    del loader, model, optimizer, ema, x, y, logits, boundary, loss, head_parameters
    gc.collect(); torch.cuda.empty_cache()
    return best_path, solo_path


def attach_ebsd_ram(cache, ebsd_root):

    ebsd_root = Path(ebsd_root)
    metadata_path = ebsd_root/"ebsd_manifest.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"실제 정렬된 EBSD 입력이 필요합니다: {metadata_path}. SEM PNG로 KAM을 생성하지 않습니다.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("registered_to_sem") is not True or metadata.get("measured_ebsd") is not True:
        raise ValueError("Manifest must explicitly identify measured EBSD aligned to the SEM images")
    row_map = {(r["split"], r["image"]):r for r in metadata.get("files",[])}
    expected = {(split,rec["name"]) for split,records in cache.items() for rec in records}
    if len(row_map) != len(metadata.get("files",[])) or set(row_map) != expected:
        raise ValueError("EBSD manifest must cover the same train70/valid20/test10 image names")
    arrays, fingerprints = {}, []
    total, total2, count = np.zeros(2,np.float64), np.zeros(2,np.float64), 0
    for split, records in cache.items():
        arrays[split] = []
        for rec in tqdm(records, desc=split+": 측정 KAM/PQ → RAM"):
            row = row_map[(split,rec["name"])]
            if row.get("image_sha256") != rec["image_sha256"]:
                raise ValueError(f"EBSD is associated with a different SEM image: {rec['name']}")
            path = pf.contained_path(ebsd_root,row["file"])
            with np.load(path,allow_pickle=False) as archive:
                physical = np.stack([archive["kam"],archive["pattern_quality"]],axis=-1).astype(np.float32)
            if physical.shape != (*rec["image"].shape[:2],2) or not np.isfinite(physical).all():
                raise ValueError(f"Registered EBSD shape/values invalid: {path}")
            arrays[split].append(physical)
            fingerprints.append({"split":split,"image":rec["name"],"image_sha256":rec["image_sha256"],
                                 "file":row["file"],"sha256":core.file_sha(path),"shape":list(physical.shape)})
            if split == "train":
                flat = physical.reshape(-1,2).astype(np.float64)
                total += flat.sum(axis=0); total2 += np.square(flat).sum(axis=0); count += flat.shape[0]
    mean = total/count
    std = np.sqrt(np.maximum(total2/count-mean**2,1e-12))
    if np.any(std <= 1e-6):raise ValueError("Measured EBSD channels have no train variation")
    attached = {}
    for split, records in cache.items():
        attached[split] = []
        for rec, physical in zip(records,arrays[split]):
            z = np.clip((physical-mean.astype(np.float32))/std.astype(np.float32),-5,5)
            image = np.concatenate([rec["image"].astype(np.float32),z.astype(np.float32)],axis=-1)
            image.setflags(write=False)
            attached[split].append({**rec,"image":image})
    report = {"files":fingerprints,"manifest_sha256":core.file_sha(metadata_path),
              "normalization":"mean/std from train70 only; clipped to +-5 std", "train_mean":mean.tolist(),
              "train_std":std.tolist(),"channels":["kam","pattern_quality"],
              "alignment_note":"Geometry declared by input producer; same shape and SEM SHA verified, physical registration cannot be inferred automatically."}
    report["physical_data_sha256"] = hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()
    print("측정 EBSD RAM 캐시 완료; 정규화 통계는 train70에서만 계산했습니다.")
    return attached, report


def blend_maps(old_maps, new_maps, beta):
    if beta not in MIX_WEIGHTS:
        raise ValueError("Only the prespecified mixing weights are permitted")
    if old_maps.keys() != new_maps.keys():
        raise ValueError("Different native image keys")
    return {name:(1-beta)*old_maps[name]+beta*new_maps[name] for name in old_maps}


def ranking_row(label, checkpoint_key, beta, summary, reference, selected=False):
    tiny = summary["eu_component_diagnostics"]["1_16"]
    return {"candidate":label,"checkpoint_key":checkpoint_key,"new_model_weight":beta,"selected":int(selected),
            "miou_ignore":summary["miou_ignore"],"miou_zero":summary["miou_zero"],
            "delta_vs_existing":summary["miou_ignore"]-reference["miou_ignore"],
            **{slug+"_iou":value for slug,value in summary["class_iou_ignore"].items()},
            "Eu_1_16_pixel_recall":tiny["pixel_recall"],
            "Eu_1_16_completely_missed_fraction":tiny["completely_missed_fraction"]}


def compare_native_candidates(records, baseline_context, candidate_maps, absent_policy):
    old = baseline_context["probabilities"]["valid"]
    reference, base_rows = score_maps(records,old)
    summaries, rows, definitions = {"existing_baseline":reference}, {"existing_baseline":base_rows}, {}
    chosen = "existing_baseline"
    best_score = reference["miou_"+absent_policy]
    for key, maps in candidate_maps.items():
        for beta in MIX_WEIGHTS[1:]:
            label = f"{key}@{beta:g}"
            summaries[label], rows[label] = score_maps(records,blend_maps(old,maps,beta))
            definitions[label] = {"checkpoint_key":key,"beta":beta}
            score = summaries[label]["miou_"+absent_policy]
            if score > best_score+1e-8:
                chosen,best_score = label,score
    ranking = [ranking_row("existing_baseline","existing_baseline",0.0,reference,reference,chosen=="existing_baseline")]
    ranking += [ranking_row(label,defs["checkpoint_key"],defs["beta"],summaries[label],reference,label==chosen)
                for label,defs in definitions.items()]
    ranking.sort(key=lambda row:row["miou_"+absent_policy], reverse=True)
    return chosen,summaries,rows,definitions,ranking


def export_candidates(cache, report, run_dir, baseline_context, physical_report=None):
    if not torch.cuda.is_available():raise RuntimeError("GPU required for native TTA export")
    run_dir = Path(run_dir)
    paths = {"best_fixed25":run_dir/"best.pt", "best_solo":run_dir/"best_solo.pt"}
    states = {key:torch.load(path,map_location="cpu",weights_only=False) for key,path in paths.items()}
    c = Config(**states["best_fixed25"]["config"])
    c.validate()
    expected_identity = identity_of(c,report,baseline_context,physical_report)
    for state in states.values():
        if state["identity"] != expected_identity:raise ValueError("Export code/settings/data/baseline differ from training")
    checkpoint_shas = {key:core.file_sha(path) for key,path in paths.items()}
    export_identity = {"experiment_identity":expected_identity,"checkpoints":checkpoint_shas,
                       "mix_weights":list(MIX_WEIGHTS),"tta_rotations":4,"output_resolution":"original"}
    digest = hashlib.sha256(json.dumps(export_identity,sort_keys=True).encode()).hexdigest()[:12]
    destination = run_dir/"exports"/("native_tta4_"+digest)
    manifest_path = destination/"manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("methods_export_identity") != export_identity:raise ValueError("Export identity differs")
        core.verify_scale_export(destination,manifest,cache)
        core.atomic_json(run_dir/"export_index.json",{"manifest":manifest_path.relative_to(run_dir).as_posix()})
        report_export(destination)
        return destination
    if destination.exists():raise ValueError("Incomplete unexpected export directory; use a new RUN_NAME")
    stage = Path("/content/komap_metal_methods_exports")/run_dir.name/destination.name
    if stage.exists():shutil.rmtree(stage)
    stage.mkdir(parents=True,exist_ok=True)
    model = None
    maps_by_key = {}
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    gpu_input = GPUInput(torch.device("cuda"),c)
    try:
        for key,state in states.items():

            earlier = next((other for other in maps_by_key if states[other]["epoch"]==state["epoch"]
                            and states[other]["uses_ema"]==state["uses_ema"]), None)
            if earlier is not None:
                maps_by_key[key] = maps_by_key[earlier]
                continue
            model = build_model(c,pretrained=False).to("cuda",memory_format=torch.channels_last)
            model.load_state_dict(state["model"],strict=True); model.eval()
            maps_by_key[key] = {rec["name"]:core.checked_probability(
                core.predict_probability(model,rec["image"],c,gpu_input,amp_dtype,4),rec["image"].shape[:2])
                for rec in tqdm(cache["valid"],desc=key+": native 회전4")}
            del model; model=None; gc.collect(); torch.cuda.empty_cache()
        chosen,summaries,rows,definitions,ranking = compare_native_candidates(
            cache["valid"],baseline_context,maps_by_key,c.absent_policy)
        if chosen == "existing_baseline":
            chosen_key,beta = "best_solo",0.0
        else:
            chosen_key,beta = definitions[chosen]["checkpoint_key"],definitions[chosen]["beta"]
        state = states[chosen_key]
        model = build_model(c,pretrained=False).to("cuda",memory_format=torch.channels_last)
        model.load_state_dict(state["model"],strict=True); model.eval()
        files = []
        for split in ("valid","test"):
            for rec in tqdm(cache[split],desc=split+": 확률·최종 결합 저장"):
                p = maps_by_key[chosen_key][rec["name"]] if split=="valid" else core.predict_probability(
                    model,rec["image"],c,gpu_input,amp_dtype,4)
                old = baseline_context["probabilities"][split][rec["name"]]
                files.append(core.write_probability_pair(stage,split,rec,p,old,beta))
            core.zip_final_masks(stage,split)
        core.write_csv(stage/"candidate_ranking.csv",ranking)
        comparisons = [ranking_row("existing_baseline","existing_baseline",0.0,summaries["existing_baseline"],summaries["existing_baseline"])]
        for key in paths:
            comparisons.append(ranking_row(key+"_solo",key,1.0,summaries[key+"@1"],summaries["existing_baseline"]))
        comparisons.append(next(row for row in ranking if row["selected"]))
        core.write_csv(stage/"ablation_comparison.csv",comparisons)
        core.write_csv(stage/"validation_selected_per_image.csv",rows[chosen])
        core.write_csv(stage/"validation_candidates_per_image.csv",[{"candidate":key,**row} for key,values in rows.items() for row in values])
        result = {"selected_candidate":chosen,"new_model_weight":beta,"existing_baseline_weight":1-beta,
                  "selected":summaries[chosen],"existing_baseline":summaries["existing_baseline"],
                  "candidates":summaries,"raw_export_checkpoint_key":chosen_key,"tta_rotations":4,
                  "absent_policy":c.absent_policy,"checkpoint_epoch":state["epoch"],
                  "note":"Selected on valid20; no independent test score. beta=0 preserves the previous final result."}
        core.atomic_json(stage/"validation_summary.json",result)
        core.atomic_json(stage/"export_settings.json",export_identity)
        with zipfile.ZipFile(stage/"analysis_bundle.zip","w",zipfile.ZIP_DEFLATED) as z:
            for name in ("candidate_ranking.csv","ablation_comparison.csv","validation_selected_per_image.csv",
                         "validation_candidates_per_image.csv","validation_summary.json","export_settings.json"):
                z.write(stage/name,arcname=name)
            for path in sorted((stage/"selected_valid_masks").glob("*.png")):
                z.write(path,arcname="selected_valid_masks/"+path.name)
        artifacts = [{"path":path.relative_to(stage).as_posix(),"sha256":core.file_sha(path)}
                     for path in sorted(stage.rglob("*")) if path.is_file()]


        manifest = {"complete":True,"format_version":2,"model_family":PROBABILITY_FAMILY,"target_class":-1,
                    "architecture":c.architecture,"experiment_family":EXPERIMENT_FAMILY,"config":asdict(c),
                    "class_names":list(kd.CLASS_NAMES),"palette":kd.PALETTE.tolist(),"dtype":"float32",
                    "checkpoint_epoch":state["epoch"],"checkpoint_sha256":checkpoint_shas[chosen_key],
                    "identity":state["identity"],"tta_rotations":4,"calibration":"none",
                    "probability":"raw new-model four-class softmax, native dimensions; final mixture stored separately",
                    "selected_candidate":chosen,"methods_export_identity":export_identity,
                    "final_blend":{"new_model_weight":beta,"baseline_identity":baseline_context["identity"]},
                    "files":files,"artifacts":artifacts}
        core.atomic_json(stage/"manifest.json",manifest)
        destination.parent.mkdir(parents=True,exist_ok=True)
        pending = destination.with_name(destination.name+".pending")
        if pending.exists():shutil.rmtree(pending)
        shutil.copytree(stage,pending)
        core.verify_scale_export(pending,manifest,cache)
        os.replace(pending,destination)
        core.atomic_json(run_dir/"export_index.json",{"manifest":manifest_path.relative_to(run_dir).as_posix()})
    finally:
        del model,states,maps_by_key
        gc.collect(); torch.cuda.empty_cache()
    report_export(destination)
    return destination


def report_export(directory):
    directory = Path(directory)
    result = json.loads((directory/"validation_summary.json").read_text(encoding="utf-8"))
    policy = result["absent_policy"]
    score = result["selected"]["miou_"+policy]
    old = result["existing_baseline"]["miou_"+policy]
    print(f"선택: {result['selected_candidate']}; 최종 valid mIoU={score:.8f}; 기존 대비={score-old:+.8f}")
    print("Drive 확률·결과:",directory)
    print("후보 순위:",directory/"candidate_ranking.csv")
    print("최종 test10 결합 마스크:",directory/"selected_test_masks.zip")
    print("후속 분석에 전달할 파일:",directory/"analysis_bundle.zip")
