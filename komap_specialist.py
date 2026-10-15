'상별 GroupNorm U-Net 학습과 확률 추론.'
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


import sympy
import sympy.core
if not hasattr(sympy, "core"):
    raise RuntimeError("SymPy import is incomplete. Restart the Colab session before running again.")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler
import komap_specialist_data as kd

MODEL_FAMILY = "komap_unet_binary_specialist_v1"
SELECTION_METRIC = "binary_miou_zero"


@dataclass
class Config:
    target_class: int
    epochs: int = 90
    patch_size: int = 384
    stride: int = 256
    batch_size: int = 4
    samples_per_image: int = 8
    base_channels: int = 24
    lr: float = 1e-3
    seed: int = 42
    eval_every: int = 5
    gamma_probability: float = 1.0
    gamma_min: float = 0.80
    gamma_max: float = 1.60

    def validate(self):
        if type(self.target_class) is not int or not 0 <= self.target_class < 4:
            raise ValueError("target_class must be an integer from 0 to 3")
        for key in ("epochs", "patch_size", "stride", "batch_size", "samples_per_image", "base_channels", "eval_every"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.patch_size < 32 or self.stride > self.patch_size:
            raise ValueError("Require patch_size >= 32 and stride <= patch_size")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not all(math.isfinite(v) for v in (self.lr, self.gamma_probability, self.gamma_min, self.gamma_max)):
            raise ValueError("Learning rate and gamma settings must be finite")
        if self.lr <= 0 or not 0 <= self.gamma_probability <= 1 or not 0 < self.gamma_min <= self.gamma_max:
            raise ValueError("Invalid learning rate or gamma range")


@dataclass
class RuntimeOptions:
    num_workers: int = 2
    prefetch_factor: int = 2
    progress_every: int = 100

    def validate(self):
        if type(self.num_workers) is not int or not 0 <= self.num_workers <= 8:
            raise ValueError("num_workers must be an integer from 0 to 8")
        if type(self.prefetch_factor) is not int or not 1 <= self.prefetch_factor <= 4:
            raise ValueError("prefetch_factor must be an integer from 1 to 4")
        if type(self.progress_every) is not int or self.progress_every < 1:
            raise ValueError("progress_every must be a positive integer")


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        groups = math.gcd(8, out_channels)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True),
        )

class UNet(nn.Module):
    def __init__(self, base_channels=24, num_classes=4):
        super().__init__()
        widths = [base_channels * 2 ** i for i in range(5)]
        self.encoders = nn.ModuleList([
            ConvBlock(3 if i == 0 else widths[i - 1], width)
            for i, width in enumerate(widths)
        ])
        self.decoders = nn.ModuleList([
            ConvBlock(widths[i + 1] + widths[i], widths[i])
            for i in reversed(range(4))
        ])
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.encoders):
            if i:
                x = F.max_pool2d(x, 2)
            x = block(x)
            skips.append(x)
        for block, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def append_performance(path, row):
    path = Path(path)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def source_hash():
    hashes = [sha256_file(Path(__file__)), sha256_file(Path(kd.__file__))]
    return hashlib.sha256("\n".join(hashes).encode()).hexdigest()


def data_manifest(data_root):
    root = Path(data_root).resolve()
    rows = []
    for split in ("train", "valid"):
        for image, mask in kd.discover_split(root, split):
            for path in (image, mask):
                rows.append({"split": split, "path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"sha256": fingerprint, "files": rows}


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("model_family") != MODEL_FAMILY:
        raise ValueError("This is not a binary specialist checkpoint from this notebook")
    if state.get("class_names") != list(kd.CLASS_NAMES) or state.get("palette") != kd.PALETTE.tolist():
        raise ValueError("Checkpoint class order/palette mismatch")
    Config(**state["config"]).validate()
    return state


def prepare(project_root, data_root, config):
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU Colab runtime; CPU training is not started automatically")
    kd.extract_archives(Path(project_root) / "data", data_root)
    report = kd.validate_data(data_root, strict_counts=True)
    counts = report["splits"]["train"]["pixel_counts"]
    if counts[config.target_class] == 0 or counts[config.target_class] == sum(counts):
        raise ValueError("Training needs both positive and negative pixels for this class")
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    print(f"Target: {config.target_class} / {kd.CLASS_NAMES[config.target_class]}")
    print(f"train70 / valid20 / test10 verified; views/epoch={70 * config.samples_per_image * 4}")
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
    return report


class CropDataset(Dataset):

    def __init__(self, pairs, config):
        self.config = config
        self.augmentation_rng = random.Random(config.seed + 104729)
        self.cache = []
        for ip, mp in pairs:
            image, mask = kd.read_image(ip), kd.read_mask(mp)
            if image.shape[:2] != mask.shape:
                raise ValueError(f"Image/mask size mismatch: {ip.name}")
            p = config.patch_size
            bottom, right = max(0, p - mask.shape[0]), max(0, p - mask.shape[1])
            image = np.pad(image, ((0, bottom), (0, right), (0, 0)), mode="edge")
            mask = np.pad(mask, ((0, bottom), (0, right)), mode="edge")
            self.cache.append((image, mask))

    def prepare_epoch(self, epoch):
        rng = random.Random(self.config.seed + 1000003 * epoch + 200003)
        p = self.config.patch_size
        self.crop_plan = [[(rng.randrange(mask.shape[0] - p + 1), rng.randrange(mask.shape[1] - p + 1))
                           for _ in range(self.config.samples_per_image)] for _, mask in self.cache]
        self.rotation_draws = np.zeros((len(self.cache), 4), dtype=np.int64)
        self.positive_pixels = 0
        self.gamma_draws = 0

    def __len__(self):
        return len(self.cache) * self.config.samples_per_image * 4

    def __getitem__(self, key):
        c = self.config
        image_index, y, x, k, contrast, brightness, use_gamma, gamma = key
        image, mask = self.cache[image_index]
        image = np.rot90(image[y:y+c.patch_size, x:x+c.patch_size], k).astype(np.float32) / 255.0
        mask = np.rot90(mask[y:y+c.patch_size, x:x+c.patch_size], k)
        image = np.clip((image - .5) * contrast + .5 + brightness, 0, 1)
        if use_gamma:
            image = np.power(image, np.float32(gamma)).astype(np.float32)
        target = mask == c.target_class
        metadata = np.array([image_index, k, int(target.sum()), int(use_gamma)], dtype=np.int64)
        return (torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)), metadata)

    def record_batch(self, metadata):

        values = metadata.numpy()
        np.add.at(self.rotation_draws, (values[:, 0], values[:, 1]), 1)
        self.positive_pixels += int(values[:, 2].sum())
        self.gamma_draws += int(values[:, 3].sum())


class EpochViewSampler(Sampler):

    def __init__(self, dataset, generator):
        self.dataset = dataset
        self.order = RandomSampler(dataset, generator=generator)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        dataset = self.dataset
        c, rng = dataset.config, dataset.augmentation_rng
        for index in self.order:
            image_index = index // (c.samples_per_image * 4)
            crop_index, k = (index // 4) % c.samples_per_image, index % 4
            y, x = dataset.crop_plan[image_index][crop_index]
            contrast, brightness = rng.uniform(.9, 1.1), rng.uniform(-.05, .05)
            use_gamma = c.gamma_probability > 0 and rng.random() < c.gamma_probability
            gamma = rng.uniform(c.gamma_min, c.gamma_max) if use_gamma else 1.0
            yield image_index, y, x, k, contrast, brightness, use_gamma, gamma


def binary_loss(logits, target):
    logits = logits[:, 0].float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(1, 2))
    denominator = probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

    dice = 1 - ((2 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice


def tile_starts(length, patch, stride):
    last = max(0, length - patch)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.inference_mode()
def predict_probability(model, image, config, device):
    'Native overlapping tiles -> average sigmoid probabilities -> H x W float32.'
    h, w = image.shape[:2]
    p = config.patch_size
    image = np.pad(image, ((0, max(0, p-h)), (0, max(0, p-w)), (0, 0)), mode="edge")
    hh, ww = image.shape[:2]
    coordinates = [(y, x) for y in tile_starts(hh, p, config.stride) for x in tile_starts(ww, p, config.stride)]
    total, counts = np.zeros((hh, ww), np.float32), np.zeros((hh, ww), np.float32)
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(coordinates), config.batch_size):
            locations = coordinates[start:start+config.batch_size]
            crops = np.stack([image[y:y+p, x:x+p] for y, x in locations]).astype(np.float32) / 255.0
            batch = torch.from_numpy(crops.transpose(0, 3, 1, 2).copy()).to(device)
            with torch.amp.autocast("cuda"):
                probability = model(batch)[:, 0].float().sigmoid().cpu().numpy()
            for array, (y, x) in zip(probability, locations):
                total[y:y+p, x:x+p] += array
                counts[y:y+p, x:x+p] += 1
        if np.any(counts == 0):
            raise RuntimeError("Tiled inference left uncovered pixels")
        result = (total / counts)[:h, :w].copy()
        if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
            raise FloatingPointError("Invalid probability map")
        return result
    finally:
        model.train(was_training)


def evaluate_binary(model, pairs, config, device):
    rows = []
    for ip, mp in pairs:
        image, mask = kd.read_image(ip), kd.read_mask(mp)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask dimensions differ: {ip.name}")
        gt = mask == config.target_class
        pred = predict_probability(model, image, config, device) >= .5
        tp, fp, fn, tn = [int(x.sum()) for x in (gt & pred, ~gt & pred, gt & ~pred, ~gt & ~pred)]
        fg_union, bg_union = tp + fp + fn, tn + fp + fn
        fg, bg = tp / fg_union if fg_union else 0.0, tn / bg_union if bg_union else 0.0
        rows.append({"image": ip.name, "target_pixels": int(gt.sum()), "tp": tp, "fp": fp, "fn": fn,
                     "foreground_iou_zero": fg, "foreground_union": fg_union,
                     "binary_miou_zero": (fg + bg) / 2.0})
    summary = {
        "target_class": config.target_class, "class_name": kd.CLASS_NAMES[config.target_class],
        "images": len(rows), "threshold": .5, "selection_metric": SELECTION_METRIC,
        "binary_miou_zero": float(np.mean([r[SELECTION_METRIC] for r in rows])),
        "foreground_iou_zero": float(np.mean([r["foreground_iou_zero"] for r in rows])),
        "negative_images": sum(r["target_pixels"] == 0 for r in rows),
        "negative_image_fp_pixels": sum(r["fp"] for r in rows if r["target_pixels"] == 0),
        "note": "Checkpoint selection is the per-image two-class mIoU with empty unions=0; this is NOT the four-class competition score.",
    }
    return summary, rows


def train(data_root, output_dir, config, runtime=None):
    'Fresh fast run; same-source resume replays an unfinished epoch.'
    runtime = runtime or RuntimeOptions()
    config.validate()
    runtime.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    output_dir = Path(output_dir)
    last, best = output_dir / "last.pt", output_dir / "best.pt"
    fingerprint = data_manifest(data_root)
    identity = {"config": asdict(config), "data_sha256": fingerprint["sha256"], "source_sha256": source_hash()}
    checkpoint = load_checkpoint(last) if last.is_file() else None
    if checkpoint is not None and checkpoint["identity"] != identity:
        raise ValueError("Resume config/data/code differs. Choose a NEW RUN_NAME; old results are preserved.")
    if checkpoint is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Run exists without last.pt. Choose a NEW RUN_NAME; do not delete old results.")
    if checkpoint is not None and not best.is_file():
        raise FileNotFoundError("best.pt is missing from this run; restore it before resuming")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda")
    print("Loading the original images into RAM once...", flush=True)
    dataset = CropDataset(kd.discover_split(data_root, "train"), config)
    valid_pairs = kd.discover_split(data_root, "valid")


    generator = torch.Generator().manual_seed(config.seed)
    worker_generator = torch.Generator().manual_seed(config.seed + 65537)
    loader_options = {"batch_size": config.batch_size,
                      "sampler": EpochViewSampler(dataset, generator),
                      "num_workers": runtime.num_workers, "pin_memory": True,
                      "generator": worker_generator, "drop_last": False}
    if runtime.num_workers > 0:
        loader_options.update(prefetch_factor=runtime.prefetch_factor, persistent_workers=True,
                              multiprocessing_context="spawn")
    loader = DataLoader(dataset, **loader_options)
    model = UNet(config.base_channels, num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda")
    history, start_epoch, best_score = [], 0, -1.0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        history, start_epoch, best_score = checkpoint["history"], checkpoint["epoch"], checkpoint["best_score"]
        generator.set_state(checkpoint["loader_rng"])
        worker_generator.set_state(checkpoint["worker_rng"])
        dataset.augmentation_rng.setstate(checkpoint["augmentation_rng"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        saved_best = load_checkpoint(best)
        if saved_best["identity"] != identity:
            raise ValueError("best.pt belongs to another run")
        best_score = max(best_score, saved_best["validation"][SELECTION_METRIC])
        del saved_best, checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "config.json", asdict(config))
    atomic_json(output_dir / "runtime.json", asdict(runtime))
    atomic_json(output_dir / "data_manifest.json", fingerprint)
    atomic_json(output_dir / "identity.json", identity)

    (output_dir / "sources").mkdir(exist_ok=True)
    for path in (Path(__file__), Path(kd.__file__)):
        (output_dir / "sources" / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"{kd.CLASS_NAMES[config.target_class]} | start={start_epoch}/{config.epochs} | parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"GPU={torch.cuda.get_device_name(0)}; batch={config.batch_size}; workers={runtime.num_workers}; {len(dataset)} views / {len(loader)} updates per epoch", flush=True)
    cache_gib = sum(image.nbytes + mask.nbytes for image, mask in dataset.cache) / 2**30
    print(f"Original image cache={cache_gib:.2f} GiB per process. First epoch includes worker startup; later epochs reuse workers.", flush=True)
    try:
        for epoch in range(start_epoch+1, config.epochs+1):
            torch.cuda.synchronize(device)
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            dataset.prepare_epoch(epoch)
            model.train()
            loss_sum = torch.zeros((), device=device, dtype=torch.float32)
            count, step, data_wait_seconds = 0, 0, 0.0
            used_lr = float(optimizer.param_groups[0]["lr"])
            iterator_started = time.monotonic()
            iterator = iter(loader)
            data_wait_seconds += time.monotonic() - iterator_started
            while True:
                wait_started = time.monotonic()
                try:
                    images, targets, metadata = next(iterator)
                except StopIteration:
                    data_wait_seconds += time.monotonic() - wait_started
                    break
                data_wait_seconds += time.monotonic() - wait_started
                step += 1
                dataset.record_batch(metadata)
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                if step == 1:
                    model_device = next(model.parameters()).device
                    if model_device.type != "cuda" or images.device.type != "cuda" or targets.device.type != "cuda":
                        raise RuntimeError("Model and both training tensors must be on CUDA")
                    if epoch == start_epoch + 1:
                        print(f"Actual training devices: model={model_device}, images={images.device}, targets={targets.device}; batch shape={tuple(images.shape)}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    loss = binary_loss(model(images), targets)


                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss; last.pt preserves the last completed epoch")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.detach() * images.shape[0]
                count += images.shape[0]
                if step % runtime.progress_every == 0 or step == len(loader):
                    running_loss = float(loss_sum) / count
                    elapsed = time.monotonic() - started
                    print(f"Epoch {epoch}/{config.epochs} step {step}/{len(loader)} | loss={running_loss:.4f} | {count/max(elapsed,1e-9):.2f} views/s", flush=True)


            del iterator
            torch.cuda.synchronize(device)
            train_seconds = time.monotonic() - started
            epoch_loss = float(loss_sum) / count
            if count != len(dataset) or step != len(loader) or not np.all(dataset.rotation_draws == config.samples_per_image):
                raise RuntimeError("Incomplete four-rotation coverage")
            scheduler.step()
            summary, rows, improved = None, None, False
            valid_started = time.monotonic()
            if epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs:
                print("Validating 20 full-size images...", flush=True)
                summary, rows = evaluate_binary(model, valid_pairs, config, device)
                improved = summary[SELECTION_METRIC] > best_score
                if improved:
                    best_score = summary[SELECTION_METRIC]
            torch.cuda.synchronize(device)
            validation_seconds = time.monotonic() - valid_started
            peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
            history.append({"epoch": epoch, "loss": epoch_loss, "valid_binary_miou_zero": summary[SELECTION_METRIC] if summary else None,
                            "valid_foreground_iou_zero": summary["foreground_iou_zero"] if summary else None,
                            "negative_image_fp_pixels": summary["negative_image_fp_pixels"] if summary else None,
                            "best_binary_miou_zero": best_score, "lr": used_lr, "views": count, "steps": step,
                            "batch_size": config.batch_size,
                            "positive_pixel_fraction": dataset.positive_pixels/(count * config.patch_size**2),
                            "gamma_fraction": dataset.gamma_draws/count,
                            "train_seconds": train_seconds, "loader_wait_seconds": data_wait_seconds,
                            "validation_seconds": validation_seconds, "peak_cuda_allocated_gib": peak_gib,
                            "seconds": time.monotonic()-started})
            state = {"model_family": MODEL_FAMILY, "class_names": list(kd.CLASS_NAMES), "palette": kd.PALETTE.tolist(),
                     "config": asdict(config), "runtime": asdict(runtime), "identity": identity,
                     "epoch": epoch, "model": model.state_dict(),
                     "validation": summary, "torch_version": str(torch.__version__)}
            save_started = time.monotonic()
            if improved:
                atomic_checkpoint(best, state)
                atomic_json(output_dir / "best_validation.json", summary)
                write_csv(output_dir / "best_validation_per_image.csv", rows)
            state.update({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                          "best_score": best_score, "history": history, "loader_rng": generator.get_state(),
                          "worker_rng": worker_generator.get_state(),
                          "augmentation_rng": dataset.augmentation_rng.getstate(), "python_rng": random.getstate(),
                          "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()})
            atomic_checkpoint(last, state)
            write_csv(output_dir / "history.csv", history)
            save_seconds = time.monotonic() - save_started
            append_performance(output_dir / "performance.csv", {
                "epoch": epoch, "train_seconds": train_seconds,
                "loader_wait_seconds": data_wait_seconds, "validation_seconds": validation_seconds,
                "save_seconds": save_seconds, "total_seconds": time.monotonic()-started,
                "train_views_per_second": count/max(train_seconds,1e-9),
                "peak_cuda_allocated_gib": peak_gib, "batch_size": config.batch_size,
                "num_workers": runtime.num_workers, "prefetch_factor": runtime.prefetch_factor,
            })
            score = f"; valid binary mIoU(zero)={summary[SELECTION_METRIC]:.4f}" if summary else ""
            print(f"Epoch {epoch}/{config.epochs}: loss={epoch_loss:.4f}{score}; best={best_score:.4f}; lr={used_lr:.6g}", flush=True)
            print(f"Timing: train={train_seconds:.1f}s (host loader wait={data_wait_seconds:.1f}s), valid={validation_seconds:.1f}s, save={save_seconds:.1f}s | peak allocated CUDA={peak_gib:.2f} GiB", flush=True)
    except KeyboardInterrupt:
        print("Interrupted. Completed-epoch last.pt and validation best.pt are preserved. Run the export cell below.", flush=True)
    finally:
        if "iterator" in locals():
            del iterator
        del model, optimizer, scheduler, scaler, loader, loader_options, dataset
        gc.collect()
        torch.cuda.empty_cache()
    if not best.is_file():
        raise FileNotFoundError("No best.pt yet; finish at least the first epoch and validation, then export")
    return best


def export_best(run_dir, data_root):
    'Export valid20 and test10 from best.pt. Can run after a manual interruption.'
    run_dir, checkpoint_path = Path(run_dir), Path(run_dir) / "best.pt"
    state = load_checkpoint(checkpoint_path)
    config = Config(**state["config"])
    if state["identity"]["source_sha256"] != source_hash():
        raise ValueError("Export must use the same notebook code as training")
    if state["identity"]["data_sha256"] != data_manifest(data_root)["sha256"]:
        raise ValueError("Training/validation data differs from this checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for probability export")
    checkpoint_sha = sha256_file(checkpoint_path)
    export_dir = run_dir / "exports" / f"best_e{state['epoch']:03d}_{checkpoint_sha[:12]}"
    manifest_path = export_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("checkpoint_sha256") != checkpoint_sha or not old.get("complete"):
            raise ValueError("Conflicting export manifest; preserve it and inspect the run")
        current = {(split, ip.name): (ip, mp) for split in ("valid", "test")
                   for ip, mp in kd.discover_split(data_root, split)}
        if set(current) != {(r["split"], r["image"]) for r in old["files"]}:
            raise ValueError("Current images differ from the existing export")
        for row in old["files"]:
            ip, mp = current[row["split"], row["image"]]
            if sha256_file(ip) != row["image_sha256"] or (sha256_file(mp) if mp is not None else None) != row["mask_sha256"]:
                raise ValueError("Current image/label bytes differ from the existing export")
            if sha256_file(export_dir / row["probability_file"]) != row["probability_sha256"]:
                raise ValueError("A saved probability map changed; inspect the export")
        atomic_json(run_dir / "export_index.json", {"manifest": manifest_path.relative_to(run_dir).as_posix()})
        print(f"Existing complete export: {export_dir}")
        return export_dir
    model = UNet(config.base_channels, num_classes=1).to("cuda")
    model.load_state_dict(state["model"], strict=True)
    files = []
    try:
        for split in ("valid", "test"):
            pairs = kd.discover_split(data_root, split)
            if len(pairs) != kd.EXPECTED_COUNTS[split]:
                raise ValueError(f"Wrong image count for {split}")
            (export_dir / split).mkdir(parents=True, exist_ok=True)
            for ip, mp in pairs:
                image = kd.read_image(ip)
                probability = predict_probability(model, image, config, torch.device("cuda"))
                destination = export_dir / split / (ip.stem + ".npz")
                temporary = destination.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, probability=probability, target_class=np.int16(config.target_class))
                os.replace(temporary, destination)
                files.append({"split": split, "image": ip.name, "image_sha256": sha256_file(ip),
                              "mask_sha256": sha256_file(mp) if mp is not None else None,
                              "shape": list(probability.shape), "probability_file": destination.relative_to(export_dir).as_posix(),
                              "probability_sha256": sha256_file(destination)})
                print(f"Exported {split}/{ip.name}", flush=True)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    manifest = {"complete": True, "format_version": 1, "model_family": MODEL_FAMILY,
                "target_class": config.target_class, "class_names": list(kd.CLASS_NAMES), "palette": kd.PALETTE.tolist(),
                "config": asdict(config), "checkpoint_epoch": state["epoch"], "checkpoint_sha256": checkpoint_sha,
                "identity": state["identity"], "dtype": "float32", "probability": "mean of overlapping tile sigmoid probabilities",
                "calibration": "none", "files": files}
    atomic_json(manifest_path, manifest)
    atomic_json(run_dir / "export_index.json", {"manifest": manifest_path.relative_to(run_dir).as_posix()})
    print(f"Complete export: {export_dir}", flush=True)
    return export_dir
