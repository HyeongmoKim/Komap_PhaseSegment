'YOLO 50%, DINO 25%, specialist 25%의 고정 경계 보정 추론.'
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image


sys.dont_write_bytecode = True
import komap_boundary_inference as bi
import komap_phase_fusion as pf
import komap_specialist_data as kd


CANDIDATE = "native1x_pairbias_-0.1"
ARCHIVED_VALIDATION_MIOU = 0.8000702044837322
YOLO_SHA256 = "665e9cfe08932df39c5cdf3b4162a2a84d1500c12bf04d55340665f4c5708bb9"
DINO_SHA256 = "1361e35f7331e417704b216bb0253ce221c8d69f58b44e7bce0bd05dcdae17b3"
DATA_SHA256 = "d794a4aa0a19331b1a90f7106b7cba6dbc199df05e13fdde1f35a399154c1872"


EXPORT_SHA256 = {
    "primary": "21c476a7ec9fbcf6f7b4dbdb4132f80157d56aa99ae9a99d5ac23865c2586bd0",
    "ni": "e6266113f0ef4af905a431eb93c3d0ab88ad64f40d57f5cbe993be1c4f2c49fa",
    "eu": "117c423fe5e851c6e4a949aad130081fb93e290a61b3edde7711ff2ec3642948",
    "al": "1e46fe2e19e0cd0878885a006d749f7b4a8ec6e3d9b7e64fe184ba5baf350352",
    "dino": "ef5d8105824ae37facb7caf475258bd9cd077128d0f69a38340b9545c1f187b0",
}
_ORIGINAL_BASELINE_IDENTITY = {
    "specialists": {
        "baseline": [EXPORT_SHA256["primary"], EXPORT_SHA256["ni"],
                     "c9283f590ed6f791db2af99222a31663a7db9e57abab66400de46f2f8c088e02",
                     EXPORT_SHA256["al"]],
        "replace_Eu": EXPORT_SHA256["eu"],
        "weights": [1.0, 1.0, 1.0, 1.0],
        "alpha": 0.5,
        "normalization": "weighted sigmoid scores divided by channel sum",
    },
    "source_export_sha256": EXPORT_SHA256["dino"],
    "source_checkpoint_sha256": DINO_SHA256,
    "source_weight": 0.5,
    "description": "Eu-replaced specialists 50% + previous control DINO best 50%",
}


CURRENT_EXPORT_SHA256 = {'dino': 'ef5d8105824ae37facb7caf475258bd9cd077128d0f69a38340b9545c1f187b0', 'primary': '21c476a7ec9fbcf6f7b4dbdb4132f80157d56aa99ae9a99d5ac23865c2586bd0', 'ni': 'e6266113f0ef4af905a431eb93c3d0ab88ad64f40d57f5cbe993be1c4f2c49fa', 'eu': '7c4e2a9e85b32d4b1d785bfbb8ec04e6bf8b5d8f7adc8f6e0ddeb45104c0c134', 'al': '1e46fe2e19e0cd0878885a006d749f7b4a8ec6e3d9b7e64fe184ba5baf350352'}
CURRENT_BASELINE_MIOU = 0.8000754847573367


def _full_digest(value: str, field: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} requires a full 64-character SHA256 digest")
    return value


def _load_fixed_export(run_dir: str | Path, role: str, expected_sha: str,
                       target_class: int) -> dict:

    root = Path(run_dir).resolve()
    if not (root / "export_index.json").is_file():
        raise FileNotFoundError(
            f"{role}: original completed export_index.json is missing under {root}. "
            "A notebook or analysis bundle does not contain this model's probabilities.")
    export = pf.load_export(root)
    digest = _full_digest(expected_sha, f"{role} export SHA256")
    if export["manifest_sha256"] != digest:
        raise ValueError(f"{role}: export differs from the frozen 0.8001 model; "
                         f"expected {digest}, found {export['manifest_sha256']}")
    manifest = export["manifest"]
    if manifest.get("target_class") != target_class:
        raise ValueError(f"{role}: wrong probability class order/target class")
    if role == "dino":
        if manifest.get("checkpoint_sha256") != DINO_SHA256 or manifest.get("tta_rotations") != 4:
            raise ValueError("DINO export must use the original epoch6 checkpoint and TTA4")
    return export


def load_test_images(images_dir: str | Path, expected_count: int = 10) -> list[dict]:

    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("expected_count must be a positive integer")
    root = Path(images_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Test image directory is missing: {root}")
    paths = sorted(root.rglob("*_image.png"), key=lambda p: (p.name, p.as_posix()))
    if len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count} test images, found {len(paths)} in {root}")
    if len({p.name for p in paths}) != len(paths):
        raise ValueError("Duplicate test image names are ambiguous")
    records = []
    for path in paths:


        import io
        content = path.read_bytes()
        with Image.open(io.BytesIO(content)) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or not all(rgb.shape[:2]):
            raise ValueError(f"Invalid RGB image: {path}")
        records.append({"name": path.name, "image": rgb, "mask": None,
                        "image_sha256": hashlib.sha256(content).hexdigest(),
                        "mask_sha256": None})
    return records


def fixed_probability(yolo: np.ndarray, dino: np.ndarray,
                      specialists: np.ndarray) -> np.ndarray:
    'Exact frozen gate/bias from komap_boundary_inference; no GT input.'
    shape = np.asarray(yolo).shape[1:]
    yolo = bi.checked_probability(yolo, shape)
    dino = bi.checked_probability(dino, shape)
    specialists = bi.checked_probability(specialists, shape)
    old = bi.checked_probability(np.float32(.5) * specialists + np.float32(.5) * dino, shape)
    native = bi.checked_probability(np.float32(.5) * old + np.float32(.5) * yolo, shape)
    gate = bi.predicted_pair_gate(native)
    adjusted = bi.apply_pair_bias(yolo, gate, -0.1)
    return bi.checked_probability(np.float32(.5) * old + np.float32(.5) * adjusted, shape)


def _checkpoint_provider(checkpoint: str | Path):

    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Original YOLO03 best50 checkpoint is missing: {path}")
    if pf.file_sha(path) != YOLO_SHA256:
        raise ValueError("YOLO checkpoint SHA256 differs from the archived SizeBalanced best50")
    import torch
    import komap_metal_methods_ram as mm
    import komap_nieu_zoom_ram as core
    if not torch.cuda.is_available():
        raise RuntimeError("YOLO checkpoint inference requires CUDA; verified raw export mode supports CPU")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if pf.file_sha(path) != YOLO_SHA256:
        raise ValueError("YOLO checkpoint changed while it was being read")
    if not isinstance(state, dict) or state.get("selection") != "fixed50" or state.get("epoch") != 1:
        raise ValueError("Unexpected original YOLO checkpoint selection/epoch")
    identity = state.get("identity", {})
    if (identity.get("baseline_context") != _ORIGINAL_BASELINE_IDENTITY
            or identity.get("data_sha256") != DATA_SHA256
            or identity.get("config") != state.get("config")):
        raise ValueError("YOLO checkpoint's training/model composition identity differs")
    c = mm.Config(**state["config"])
    c.validate()
    if (c.architecture != "yolo11_edge_semantic" or c.patch != 480
            or c.eval_stride != 240 or not c.eu_component_balanced
            or c.decoder_p1 or c.detail_channels != 16):
        raise ValueError("Unexpected original SizeBalanced YOLO architecture/inference profile")
    backend = state.get("execution_backend", {})
    if "selected" not in backend:
        raise ValueError("Saved YOLO inference backend is missing; no profile will be guessed")


    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = mm.apply_execution_backend(backend["selected"])
    model = mm.build_model(c, pretrained=False).to("cuda", memory_format=mm.model_memory_format(c))
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    gpu_input = mm.GPUInput(torch.device("cuda"), c)
    profile = {"mode": "checkpoint", "checkpoint_sha256": YOLO_SHA256,
               "patch": 480, "stride": 240, "overlap": "gaussian", "tta_rotations": 4,
               "saved_backend": backend["selected"], "original_gpu": backend.get("gpu"),
               "current_gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
               "torch": str(torch.__version__), "cuda": torch.version.cuda}
    del state

    def predict(image):
        with torch.inference_mode():
            return core.predict_probability(model, image, c, gpu_input, amp_dtype, 4)

    return predict, profile


def _raw_yolo_export(run_dir: str | Path, manifest_sha256: str | None) -> dict:
    if manifest_sha256 is None:
        raise ValueError("Raw YOLO export mode requires its full --yolo-export-sha256 digest "
                         "from the original completed run")
    export = _load_fixed_export(run_dir, "yolo", manifest_sha256, -1)
    manifest = export["manifest"]
    if (manifest.get("checkpoint_sha256") != YOLO_SHA256
            or manifest.get("architecture") != "yolo11_edge_semantic"
            or manifest.get("tta_rotations") != 4
            or manifest.get("boundary_export_identity") is not None
            or manifest.get("methods_export_identity") is None
            or manifest.get("probability") != "raw new-model four-class softmax, native dimensions; final mixture stored separately"):
        raise ValueError("YOLO export must contain the native raw original best50 probability; "
                         "a blended/postprocessed export cannot be used as raw YOLO")
    config = manifest.get("config", {})
    if (config.get("patch") != 480 or config.get("eval_stride") != 240
            or manifest.get("identity", {}).get("baseline_context") != _ORIGINAL_BASELINE_IDENTITY):
        raise ValueError("Raw YOLO export profile/model composition differs")
    return export


def predict_test(images_dir: str | Path, output_dir: str | Path, *,
                 dino_run: str | Path, primary_run: str | Path, ni_run: str | Path,
                 eu_run: str | Path, al_run: str | Path,
                 yolo_checkpoint: str | Path | None = None,
                 yolo_run: str | Path | None = None,
                 yolo_export_sha256: str | None = None,
                 expected_count: int = 10, reference_mode: str = 'current_base') -> dict:
    'Produce native palette PNGs using frozen external assets.'
    if (yolo_checkpoint is None) == (yolo_run is None):
        raise ValueError("Provide exactly one of yolo_checkpoint or yolo_run")
    if yolo_checkpoint is not None and yolo_export_sha256 is not None:
        raise ValueError("yolo_export_sha256 applies only to yolo_run mode")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists; choose a new output directory: {destination}")
    records = load_test_images(images_dir, expected_count)
    runs: Mapping[str, str | Path] = {"dino": dino_run, "primary": primary_run,
                                     "ni": ni_run, "eu": eu_run, "al": al_run}
    targets = {"dino": -1, "primary": 0, "ni": 1, "eu": 2, "al": 3}
    if reference_mode not in ('original','current_base'):
        raise ValueError('reference_mode must be original or current_base')
    expected_exports = EXPORT_SHA256 if reference_mode == 'original' else CURRENT_EXPORT_SHA256
    exports = {role: _load_fixed_export(run, role, expected_exports[role], targets[role])
               for role, run in runs.items()}
    yolo_export = _raw_yolo_export(yolo_run, yolo_export_sha256) if yolo_run is not None else None


    old_maps = {}
    raw_maps = {}
    for rec in records:
        dino = pf.read_probability(exports["dino"], "test", rec)
        score = np.stack([pf.read_probability(exports[role], "test", rec)
                          for role in ("primary", "ni", "eu", "al")])
        score /= np.maximum(score.sum(axis=0, keepdims=True), np.float32(1e-8))
        shape = rec["image"].shape[:2]
        old_maps[rec["name"]] = (bi.checked_probability(dino, shape),
                                   bi.checked_probability(score, shape))
        if yolo_export is not None:
            raw_maps[rec["name"]] = pf.read_probability(yolo_export, "test", rec)
    provider = None
    if yolo_export is None:
        provider, profile = _checkpoint_provider(yolo_checkpoint)
    else:
        profile = {"mode": "raw_export", "checkpoint_sha256": YOLO_SHA256,
                   "manifest_sha256": yolo_export["manifest_sha256"],
                   "patch": 480, "stride": 240, "overlap": "gaussian", "tta_rotations": 4}
    labels = {}
    for rec in records:
        raw = raw_maps[rec["name"]] if yolo_export is not None else provider(rec["image"])
        dino, specialists = old_maps[rec["name"]]
        final = fixed_probability(raw, dino, specialists)
        labels[rec["name"].replace("_image.png", "_mask.png")] = final.argmax(axis=0).astype(np.uint8)


    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    for name, prediction in labels.items():
        kd.save_rgb_mask(prediction, destination / name)
    return {"candidate": CANDIDATE, "images": len(records),
            "output_dir": str(destination), "masks": list(labels),
            "yolo": profile, "reference_mode": reference_mode, "reference_export_manifest_sha256": dict(expected_exports),
            "ground_truth_used": False,
            "note": "Inference completed. No test score is available without independent GT."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="Directory containing original test *_image.png files")
    parser.add_argument("--output", required=True, help="New output directory; existing directories are rejected")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--yolo-checkpoint", help="Original YOLO03 SizeBalanced best50.pt")
    source.add_argument("--yolo-run", help="Original completed native raw YOLO probability export run")
    parser.add_argument("--yolo-export-sha256", help="Full original raw YOLO export manifest SHA256; required with --yolo-run")
    for role in ("dino", "primary", "ni", "eu", "al"):
        parser.add_argument(f"--{role}-run", required=True, help=f"Original completed {role} export run")
    parser.add_argument("--expected-count", type=int, default=10,
                        help="Image count check; changing it does not relax frozen model/image validation")
    parser.add_argument('--reference-mode',choices=('original','current_base'),default='current_base')
    args = parser.parse_args(argv)
    result = predict_test(args.images, args.output, dino_run=args.dino_run,
                          primary_run=args.primary_run, ni_run=args.ni_run,
                          eu_run=args.eu_run, al_run=args.al_run,
                          yolo_checkpoint=args.yolo_checkpoint, yolo_run=args.yolo_run,
                          yolo_export_sha256=args.yolo_export_sha256,
                          expected_count=args.expected_count, reference_mode=args.reference_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
