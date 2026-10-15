'예측 경계 기반 Ni/Eu 확률 보정과 검증.'
from __future__ import annotations
import gc
import hashlib
import importlib.metadata
import json
import os
import zipfile
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage
import komap_specialist_data as kd
import komap_phase_fusion as pf

CLASS_SLUGS = ("PrimarySi", "Al3Ni", "EutecticSi", "Al")
FAMILY = "komap_frozen_yolo_boundary_inference_v1"
TRANSFORMS = (
    {"name": "native1x", "scale": 1, "interpolation": "linear", "sharp_amount": 0.0, "sharp_sigma_native": 1.0},
    {"name": "linear2x", "scale": 2, "interpolation": "linear", "sharp_amount": 0.0, "sharp_sigma_native": 1.0},
    {"name": "cubic2x", "scale": 2, "interpolation": "cubic", "sharp_amount": 0.0, "sharp_sigma_native": 1.0},
    {"name": "sharp_native1x", "scale": 1, "interpolation": "linear", "sharp_amount": 0.25, "sharp_sigma_native": 1.0},
    {"name": "sharp_linear2x", "scale": 2, "interpolation": "linear", "sharp_amount": 0.25, "sharp_sigma_native": 1.0},
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _array_sha(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def checked_probability(probability, shape):
    p = np.asarray(probability, dtype=np.float32)
    if p.shape != (4, *shape) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1.00001:
        raise ValueError("Invalid four-class native probability map")
    if not np.allclose(p.sum(axis=0), 1, atol=1e-4):
        raise ValueError("Four-class probabilities must sum to one")
    return p


def transform_image(image, spec):

    x = np.asarray(image)
    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("uint8 H,W,3 RGB is required")
    if spec.get("scale") not in (1, 2) or spec.get("interpolation") not in ("linear", "cubic"):
        raise ValueError("Only prespecified 1x/2x linear/cubic transformations are allowed")
    amount, sigma = float(spec.get("sharp_amount", 0)), float(spec.get("sharp_sigma_native", 1))
    if amount not in (0, 0.25) or sigma != 1:
        raise ValueError("Only the prespecified mild native-pixel unsharp is allowed")
    result = x.copy()
    if amount:
        floating = result.astype(np.float32)
        blurred = cv2.GaussianBlur(floating, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        result = np.rint(np.clip(floating + amount * (floating-blurred), 0, 255)).astype(np.uint8)
    if spec["scale"] == 2:
        h, w = result.shape[:2]
        interpolation = cv2.INTER_LINEAR if spec["interpolation"] == "linear" else cv2.INTER_CUBIC
        result = cv2.resize(result, (2*w, 2*h), interpolation=interpolation)
    return np.ascontiguousarray(result)


def restore_probability(probability, native_shape):

    h, w = tuple(native_shape)
    p = np.asarray(probability, dtype=np.float32)
    if p.ndim != 3 or p.shape[0] != 4:
        raise ValueError("Four probability channels are required")
    checked_probability(p, p.shape[1:])
    if p.shape[1:] == (h, w):
        return p.copy()
    if p.shape[1:] != (2*h, 2*w):
        raise ValueError("Only exact 2x probability maps may be restored")
    restored = np.stack([cv2.resize(channel, (w, h), interpolation=cv2.INTER_AREA) for channel in p]).astype(np.float32)
    restored /= np.maximum(restored.sum(axis=0, keepdims=True), 1e-12)
    return checked_probability(restored, (h, w))


def predicted_pair_gate(reference_final, pair_mass=0.6, margin=0.2, boundary_radius=2):
    'Gate from frozen baseline predictions, never a ground-truth boundary.'
    if (pair_mass, margin, boundary_radius) != (0.6, 0.2, 2):
        raise ValueError("Gate thresholds are prespecified; do not tune per image")
    p = checked_probability(reference_final, np.asarray(reference_final).shape[1:])
    top2 = np.sort(np.argsort(p, axis=0)[-2:], axis=0)
    uncertain_pair = (top2[0] == 1) & (top2[1] == 2) & (p[1]+p[2] >= pair_mass) & (np.abs(p[1]-p[2]) <= margin)
    labels = p.argmax(axis=0)
    boundary = np.zeros(labels.shape, bool)
    ydiff, xdiff = labels[1:] != labels[:-1], labels[:, 1:] != labels[:, :-1]
    boundary[1:] |= ydiff; boundary[:-1] |= ydiff
    boundary[:, 1:] |= xdiff; boundary[:, :-1] |= xdiff
    near = ndimage.distance_transform_edt(~boundary) <= boundary_radius if boundary.any() else np.zeros_like(boundary)
    return uncertain_pair & near


def apply_pair_bias(raw_probability, gate, bias_to_eu):
    'Change only Ni/Eu ratio inside gate, conserving their probability mass.'
    p = checked_probability(raw_probability, np.asarray(raw_probability).shape[1:])
    gate = np.asarray(gate)
    if gate.dtype != np.bool_ or gate.shape != p.shape[1:]:
        raise ValueError("Boolean native-resolution gate is required")
    bias = float(bias_to_eu)
    if bias not in (0.0, -0.2, -0.1, 0.1, 0.2):
        raise ValueError("Pair bias must be prespecified in [-.2,-.1,0,.1,.2]")
    result = p.copy()
    if bias == 0 or not gate.any():
        return result
    ni, eu = p[1, gate].astype(np.float64), p[2, gate].astype(np.float64)
    mass = ni+eu
    a, b = ni*np.exp(-bias), eu*np.exp(bias)
    changed_eu = np.divide(mass*b, a+b, out=np.zeros_like(mass), where=a+b > 0)
    result[2, gate] = changed_eu.astype(np.float32)
    result[1, gate] = (mass-changed_eu).astype(np.float32)
    return checked_probability(result, p.shape[1:])


def boundary_diagnostics(target, prediction, radius=2):

    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape or target.ndim != 2 or radius != 2:
        raise ValueError("Matching 2D labels and radius2 are required")
    if target.min() < 0 or target.max() > 3 or prediction.min() < 0 or prediction.max() > 3:
        raise ValueError("Class ids must be 0,1,2,3")
    cm = np.bincount((4*target.astype(np.int64)+prediction).ravel(), minlength=16).reshape(4,4)
    near = np.zeros(target.shape, bool)
    for c in range(4):
        binary = target == c
        distance = ndimage.distance_transform_edt(np.pad(binary, 1, constant_values=False))[1:-1, 1:-1]
        near |= binary & (distance <= radius)
    bmatrix = np.bincount((4*target[near].astype(np.int64)+prediction[near]).ravel(), minlength=16).reshape(4,4)
    fields = {}
    for c, slug in enumerate(CLASS_SLUGS):
        tp = int(cm[c,c]); fp = int(cm[:,c].sum()-tp); fn = int(cm[c].sum()-tp)
        fp_near = int(bmatrix[:,c].sum()-bmatrix[c,c]); fn_near = int(bmatrix[c].sum()-bmatrix[c,c])
        fields.update({slug+"_fp": fp, slug+"_fn": fn, slug+"_fp_gt_source_boundary2": fp_near,
                       slug+"_fn_gt_boundary2": fn_near,
                       slug+"_precision": tp/(tp+fp) if tp+fp else None,
                       slug+"_recall": tp/(tp+fn) if tp+fn else None})
    for source in range(4):
        for destination in range(4):
            if source != destination:
                key = CLASS_SLUGS[source]+"_to_"+CLASS_SLUGS[destination]
                fields[key+"_pixels"] = int(cm[source,destination])
                fields[key+"_gt_source_boundary2_pixels"] = int(bmatrix[source,destination])
    return fields, cm, bmatrix


def score_probability_maps(records, maps):
    rows, cm_total, boundary_total = [], np.zeros((4,4),np.int64), np.zeros((4,4),np.int64)
    for rec in records:
        if rec.get("mask") is None:
            raise ValueError("Validation scoring requires GT; test inference must not score")
        p = checked_probability(maps[rec["name"]], rec["image"].shape[:2])
        pred = p.argmax(axis=0).astype(np.uint8)
        ignore, zero = kd.image_iou(rec["mask"], pred, "ignore"), kd.image_iou(rec["mask"], pred, "zero")
        diagnostics, cm, bmatrix = boundary_diagnostics(rec["mask"], pred)
        row = {"image":rec["name"],"height":pred.shape[0],"width":pred.shape[1],
               "miou_ignore":ignore["miou"],"miou_zero":zero["miou"],**diagnostics}
        row.update({slug+"_iou":ignore["class_iou"][c] for c,slug in enumerate(CLASS_SLUGS)})
        rows.append(row); cm_total += cm; boundary_total += bmatrix
    if not rows:
        raise ValueError("Validation cannot be empty")
    summary = {"images":len(rows), "miou_ignore":float(np.mean([r["miou_ignore"] for r in rows])),
               "miou_zero":float(np.mean([r["miou_zero"] for r in rows])),
               "class_iou_ignore":{slug:float(np.mean(values)) if (values := [r[slug+"_iou"] for r in rows if r[slug+"_iou"] is not None]) else None for slug in CLASS_SLUGS},
               "confusion_gt_rows_pred_columns":cm_total.tolist(),
               "boundary2_confusion_gt_rows_pred_columns":boundary_total.tolist(),
               "class_names":list(kd.CLASS_NAMES),"aggregation":"mean of per-image mIoU; absent union==0 ignored or zero",
               "boundary_definition":"GT source-class Euclidean interior distance <=2 native pixels; frame is boundary"}
    for c,slug in enumerate(CLASS_SLUGS):
        tp = int(cm_total[c,c]); fp = int(cm_total[:,c].sum()-tp); fn = int(cm_total[c].sum()-tp)
        summary[slug+"_fp"] = fp; summary[slug+"_fn"] = fn
        summary[slug+"_precision"] = tp/(tp+fp) if tp+fp else None
        summary[slug+"_recall"] = tp/(tp+fn) if tp+fn else None
        summary[slug+"_fp_gt_source_boundary2"] = int(boundary_total[:,c].sum()-boundary_total[c,c])
        summary[slug+"_fn_gt_boundary2"] = int(boundary_total[c].sum()-boundary_total[c,c])
    total_errors = int(cm_total.sum()-np.trace(cm_total))
    boundary_errors = int(boundary_total.sum()-np.trace(boundary_total))
    summary["misclassified_pixels"] = total_errors
    summary["misclassified_gt_source_boundary2_pixels"] = boundary_errors
    summary["misclassified_gt_source_boundary2_fraction"] = boundary_errors/total_errors if total_errors else None
    return summary, rows


def choose_candidate(summaries, absent_policy="ignore", baseline_key="native1x", tolerance=1e-8):
    if absent_policy not in ("ignore","zero") or baseline_key not in summaries:
        raise ValueError("A fixed absent policy and current native baseline are required")
    metric = "miou_"+absent_policy
    scores = {name:float(value[metric]) for name,value in summaries.items()}
    if not all(np.isfinite(v) for v in scores.values()):
        raise ValueError("Every candidate mIoU must be finite")
    maximum = max(scores.values())
    if scores[baseline_key] >= maximum-tolerance:
        return baseline_key

    return next(name for name in summaries if scores[name] >= maximum-tolerance)


def cache_write(path, probability, provenance):
    path = Path(path)
    p = checked_probability(probability,np.asarray(probability).shape[1:])
    if path.exists() or path.with_suffix(path.suffix+".json").exists():
        existing = cache_read(path,provenance,p.shape[1:])
        if not np.array_equal(existing,p):raise ValueError("Refusing to overwrite a different probability cache")
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("wb") as f:np.savez_compressed(f,probabilities=p,class_ids=np.arange(4,dtype=np.int16))
    os.replace(temporary,path)
    pf.atomic_json(path.with_suffix(path.suffix+".json"),{"provenance":provenance,"file_sha256":pf.file_sha(path),
                   "array_sha256":_array_sha(p),"shape":list(p.shape)})


def cache_read(path, provenance, shape):
    path = Path(path)
    sidecar = path.with_suffix(path.suffix+".json")
    if not path.exists() and not sidecar.exists():return None
    if not path.is_file() or not sidecar.is_file():raise ValueError("Incomplete probability cache; use a new run directory")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata["provenance"] != provenance or metadata["file_sha256"] != pf.file_sha(path):
        raise ValueError("Prediction cache provenance or file changed")
    with np.load(path,allow_pickle=False) as z:
        if not np.array_equal(z["class_ids"],np.arange(4,dtype=np.int16)):raise ValueError("Cached class order differs")
        p = checked_probability(z["probabilities"],shape).copy()
    if metadata["array_sha256"] != _array_sha(p) or metadata["shape"] != list(p.shape):
        raise ValueError("Prediction cache array changed")
    return p


def candidate_definitions():
    definitions = {spec["name"]:{"components":[[spec["name"],1.0]],"pair_bias":0.0} for spec in TRANSFORMS}
    for weight in (0.25,0.5,0.75):
        definitions[f"native_linear2x_mix_{weight:g}"] = {"components":[["native1x",1-weight],["linear2x",weight]],"pair_bias":0.0}
    for transform in ("cubic2x","sharp_native1x","sharp_linear2x"):
        definitions[f"native_{transform}_mix_0.5"] = {"components":[["native1x",0.5],[transform,0.5]],"pair_bias":0.0}
    for parent in ("native1x","native_linear2x_mix_0.5"):
        for bias in (-0.2,-0.1,0.1,0.2):
            definitions[f"{parent}_pairbias_{bias:+g}"] = {"components":definitions[parent]["components"],"pair_bias":bias}
    return definitions


def _candidate_map(definition, transforms, gate):
    parts = definition["components"]
    if len(parts) == 1 and parts[0][1] == 1:
        result = transforms[parts[0][0]].copy()
    else:
        result = sum(transforms[name]*np.float32(weight) for name,weight in parts).astype(np.float32)
    return apply_pair_bias(result,gate,definition["pair_bias"])


def _runtime_metadata(device="cpu"):
    versions = {}
    for package in ("numpy","scipy","opencv-python","opencv-python-headless","torch","torchvision","ultralytics","segmentation-models-pytorch"):
        try:versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package] = None
    runtime = {"packages":versions,"opencv_actual":cv2.__version__,"device":device}
    try:
        import torch
        runtime.update({"torch_actual":torch.__version__,"cuda_runtime":torch.version.cuda,"cudnn_version":torch.backends.cudnn.version(),
                        "tf32_matmul":torch.backends.cuda.matmul.allow_tf32,"tf32_cudnn":torch.backends.cudnn.allow_tf32,
                        "cudnn_enabled":torch.backends.cudnn.enabled,"cudnn_benchmark":torch.backends.cudnn.benchmark,
                        "cudnn_deterministic":torch.backends.cudnn.deterministic,"deterministic_algorithms":torch.are_deterministic_algorithms_enabled()})
        if device == "cuda" and torch.cuda.is_available():
            index = torch.cuda.current_device()
            runtime.update({"gpu_name":torch.cuda.get_device_name(index),"gpu_capability":list(torch.cuda.get_device_capability(index)),
                            "gpu_total_memory":torch.cuda.get_device_properties(index).total_memory})
    except ImportError:
        pass
    return runtime


def _code_hashes():
    directory = Path(__file__).resolve().parent
    names = (Path(__file__).name,"komap_boundary_source.py","komap_metal_methods_ram.py","komap_nieu_zoom_ram.py","komap_specialist_data.py","komap_phase_fusion.py")
    return {name:pf.file_sha(directory/name) for name in names if (directory/name).is_file()}


def _source_native(source_info, source_export, split, rec):
    native = source_info.get("native_probabilities",{}).get(split,{}).get(rec["name"])
    if native is not None:return checked_probability(native,rec["image"].shape[:2])
    if source_export is None:return None
    if source_export["manifest"].get("checkpoint_sha256") != source_info["sha256"]:
        raise ValueError("Native export checkpoint differs from source best50")
    return pf.read_probability(source_export,split,rec)


def _inference_provider(source_info,device):
    import torch
    import komap_metal_methods_ram as mm
    import komap_nieu_zoom_ram as core
    if device != "cuda" or not torch.cuda.is_available():raise RuntimeError("GPU required unless an explicit test inference callback is supplied")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    path = Path(source_info["path"])
    if pf.file_sha(path) != source_info["sha256"]:raise ValueError("Frozen source checkpoint changed")
    state = torch.load(path,map_location="cpu",weights_only=False)
    if state["config"] != source_info["config"] or state["epoch"] != source_info["epoch"]:
        raise ValueError("Source config/epoch differs from frozen snapshot")
    if state["identity"] != source_info["identity"]:
        raise ValueError("Source training identity differs from frozen snapshot")
    c = mm.Config(**state["config"]); c.validate()
    profile = source_info["backend"]
    if state.get("execution_backend") != profile:raise ValueError("Source execution backend changed")
    amp_dtype = mm.apply_execution_backend(profile["selected"])
    model = mm.build_model(c,pretrained=False).to(device,memory_format=mm.model_memory_format(c))
    model.load_state_dict(state["model"],strict=True); model.eval()
    gpu_input = mm.GPUInput(torch.device(device),c)
    del state
    def predict(image,spec):
        with torch.inference_mode():
            return core.predict_probability(model,image,c,gpu_input,amp_dtype,4)
    return predict,model


def _export_maps(destination, cache, selected_raw, selected_final, baseline_valid, source_info, identity, summaries, rows, ranking, selected, policy, strict_counts):
    destination = Path(destination); destination.mkdir(parents=True,exist_ok=False)
    files = []
    for split in ("valid","test"):
        if strict_counts and len(cache[split]) != kd.EXPECTED_COUNTS[split]:raise ValueError("Expected valid20/test10")
        for rec in cache[split]:
            raw = checked_probability(selected_raw[split][rec["name"]],rec["image"].shape[:2])
            final = checked_probability(selected_final[split][rec["name"]],rec["image"].shape[:2])
            stem = Path(rec["name"]).stem
            probability_path, final_path = destination/split/(stem+".npz"),destination/("blend_"+split)/(stem+".npz")
            for path,array in ((probability_path,raw),(final_path,final)):
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open("wb") as f:np.savez_compressed(f,probabilities=array,class_ids=np.arange(4,dtype=np.int16))
            mask_name = rec["name"].replace("_image.png","_mask.png")
            raw_mask = destination/("raw_"+split+"_masks")/mask_name
            selected_mask = destination/("selected_"+split+"_masks")/mask_name
            kd.save_rgb_mask(raw.argmax(axis=0).astype(np.uint8),raw_mask)
            kd.save_rgb_mask(final.argmax(axis=0).astype(np.uint8),selected_mask)
            if split == "valid":
                baseline = checked_probability(baseline_valid[rec["name"]],rec["image"].shape[:2])
                kd.save_rgb_mask(baseline.argmax(axis=0).astype(np.uint8),destination/"baseline_valid_masks"/mask_name)
            files.append({"split":split,"image":rec["name"],"image_sha256":rec["image_sha256"],"mask_sha256":rec.get("mask_sha256"),
                          "shape":list(raw.shape),"probability_file":probability_path.relative_to(destination).as_posix(),
                          "probability_sha256":pf.file_sha(probability_path),"final_blend_probability_file":final_path.relative_to(destination).as_posix(),
                          "final_blend_probability_sha256":pf.file_sha(final_path),"raw_mask_file":raw_mask.relative_to(destination).as_posix(),
                          "selected_mask_file":selected_mask.relative_to(destination).as_posix()})
        with zipfile.ZipFile(destination/("selected_"+split+"_masks.zip"),"w",zipfile.ZIP_DEFLATED) as z:
            for path in sorted((destination/("selected_"+split+"_masks")).glob("*.png")):z.write(path,arcname=path.name)
    pf.write_csv(destination/"candidate_ranking.csv",ranking)
    pf.write_csv(destination/"validation_selected_per_image.csv",rows[selected])
    pf.write_csv(destination/"validation_candidates_per_image.csv",[{"candidate":name,**r} for name,values in rows.items() for r in values])
    result = {"selected_candidate":selected,"selected":summaries[selected],"existing_baseline":summaries["native1x"],"candidates":summaries,
              "absent_policy":policy,"new_model_weight":identity["new_model_weight"],"checkpoint_epoch":source_info["epoch"],
              "checkpoint_sha256":source_info["sha256"],"source_checkpoint_score":source_info.get("source_score"),
              "source_score_recomputed":summaries["native1x"]["miou_"+policy],
              "source_score_reproduction_delta":(summaries["native1x"]["miou_"+policy]-float(source_info["source_score"])) if source_info.get("source_score") is not None else None,
              "selection":"maximum mean per-image overall 4-class mIoU; native baseline wins ties within 1e-8; no FP or recall constraint",
              "note":"Validation-selected experiment, not an independent generalization score. No GPU training is performed."}
    pf.atomic_json(destination/"validation_summary.json",result)
    pf.atomic_json(destination/"export_settings.json",identity)
    with zipfile.ZipFile(destination/"analysis_bundle.zip","w",zipfile.ZIP_DEFLATED) as z:
        for name in ("candidate_ranking.csv","validation_selected_per_image.csv","validation_candidates_per_image.csv","validation_summary.json","export_settings.json"):
            z.write(destination/name,arcname=name)
        for path in sorted((destination/"selected_valid_masks").glob("*.png")):z.write(path,arcname="selected_valid_masks/"+path.name)
        for path in sorted((destination/"baseline_valid_masks").glob("*.png")):z.write(path,arcname="baseline_valid_masks/"+path.name)
    artifacts = [{"path":p.relative_to(destination).as_posix(),"sha256":pf.file_sha(p)} for p in sorted(destination.rglob("*")) if p.is_file()]
    manifest = {"complete":True,"format_version":2,"model_family":"komap_pretrained_unet_multiclass_ram_v1","experiment_family":FAMILY,
                "target_class":-1,"architecture":source_info["config"]["architecture"],"config":source_info["config"],
                "class_names":list(kd.CLASS_NAMES),"palette":kd.PALETTE.tolist(),"dtype":"float32","identity":source_info["identity"],
                "checkpoint_sha256":source_info["sha256"],"checkpoint_epoch":source_info["epoch"],"tta_rotations":4,
                "selected_candidate":selected,"boundary_export_identity":identity,
                "probability":"selected transformed/bounded-pair-adjusted raw YOLO four-class probability; final mixture stored separately",
                "final_blend":{"new_model_weight":identity["new_model_weight"],"baseline_identity":identity["old_baseline_identity"]},
                "files":files,"artifacts":artifacts}
    pf.atomic_json(destination/"manifest.json",manifest)
    return result,manifest


def verify_export(destination,cache,identity):
    destination = Path(destination)
    manifest = json.loads((destination/"manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("boundary_export_identity") != identity:
        raise ValueError("Boundary export identity differs")
    for artifact in manifest["artifacts"]:
        if pf.file_sha(pf.contained_path(destination,artifact["path"])) != artifact["sha256"]:
            raise ValueError("Boundary export artifact changed")
    export = {"directory":destination,"manifest":manifest,"rows":{(r["split"],r["image"]):r for r in manifest["files"]}}
    expected = {(s,r["name"]) for s in ("valid","test") for r in cache[s]}
    if len(export["rows"]) != len(manifest["files"]) or set(export["rows"]) != expected:raise ValueError("Export image list differs")
    for split in ("valid","test"):
        for rec in cache[split]:
            raw = pf.read_probability(export,split,rec)
            row = export["rows"][(split,rec["name"])]
            with np.load(pf.contained_path(destination,row["final_blend_probability_file"]),allow_pickle=False) as z:
                final = checked_probability(z["probabilities"],rec["image"].shape[:2])
            selected_path = pf.contained_path(destination,row["selected_mask_file"])
            if not np.array_equal(kd.read_mask(selected_path),final.argmax(axis=0).astype(np.uint8)):
                raise ValueError("Selected native mask differs from exported final probability")
    return manifest


def run_boundary_experiment(cache, report, source_info, old_baseline, run_dir, *, source_export=None,
                            new_model_weight=0.5, device="cuda", inference_callback=None,
                            strict_counts=True, source_score_tolerance=5e-5):

    if new_model_weight != 0.5:raise ValueError("Keep current best50 composition: YOLO50%, old baseline50%")
    source_score_tolerance = float(source_score_tolerance)
    if not np.isfinite(source_score_tolerance) or source_score_tolerance < 0:
        raise ValueError("Source score tolerance must be finite and nonnegative")
    if source_info.get("source_score") is not None:
        source_score = float(source_info["source_score"])
        if not np.isfinite(source_score) or not 0 <= source_score <= 1:
            raise ValueError("Source checkpoint mIoU must be finite and in [0,1]")
    policy = source_info.get("source_policy",source_info["config"].get("absent_policy","ignore"))
    if policy not in ("ignore","zero"):raise ValueError("Invalid source absent-class policy")
    if strict_counts and any(len(cache[s]) != kd.EXPECTED_COUNTS[s] for s in ("valid","test")):
        raise ValueError("Expected valid20/test10")
    for split in ("valid","test"):
        if len({r["name"] for r in cache[split]}) != len(cache[split]):raise ValueError("Duplicate image names")
        for rec in cache[split]:checked_probability(old_baseline["probabilities"][split][rec["name"]],rec["image"].shape[:2])
    if inference_callback is None and pf.file_sha(source_info["path"]) != source_info["sha256"]:
        raise ValueError("Source checkpoint SHA mismatch")
    if inference_callback is None:
        import torch
        import komap_metal_methods_ram as mm
        if device != "cuda" or not torch.cuda.is_available():raise RuntimeError("CUDA is required for frozen source inference")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        mm.apply_execution_backend(source_info["backend"]["selected"])
    definitions = candidate_definitions()
    source_identity = {k:source_info.get(k) for k in ("sha256","config","epoch","backend","identity","source_score","source_policy")}
    source_identity["native_map_hashes"] = {split:{name:_array_sha(p) for name,p in maps.items()} for split,maps in source_info.get("native_probabilities",{}).items()}
    source_identity["native_export_manifest_sha256"] = source_export.get("manifest_sha256") if source_export else None
    identity = {"family":FAMILY,"source":source_identity,"data_sha256":report["data_sha256"],
                "old_baseline_identity":old_baseline["identity"],"old_baseline_probability_hashes":{s:{r["name"]:_array_sha(old_baseline["probabilities"][s][r["name"]]) for r in cache[s]} for s in ("valid","test")},
                "code_sha256":_code_hashes(),"runtime":_runtime_metadata(device),"tta_rotations":4,"new_model_weight":new_model_weight,
                "transforms":list(TRANSFORMS),"candidate_definitions":definitions,"restore":"INTER_AREA four channels then renormalize",
                "gate":{"source":"frozen native final prediction","pair":[1,2],"pair_mass":0.6,"margin":0.2,"boundary_radius":2},
                "selection_policy":policy,"tie_tolerance":1e-8,"source_score_tolerance":source_score_tolerance,"strict_counts":strict_counts,
                "prediction_provider":"GPU" if inference_callback is None else source_info.get("prediction_provider_identity","explicit_cpu_test_callback")}
    digest = _digest(identity)[:16]
    run_dir = Path(run_dir); run_dir.mkdir(parents=True,exist_ok=True)
    destination = run_dir/"exports"/("boundary_tta4_"+digest)
    if (destination/"manifest.json").is_file():
        verify_export(destination,cache,identity)
        pf.atomic_json(run_dir/"export_index.json",{"manifest":(destination/"manifest.json").relative_to(run_dir).as_posix()})
        print("완료 export 재사용:",destination,flush=True)
        return destination
    if destination.exists():raise ValueError("Incomplete export exists; use a new run directory")
    pf.atomic_json(run_dir/("experiment_identity_"+digest+".json"),identity)
    prediction_cache = run_dir/"prediction_cache"/digest
    provider, model = inference_callback, None
    def transformed(split,rec,spec):
        nonlocal provider,model
        provenance = {"experiment_digest":digest,"source_checkpoint_sha256":source_info["sha256"],"split":split,
                      "image":rec["name"],"image_sha256":rec["image_sha256"],"image_array_sha256":_array_sha(rec["image"]),
                      "transform":spec,"tta_rotations":4,"backend":source_info.get("backend"),"module_sha256":identity["code_sha256"]}
        filename = _digest({"image":rec["name"],"image_sha256":rec["image_sha256"]})[:20]+".npz"
        path = prediction_cache/split/spec["name"]/filename
        probability = cache_read(path,provenance,rec["image"].shape[:2])
        if probability is not None:return probability
        native = _source_native(source_info,source_export,split,rec) if spec["name"] == "native1x" else None
        if native is not None:probability = native.copy()
        else:
            if provider is None:provider,model = _inference_provider(source_info,device)
            image = transform_image(rec["image"],spec)
            probability = restore_probability(provider(image,spec),rec["image"].shape[:2])
        cache_write(path,probability,provenance)
        return probability
    summaries, rows, ranking = {}, {}, []
    try:
        transform_valid = {}
        for spec in TRANSFORMS:
            print("검증 추론:",spec["name"],flush=True)
            transform_valid[spec["name"]] = {r["name"]:transformed("valid",r,spec) for r in cache["valid"]}
        native_final = {r["name"]:np.float32(0.5)*old_baseline["probabilities"]["valid"][r["name"]]+np.float32(0.5)*transform_valid["native1x"][r["name"]] for r in cache["valid"]}
        gates = {r["name"]:predicted_pair_gate(native_final[r["name"]]) for r in cache["valid"]}
        for name,definition in definitions.items():
            raw = {r["name"]:_candidate_map(definition,{t:v[r["name"]] for t,v in transform_valid.items()},gates[r["name"]]) for r in cache["valid"]}
            final = native_final if name == "native1x" else {r["name"]:np.float32(0.5)*old_baseline["probabilities"]["valid"][r["name"]]+np.float32(0.5)*raw[r["name"]] for r in cache["valid"]}
            summaries[name],rows[name] = score_probability_maps(cache["valid"],final)
            print(f"{name}: mIoU={summaries[name]['miou_'+policy]:.8f}, Eu FP={summaries[name]['EutecticSi_fp']}",flush=True)
        reference_score = source_info.get("source_score")
        if reference_score is not None and abs(summaries["native1x"]["miou_"+policy]-float(reference_score)) > source_score_tolerance:
            raise ValueError("Native best50 reproduction differs from source checkpoint score; verify checkpoint/baseline/data/backend before selection")
        selected = choose_candidate(summaries,policy)
        for name,summary in summaries.items():
            ranking.append({"candidate":name,"selected":name==selected,"miou_ignore":summary["miou_ignore"],"miou_zero":summary["miou_zero"],
                            "delta_miou":summary["miou_"+policy]-summaries["native1x"]["miou_"+policy],
                            "Primary_iou":summary["class_iou_ignore"]["PrimarySi"],"Ni_iou":summary["class_iou_ignore"]["Al3Ni"],
                            "Eu_iou":summary["class_iou_ignore"]["EutecticSi"],"Al_iou":summary["class_iou_ignore"]["Al"],
                            "Eu_fp":summary["EutecticSi_fp"],"Eu_fn":summary["EutecticSi_fn"],
                            "Eu_precision":summary["EutecticSi_precision"],"Eu_recall":summary["EutecticSi_recall"]})
        ranking.sort(key=lambda row:(-row["miou_"+policy],not row["selected"],row["candidate"]))
        selected_raw,selected_final = {"valid":{},"test":{}},{"valid":{},"test":{}}
        definition = definitions[selected]
        specs = {spec["name"]:spec for spec in TRANSFORMS}
        for split in ("valid","test"):
            for rec in cache[split]:
                name = rec["name"]
                needed = {key for key,_ in definition["components"]}
                if definition["pair_bias"]:needed.add("native1x")
                transforms = {key:(transform_valid[key][name] if split=="valid" else transformed(split,rec,specs[key])) for key in needed}
                old = old_baseline["probabilities"][split][name]
                gate = gates[name] if split=="valid" else (predicted_pair_gate(np.float32(0.5)*old+np.float32(0.5)*transforms["native1x"]) if definition["pair_bias"] else np.zeros(rec["image"].shape[:2],bool))
                p = _candidate_map(definition,transforms,gate)
                selected_raw[split][name] = p
                selected_final[split][name] = np.float32(0.5)*old+np.float32(0.5)*p
        staging = destination.with_name(destination.name+".pending")
        if staging.exists():raise ValueError("Incomplete pending export exists; use a new run directory")
        result,manifest = _export_maps(staging,cache,selected_raw,selected_final,native_final,source_info,identity,summaries,rows,ranking,selected,policy,strict_counts)
        verify_export(staging,cache,identity)
        destination.parent.mkdir(parents=True,exist_ok=True)
        os.replace(staging,destination)
        pf.atomic_json(run_dir/"export_index.json",{"manifest":(destination/"manifest.json").relative_to(run_dir).as_posix()})
        print("선택:",selected,"전체 mIoU:",result["selected"]["miou_"+policy],"기준:",result["existing_baseline"]["miou_"+policy],flush=True)
        print("결과:",destination,flush=True)
        return destination
    finally:
        provider = None; model = None; gc.collect()
        if device == "cuda":
            import torch
            if torch.cuda.is_available():torch.cuda.empty_cache()
