'완료 export 검증과 모델 확률 결합.'
from __future__ import annotations
import csv
import hashlib
import json
import os
import zipfile
from pathlib import Path
import numpy as np
import komap_specialist_data as kd

CLASS_SLUGS = ("PrimarySi", "Al3Ni", "EutecticSi", "Al")
BASE_HINTS = (
    "0302538a5b574c0c777f3f8bd8f451e62d24e8211e6091ecee85159330957a77",
    "02ba82339505a45cd0ca5691ee0c7a1b9687059f006294e33142b6531bb2b556",
    "062d5307e9108398b62680442acdab765906c71cb92266e1a6e159f806af974c",
    "a692eb22ff67476b4e69f6b8e6b23059de02ab072725bb42a34b0b2ac2c58b01")

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)

def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)

def contained_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Invalid export path")
    return path

def load_export(run_dir):
    run_dir = Path(run_dir)
    index = json.loads((run_dir / "export_index.json").read_text(encoding="utf-8"))
    path = contained_path(run_dir, index["manifest"])
    m = json.loads(path.read_text(encoding="utf-8"))
    if not m.get("complete") or m.get("palette") != kd.PALETTE.tolist() or m.get("class_names") != list(kd.CLASS_NAMES):
        raise ValueError(f"Incomplete export or different class order: {path}")
    if m.get("model_family") not in ("komap_unet_binary_specialist_v1", "komap_pretrained_unet_multiclass_ram_v1"):
        raise ValueError(f"Unsupported probability export: {path}")
    rows = {(r["split"], r["image"]): r for r in m["files"]}
    expected = {"valid": 20, "test": 10}
    if len(rows) != len(m["files"]) or set(s for s, _ in rows) != set(expected):
        raise ValueError("Duplicate or unexpected exported images")
    if any(sum(s == split for s, _ in rows) != count for split, count in expected.items()):
        raise ValueError("Export must contain valid20/test10")
    return {"run_dir": run_dir, "directory": path.parent, "manifest": m, "rows": rows, "manifest_sha256": file_sha(path)}

def resolve_baseline(project_root, run_names=(None, None, None, None)):
    if len(run_names) != 4:
        raise ValueError("Four baseline names are required")
    candidates = {c: [] for c in range(4)}
    for path in sorted((Path(project_root) / "runs").rglob("export_index.json")):
        try:
            export = load_export(path.parent)
        except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError):
            continue
        m = export["manifest"]
        if m.get("model_family") == "komap_unet_binary_specialist_v1" and m.get("target_class") in candidates:
            candidates[m["target_class"]].append(export)
    result = []
    for c, requested in enumerate(run_names):
        if requested is not None:
            export = load_export(Path(project_root) / "runs" / requested)
        else:
            matching = [e for e in candidates[c] if e["manifest"].get("checkpoint_sha256") == BASE_HINTS[c]]
            pool = matching if matching else candidates[c]
            if len(pool) != 1:
                names = [str(e["run_dir"].relative_to(Path(project_root) / "runs")) for e in candidates[c]]
                raise ValueError(f"{CLASS_SLUGS[c]}: BASE_RUN_NAMES[{c}]를 지정하세요. 후보={names}")
            export = pool[0]
        if export["manifest"].get("target_class") != c:
            raise ValueError(f"Wrong class at baseline position {c}")
        result.append(export)
    first = result[0]["manifest"]["identity"]["data_sha256"]
    if any(e["manifest"]["identity"]["data_sha256"] != first for e in result):
        raise ValueError("Baseline specialists used different data")
    print("기존 조합:", [str(e["run_dir"].name) for e in result])
    return result

def read_probability(export, split, rec):
    row = export["rows"].get((split, rec["name"]))
    if row is None or row["image_sha256"] != rec["image_sha256"] or row["mask_sha256"] != rec["mask_sha256"]:
        raise ValueError(f"Image/mask bytes differ from export: {rec['name']}")
    path = contained_path(export["directory"], row["probability_file"])
    if file_sha(path) != row["probability_sha256"]:
        raise ValueError(f"Probability file changed: {path}")
    target = export["manifest"].get("target_class", -1)
    with np.load(path, allow_pickle=False) as z:
        raw = z["probability"] if target >= 0 else z["probabilities"]
        if target >= 0 and int(z["target_class"].item()) != target:
            raise ValueError("Probability target class mismatch")
        p = raw.astype(np.float32, copy=True)
    h, w = rec["image"].shape[:2]
    expected = (h, w) if target >= 0 else (4, h, w)
    if p.shape != expected or list(p.shape) != row["shape"] or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1.00001:
        raise ValueError(f"Invalid probability map: {path}")
    if target < 0 and not np.allclose(p.sum(axis=0), 1, atol=1e-4):
        raise ValueError("Multiclass probabilities must sum to one")
    return p

def fusion_context(exports, cache, weights=(1.,1.,1.,1.)):
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (4,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Four positive finite weights are required")
    return {"probabilities": {r["name"]: np.stack([read_probability(e, "valid", r) for e in exports]) for r in cache["valid"]},
            "weights": weights.tolist(), "identity": {"manifests": [e["manifest_sha256"] for e in exports], "weights": weights.tolist()}}

def prepare_cpu_ram(project_root, data_root):
    import io
    from PIL import Image
    from tqdm.auto import tqdm
    kd.extract_archives(Path(project_root)/"data", data_root)
    cache = {}
    for split in ("valid", "test"):
        pairs = kd.discover_split(data_root, split)
        if len(pairs) != kd.EXPECTED_COUNTS[split]:
            raise ValueError(f"Unexpected {split} count")
        cache[split] = []
        for ip, mp in tqdm(pairs, desc=split + " → RAM"):
            b = ip.read_bytes()
            with Image.open(io.BytesIO(b)) as im:
                image = np.array(im.convert("RGB"))
            mask, mask_sha = None, None
            if mp:
                mb = mp.read_bytes()
                with Image.open(io.BytesIO(mb)) as im:
                    mask = kd.decode_mask(np.array(im.convert("RGBA")))
                mask_sha = hashlib.sha256(mb).hexdigest()
            cache[split].append({"name": ip.name, "image": image, "mask": mask,
                                 "image_sha256": hashlib.sha256(b).hexdigest(), "mask_sha256": mask_sha})
    return cache

def metric(name, true, pred):
    cm = np.bincount((4*true.astype(np.int64)+pred).ravel(), minlength=16).reshape(4,4)
    ignore, zero = kd.image_iou(true,pred,"ignore"), kd.image_iou(true,pred,"zero")
    row = {"image":name,"miou_ignore":ignore["miou"],"miou_zero":zero["miou"]}
    for c, slug in enumerate(CLASS_SLUGS):
        row[slug+"_iou"] = ignore["class_iou"][c]
        row[slug+"_fp"] = int(cm[:,c].sum()-cm[c,c])
        row[slug+"_fn"] = int(cm[c].sum()-cm[c,c])
    return row

def blend_compare(cache, baseline, ni_export, eu_export, multiclass_exports, weights, output_dir, absent_policy="ignore"):

    weights = np.asarray(weights,dtype=np.float32)
    if weights.shape != (4,) or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError("Invalid baseline weights")
    if absent_policy not in ("ignore","zero"):
        raise ValueError("Invalid absent policy")
    output_dir = Path(output_dir)
    identity = {"baseline": [e["manifest_sha256"] for e in baseline],
                "ni": ni_export["manifest_sha256"] if ni_export else None,
                "eu": eu_export["manifest_sha256"] if eu_export else None,
                "multiclass": {n:e["manifest_sha256"] for n,e in multiclass_exports.items()},
                "weights":weights.tolist(),"absent_policy":absent_policy,"source_sha256":file_sha(__file__)}
    run_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    output_dir = output_dir / run_hash
    output_dir.mkdir(parents=True,exist_ok=True)
    atomic_json(output_dir/"identity.json",identity)
    variants = {"baseline":(False,False)}
    if ni_export: variants["replace_Ni"]=(True,False)
    if eu_export: variants["replace_Eu"]=(False,True)
    if ni_export and eu_export: variants["replace_Ni_Eu"]=(True,True)
    candidates = {n:{"kind":"specialists","variant":n} for n in variants}
    for name in multiclass_exports:
        candidates[name]={"kind":"multiclass","models":{name:1.}}
    names=list(multiclass_exports)
    if len(names)>1:
        candidates["new_models_mean"]={"kind":"multiclass","models":{n:1./len(names) for n in names}}
    mixes=list(candidates.items())
    for specialist in variants:
        for name, definition in mixes:
            if definition["kind"] == "multiclass":
                for alpha in (.25,.5,.75):
                    candidates[f"{specialist}+{name}@{alpha}"]={"kind":"blend","variant":specialist,
                        "models":definition["models"],"alpha":alpha}
    all_rows={name:[] for name in candidates}

    def stacks(split,rec):
        base=np.stack([read_probability(e,split,rec) for e in baseline])
        ni=read_probability(ni_export,split,rec) if ni_export else None
        eu=read_probability(eu_export,split,rec) if eu_export else None
        special={}
        for n,(use_ni,use_eu) in variants.items():
            score=base.copy()
            if use_ni: score[1]=ni
            if use_eu: score[2]=eu
            score*=weights[:,None,None]

            special[n]=score/score.sum(axis=0,keepdims=True).clip(min=1e-8)
        multi={n:read_probability(e,split,rec) for n,e in multiclass_exports.items()}
        return special,multi

    def predict(definition,special,multi):
        if definition["kind"]=="specialists":
            return special[definition["variant"]]
        joint=sum(weight*multi[name] for name,weight in definition["models"].items())
        if definition["kind"]=="multiclass": return joint
        alpha=definition["alpha"]
        return (1-alpha)*special[definition["variant"]]+alpha*joint

    for rec in cache["valid"]:
        special,multi=stacks("valid",rec)
        for name,definition in candidates.items():
            pred=predict(definition,special,multi).argmax(axis=0).astype(np.uint8)
            all_rows[name].append(metric(rec["name"],rec["mask"],pred))
        print("비교:",rec["name"],flush=True)
    ranking=[]
    for name,rows in all_rows.items():
        row={"candidate":name,"miou_ignore":float(np.mean([r["miou_ignore"] for r in rows])),
             "miou_zero":float(np.mean([r["miou_zero"] for r in rows]))}
        for slug in CLASS_SLUGS:
            values=[r[slug+"_iou"] for r in rows if r[slug+"_iou"] is not None]
            row[slug+"_iou"]=float(np.mean(values)) if values else None
        ranking.append(row)
    baseline_score=ranking[0]["miou_"+absent_policy]
    for row in ranking: row["delta_vs_baseline"]=row["miou_"+absent_policy]-baseline_score
    ranking.sort(key=lambda r:r["miou_"+absent_policy],reverse=True)
    selected=ranking[0]["candidate"]
    write_csv(output_dir/"candidate_ranking.csv",ranking)
    write_csv(output_dir/"validation_selected_per_image.csv",all_rows[selected])
    atomic_json(output_dir/"selection.json",{"selected":selected,"definition":candidates[selected],"ranking":ranking,
                "note":"Selected on valid20; this is not an independent test performance estimate. Sigmoid scores are normalized, not calibrated."})
    for split in ("valid","test"):
        paths=[]
        for rec in cache[split]:
            special,multi=stacks(split,rec)
            labels=predict(candidates[selected],special,multi).argmax(axis=0).astype(np.uint8)
            path=output_dir/("valid_selected_masks" if split=="valid" else "candidate_test_masks")/rec["name"].replace("_image.png","_mask.png")
            kd.save_rgb_mask(labels,path)
            paths.append(path)
        if split=="test":
            with zipfile.ZipFile(output_dir/"candidate_test_masks.zip","w",zipfile.ZIP_DEFLATED) as z:
                for path in paths:z.write(path,arcname=path.name)
    print("선택:",selected,"mIoU:",ranking[0]["miou_"+absent_policy],"기존 대비:",ranking[0]["delta_vs_baseline"])
    print("결과:",output_dir)
    return output_dir,ranking
