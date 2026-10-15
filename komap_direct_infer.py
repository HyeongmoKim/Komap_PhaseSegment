"""test.zip 또는 이미지 폴더에서 6개 모델을 직접 추론해 마스크를 저장."""
from pathlib import Path, PurePosixPath
import argparse
import gc
import io
import json
import tempfile
import zipfile
import numpy as np
from PIL import Image
import komap_phase_fusion as pf
import komap_specialist_data as kd
import komap_submission_infer as fixed

ROLES = ('primary', 'ni', 'eu', 'al', 'dino', 'yolo')
TARGETS = {'primary': 0, 'ni': 1, 'eu': 2, 'al': 3, 'dino': -1}
RUN_NAMES = {
    'yolo': 'yolo_eu_03_sizebalanced_20261011_seed2026',
    'dino': 'dinov3_hard_ablation_control_01',
    'primary': 'specialist_primary_fast_01',
    'ni': 'specialist_al3ni_fast_01',
    'eu': 'advanced_eu_balanced_ram_01',
    'al': 'specialist_al_fast_01',
}


def _mask_name(name):
    return name[:-10] + '_mask.png' if name.lower().endswith('_image.png') else Path(name).stem + '.png'


def _select_images(entries):
    selected = []
    for name, read in entries:
        path = PurePosixPath(name.replace('\\', '/'))
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('잘못된 이미지 경로: ' + name)
        if path.suffix.lower() != '.png' or any(part.lower() in ('masks', 'mask', 'labels', '__macosx') for part in path.parts):
            continue
        if path.name.lower().endswith('_mask.png') or path.name.startswith('.'):
            continue
        selected.append((path, read))
    image_dirs = [entry for entry in selected if any(part.lower() == 'images' for part in entry[0].parts[:-1])]
    if image_dirs:
        selected = image_dirs
    named_images = [entry for entry in selected if entry[0].name.endswith('_image.png')]
    if named_images:
        selected = named_images
    return sorted(selected, key=lambda entry: str(entry[0]))


def load_test_images(source, expected_count=None):
    source = Path(source)
    archive = zipfile.ZipFile(source) if source.is_file() and source.suffix.lower() == '.zip' else None
    try:
        if archive is not None:
            entries = [(info.filename, lambda info=info: archive.read(info)) for info in archive.infolist() if not info.is_dir()]
        elif source.is_dir():
            entries = [(path.relative_to(source).as_posix(), path.read_bytes) for path in source.rglob('*') if path.is_file()]
        else:
            raise FileNotFoundError('test.zip 또는 이미지 폴더가 필요합니다: ' + str(source))
        selected = _select_images(entries)
        if not selected or (expected_count is not None and len(selected) != expected_count):
            raise ValueError(f'test 이미지 수가 다릅니다: {len(selected)}장, 기대값={expected_count}')
        if expected_count is not None and (type(expected_count) is not int or expected_count <= 0):
            raise ValueError('expected_count는 양의 정수여야 합니다.')
        names = [_mask_name(path.name) for path, _ in selected]
        if len(names) != len(set(names)):
            raise ValueError('중복 출력 파일명이 있습니다.')
        records = []
        import hashlib
        for (path, read), name in zip(selected, names):
            content = read()
            with Image.open(io.BytesIO(content)) as image:
                rgb = np.array(image.convert('RGB'), dtype=np.uint8)
            records.append({'name': path.name, 'output_name': name, 'image': rgb,
                            'image_sha256': hashlib.sha256(content).hexdigest()})
        return records
    finally:
        if archive is not None:
            archive.close()


def inspect_assets(run_paths, reference_mode=None, *, eu_manifest_sha256=None, checkpoint_names=None):
    if set(run_paths) != set(ROLES):
        raise ValueError('yolo, dino, primary, ni, eu, al run 폴더를 모두 지정하세요.')
    filenames = {role: ('best50.pt' if role == 'yolo' else 'best.pt') for role in ROLES}
    if checkpoint_names:
        if set(checkpoint_names) - set(ROLES):
            raise ValueError('알 수 없는 checkpoint role')
        filenames.update(checkpoint_names)
    assets = {}
    for role in ROLES:
        run = Path(run_paths[role]).resolve()
        checkpoint = pf.contained_path(run, filenames[role])
        if not checkpoint.is_file():
            raise FileNotFoundError(str(checkpoint))
        digest = pf.file_sha(checkpoint)
        manifest = manifest_digest = None
        if role != 'yolo':
            export = pf.load_export(run)
            manifest = export['manifest']
            manifest_digest = export['manifest_sha256']
            if manifest.get('target_class') != TARGETS[role]:
                raise ValueError(f'{role}: 선택 export의 target_class가 다릅니다.')
            if manifest.get('checkpoint_sha256') != digest:
                raise ValueError(f'{role}: 선택한 가중치와 같은 run의 export가 서로 다릅니다. 해당 가중치로 export를 다시 생성하세요.')
            tta = manifest.get('tta_rotations', 1)
            if type(tta) is not int or tta not in (1, 4):
                raise ValueError(f'{role}: 지원하지 않는 TTA 설정: {tta}')
        assets[role] = {'run': run, 'checkpoint': checkpoint, 'checkpoint_sha256': digest,
                        'manifest': manifest, 'manifest_sha256': manifest_digest}
    return assets


def _yolo_provider(asset):
    import torch
    import komap_metal_methods_ram as mm
    import komap_nieu_zoom_ram as core
    state = torch.load(asset['checkpoint'], map_location='cpu', weights_only=False)
    if pf.file_sha(asset['checkpoint']) != asset['checkpoint_sha256']:
        raise ValueError('YOLO 가중치 파일이 로딩 중 변경됐습니다.')
    c = mm.Config(**state['config'])
    c.validate()
    if c.architecture != 'yolo11_edge_semantic' or c.target_class != -1:
        raise ValueError('YOLO11 semantic 4클래스 checkpoint가 필요합니다.')
    backend = state.get('execution_backend', {})
    if 'selected' not in backend:
        raise ValueError('YOLO checkpoint에 execution_backend.selected가 없습니다.')
    amp_dtype = mm.apply_execution_backend(backend['selected'])
    model = mm.build_model(c, pretrained=False).to('cuda', memory_format=mm.model_memory_format(c))
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    gpu_input = mm.GPUInput(torch.device('cuda'), c)
    tta = c.validation_tta_rotations
    profile = {'checkpoint_sha256': asset['checkpoint_sha256'], 'patch': c.patch,
               'stride': c.eval_stride, 'overlap': 'gaussian', 'tta_rotations': tta,
               'saved_backend': backend['selected']}
    del state
    def predict(image):
        with torch.inference_mode():
            return core.predict_probability(model, image, c, gpu_input, amp_dtype, tta)
    return predict, profile


def _load_dino_state(model, saved):
    expected = model.state_dict()
    mapped = {}
    for key, tensor in saved.items():
        candidates = [key]
        if key.startswith('encoder.model.'):
            candidates.append('encoder.' + key[len('encoder.model.'):])
        elif key.startswith('encoder.'):
            candidates.append('encoder.model.' + key[len('encoder.'):])
        matches = [name for name in candidates if name in expected]
        if len(matches) != 1:
            raise RuntimeError(f'DINO weight name mismatch: {key}; matches={matches}')
        name = matches[0]
        if name in mapped or tensor.shape != expected[name].shape:
            raise RuntimeError(f'DINO duplicate or shape mismatch: {key} -> {name}')
        mapped[name] = tensor
    missing = set(expected) - set(mapped)
    if missing:
        raise RuntimeError(f'DINO missing weights: {sorted(missing)[:10]}')
    model.load_state_dict(mapped, strict=True)


def _make_provider(role, asset):
    if role == 'yolo':
        return _yolo_provider(asset)
    import torch
    state = torch.load(asset['checkpoint'], map_location='cpu', weights_only=False)
    if pf.file_sha(asset['checkpoint']) != asset['checkpoint_sha256']:
        raise ValueError(role + ': 읽는 동안 가중치 파일이 바뀌었습니다.')
    manifest = asset['manifest']
    if state.get('config') != manifest.get('config'):
        raise ValueError(role + ': 가중치와 manifest의 모델 설정이 다릅니다.')
    profile = {'checkpoint_sha256': asset['checkpoint_sha256'], 'manifest_sha256': asset['manifest_sha256']}
    if 'patch_size' in state['config']:
        import komap_specialist as model_code
        config = model_code.Config(**state['config'])
        config.validate()
        if role == 'dino' or config.target_class != TARGETS[role]:
            raise ValueError(role + ': 원본 specialist 추론 설정이 다릅니다.')
        model = model_code.UNet(config.base_channels, num_classes=1).to('cuda')
        model.load_state_dict(state['model'], strict=True)
        profile.update(patch=config.patch_size, stride=config.stride, overlap='uniform', tta_rotations=1, amp_dtype='float16')

        def predict(image):
            return model_code.predict_probability(model, image, config, torch.device('cuda'))
    else:
        if role == 'dino':
            import komap_ablation_ram as model_code
        else:
            import komap_efficient_ram as model_code
        config = model_code.Config(**state['config'])
        config.validate()
        target = TARGETS[role]
        architecture = 'dinov3' if role == 'dino' else 'specialist'
        patch, stride = config.patch, config.eval_stride
        if config.architecture != architecture or config.target_class != target or config.patch != patch or config.eval_stride != stride:
            raise ValueError(role + ': 원본 타일/모델 구조가 다릅니다.')
        model = model_code.build_model(config, pretrained=False, backbone_config=state.get('backbone_config')).to('cuda', memory_format=torch.channels_last)
        if role == 'dino':
            _load_dino_state(model, state['model'])
        else:
            model.load_state_dict(state['model'], strict=True)
        model.eval()
        gpu_input = model_code.GPUInput(torch.device('cuda'), normalize=role == 'dino')
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tta = manifest['tta_rotations']
        profile.update(patch=patch, stride=stride, overlap='gaussian', tta_rotations=tta, amp_dtype=str(amp_dtype))

        def predict(image):
            result = model_code.predict_probability(model, image, config, gpu_input, amp_dtype, tta)
            return result if role == 'dino' else result[0]
    del state
    return predict, profile


def check_dependencies():
    import importlib
    errors = []
    for name in ('torch', 'torchvision', 'numpy', 'scipy.ndimage', 'cv2', 'PIL.Image',
                 'timm', 'transformers', 'segmentation_models_pytorch', 'ultralytics.nn.tasks'):
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
    if errors:
        raise RuntimeError('추론 전 의존성 검사 실패:\n' + '\n'.join(errors))


def preflight_models(assets):
    import torch
    errors = []
    for role in ROLES:
        try:
            provider, _ = _make_provider(role, assets[role])
            del provider
            print(f'모델 로딩 검사 통과: {role}', flush=True)
        except Exception as exc:
            errors.append(f'{role}: {type(exc).__name__}: {exc}')
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    if errors:
        raise RuntimeError('이미지 처리 전 모델 검사 실패:\n' + '\n'.join(errors))


def predict_test(test_input, output_dir, run_paths, *, reference_mode='current_base', expected_count=None, eu_manifest_sha256=None, checkpoint_names=None):
    check_dependencies()
    import torch
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('새 출력 폴더를 지정하세요: ' + str(output))
    assets = inspect_assets(run_paths, checkpoint_names=checkpoint_names)
    records = load_test_images(test_input, expected_count)
    if not torch.cuda.is_available():
        raise RuntimeError('6개 가중치 직접 추론에는 CUDA GPU가 필요합니다.')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    preflight_models(assets)
    maps, profiles = {}, {}
    for role in ROLES:
        predict, profiles[role] = _make_provider(role, assets[role])
        probabilities = []
        with torch.inference_mode():
            for index, record in enumerate(records, 1):
                print(f'{role}: {index}/{len(records)} {record["name"]}', flush=True)
                probability = np.asarray(predict(record['image']), dtype=np.float32)
                height, width = record['image'].shape[:2]
                expected_shape = (4, height, width) if role in ('yolo', 'dino') else (height, width)
                if probability.shape != expected_shape or not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1.00001:
                    raise ValueError(role + ': 잘못된 예측 확률입니다.')
                probabilities.append(probability)
        maps[role] = probabilities
        del predict
        gc.collect()
        torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.komap_masks_', dir=output.parent) as temporary:
        stage = Path(temporary) / 'result'
        masks = stage / 'masks'
        masks.mkdir(parents=True)
        for index, record in enumerate(records):
            specialist = np.stack([maps[role][index] for role in ('primary', 'ni', 'eu', 'al')])
            specialist /= np.maximum(specialist.sum(axis=0, keepdims=True), np.float32(1e-8))
            probability = fixed.fixed_probability(maps['yolo'][index], maps['dino'][index], specialist)
            kd.save_rgb_mask(probability.argmax(axis=0).astype(np.uint8), masks / record['output_name'])
        with zipfile.ZipFile(stage / 'test_masks.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(masks.glob('*.png')):
                archive.write(path, arcname=path.name)
        report = {'images': len(records), 'reference_mode': 'selected_runs', 'mode': 'six_checkpoint_direct_inference',
                  'checkpoint_names': checkpoint_names,
                  'weights': {'yolo': .5, 'dino': .25, 'specialists': .25}, 'pair_bias': -.1,
                  'profiles': profiles, 'ground_truth_used': False, 'cached_probabilities_used': False,
                  'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                  'inputs': [{key: record[key] for key in ('name', 'output_name', 'image_sha256')} for record in records],
                  'classes': list(kd.CLASS_NAMES), 'palette': kd.PALETTE.tolist()}
        (stage / 'inference_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if output.exists():
            raise FileExistsError(str(output))
        stage.rename(output)
    return {**report, 'output_dir': str(output), 'masks_zip': str(output / 'test_masks.zip')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('komap_test_output'))
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--reference-mode', choices=('current_base', 'original'), default='current_base')
    parser.add_argument('--expected-count', type=int, default=None)
    parser.add_argument('--eu-manifest-sha256', help='Explicitly selected new Eu export manifest SHA256')
    parser.add_argument('--check-assets', action='store_true')
    args = parser.parse_args()
    runs = {role: args.runs / name for role, name in RUN_NAMES.items()}
    if args.check_assets:
        assets = inspect_assets(runs, args.reference_mode, eu_manifest_sha256=args.eu_manifest_sha256)
        print(json.dumps({role: {key: str(value) for key, value in asset.items() if key != 'manifest'} for role, asset in assets.items()}, ensure_ascii=False, indent=2))
    else:
        if args.input is None:
            parser.error('--input test.zip 또는 이미지 폴더를 지정하세요.')
        print(json.dumps(predict_test(args.input, args.output, runs, reference_mode=args.reference_mode, expected_count=args.expected_count, eu_manifest_sha256=args.eu_manifest_sha256), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
