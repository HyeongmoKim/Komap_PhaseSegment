'YOLO 시작 가중치 검증과 SizeBalanced 학습 지원.'
from __future__ import annotations
import gc
import hashlib
import json
import os
import zipfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import komap_specialist_data as kd
import komap_phase_fusion as pf
import komap_nieu_zoom_ram as core
import komap_metal_methods_ram as metal

CURRENT_SHA = 'c5c85272700c0d663e10789ccffef626c6a70fed92979df68da53d2e6cc93822'
CURRENT_EPOCH = 26
CURRENT_MIOU = .7993595212707281
OLD_MIOU = .7959705202429518


def discover_current_export(project_root, run_name=None, expected_sha=CURRENT_SHA):
    runs = Path(project_root)/'runs'
    if run_name:
        export = pf.load_export(pf.contained_path(runs, run_name))
        if export['manifest'].get('checkpoint_sha256') != expected_sha:
            raise ValueError('지정한 run의 checkpoint SHA가 확인된 현재 YOLO와 다릅니다: '+str(export['run_dir']))
        return export
    matches = []
    for path in sorted(runs.rglob('export_index.json')):
        try:
            export = pf.load_export(path.parent)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if export['manifest'].get('checkpoint_sha256') == expected_sha:
            matches.append(export)
    if len(matches) != 1:
        names = [e['run_dir'].relative_to(runs).as_posix() for e in matches]
        raise ValueError(f'현재 YOLO 완료 export를 하나로 식별해야 합니다. CURRENT_YOLO_RUN_NAME에 지정하세요. 후보={names}. '
                         '0개이면 원래 YOLO 노트북의 export 셀부터 실행하세요. analysis_bundle만으로 확률을 복원하지 않습니다.')
    return matches[0]


def resolve_current_yolo(project_root, cache, report, run_name=None,
                         expected_sha=CURRENT_SHA, expected_epoch=CURRENT_EPOCH):
    export = discover_current_export(project_root, run_name, expected_sha)
    core.validate_export_dataset(export, cache, report)
    manifest = export['manifest']
    if manifest.get('architecture') != 'yolo11_edge_semantic' or manifest.get('target_class') != -1:
        raise ValueError('현재 기준은 YOLO11 edge semantic 4클래스 모델이어야 합니다.')
    if manifest.get('checkpoint_epoch') != expected_epoch or manifest.get('tta_rotations') != 4:
        raise ValueError('현재 기준 epoch/TTA가 다릅니다.')
    path = export['run_dir']/'best_solo.pt'
    if not path.is_file() or core.file_sha(path) != expected_sha:
        raise ValueError('현재 기준 best_solo.pt 파일이 없거나 전체 SHA가 다릅니다: '+str(path))
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('epoch') != expected_epoch or state.get('selection') != 'solo':
        raise ValueError('현재 체크포인트가 확인된 solo e026이 아닙니다.')
    config = state.get('config', {})
    if config.get('architecture') != 'yolo11_edge_semantic' or config.get('decoder_p1', False):
        raise ValueError('현재 체크포인트 decoder 구조가 다릅니다.')
    if state.get('identity', {}).get('data_sha256') != manifest.get('identity', {}).get('data_sha256'):
        raise ValueError('현재 체크포인트와 export의 데이터 identity가 다릅니다.')
    info = {'path':path, 'sha256':expected_sha, 'source_epoch':expected_epoch,
            'source_config':dict(config), 'source_export':export}
    del state
    gc.collect()
    print('확인된 현재 YOLO:',path, 'epoch',expected_epoch, 'SHA',expected_sha)
    return info


def current_yolo_context(cache, report, info):
    export = info['source_export']
    core.validate_export_dataset(export,cache,report)
    probabilities = {s:{} for s in ('valid','test')}
    for split in probabilities:
        for rec in cache[split]:
            p = core.checked_probability(pf.read_probability(export,split,rec), rec['image'].shape[:2])
            p.setflags(write=False)
            probabilities[split][rec['name']] = p
    summary,rows = metal.score_maps(cache['valid'],probabilities['valid'])
    return {'identity':{'checkpoint_sha256':info['sha256'],'checkpoint_epoch':info['source_epoch'],
                         'export_manifest_sha256':export['manifest_sha256'],'tta_rotations':4,
                         'data_sha256':report['data_sha256'],'probability':'raw YOLO softmax'},
            'probabilities':probabilities,'validation':summary,'rows':rows}


def current_reference(cache, raw, old, expected_miou=CURRENT_MIOU, tolerance=1e-8):
    probabilities = {s:{} for s in ('valid','test')}
    for split in probabilities:
        for rec in cache[split]:
            p = core.checked_probability(.5*raw['probabilities'][split][rec['name']]+
                                         .5*old['probabilities'][split][rec['name']],rec['image'].shape[:2])
            p.setflags(write=False)
            probabilities[split][rec['name']] = p
    summary,rows = metal.score_maps(cache['valid'],probabilities['valid'])
    if expected_miou is not None and abs(summary['miou_ignore']-expected_miou)>tolerance:
        raise ValueError(f"현재 기준 점수 불일치: {summary['miou_ignore']:.12f} vs {expected_miou:.12f}. 원본 export/결합을 확인하세요.")
    print('현재 최고 조합 재현:',summary['miou_ignore'], '(ignore)',summary['miou_zero'],'(zero)')
    return {'identity':{'raw_yolo':raw['identity'],'old_baseline':old['identity'],
                         'yolo_weight':.5,'description':'YOLO50 + DINO25 + Eu-replaced specialists25'},
            'probabilities':probabilities,'validation':summary,'rows':rows}


def experiment_config(info, seed=2026, **overrides):
    values = dict(info['source_config'])

    values.update(epochs=60,eval_every=2,freeze_encoder_epochs=0,seed=seed,
                  architecture='yolo11_edge_semantic',decoder_p1=False,detail_channels=16,
                  eu_component_balanced=False,fixed_new_model_weight=.25)
    values.update(overrides)
    c = metal.Config(**values)
    c.validate()
    return c


def read_final_probability(export, split, rec):
    row = export['rows'][(split,rec['name'])]
    if row['image_sha256'] != rec['image_sha256'] or row['mask_sha256'] != rec['mask_sha256']:
        raise ValueError('Seed export input bytes differ')
    path = pf.contained_path(export['directory'],row['final_blend_probability_file'])
    if core.file_sha(path) != row['final_blend_probability_sha256']:
        raise ValueError('Seed final probability file SHA differs')
    with np.load(path,allow_pickle=False) as z:
        return core.checked_probability(z['probabilities'].astype(np.float32),rec['image'].shape[:2])


def export_seed_ensemble(cache, report, output_run_dir, seed_run_dirs, old, reference, absent_policy='ignore'):

    if absent_policy not in ('ignore','zero') or not seed_run_dirs:
        raise ValueError('Invalid seed ensemble settings')
    if len({str(Path(path).resolve()) for path in seed_run_dirs}) != len(seed_run_dirs):
        raise ValueError('Seed run directories must be unique')
    exports = [pf.load_export(path) for path in seed_run_dirs]
    seeds = [e['manifest'].get('config',{}).get('seed') for e in exports]
    if all(seed is not None for seed in seeds) and len(set(seeds)) != len(seeds):
        raise ValueError('Seed exports must have different random seeds')
    for export in exports:core.validate_export_dataset(export,cache,report)
    identity = {'data_sha256':report['data_sha256'],'seeds':[e['manifest_sha256'] for e in exports],
                'reference':reference['identity'],'old_baseline':old['identity'],
                'absent_policy':absent_policy,'source_sha256':core.file_sha(__file__),
                'mix_weights':[.25,.5,1.0],'raw_mean_weights':'uniform'}
    digest = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(output_run_dir)
    destination = run_dir/'exports'/('seed_comparison_'+digest)
    if (destination/'manifest.json').exists():
        m = json.loads((destination/'manifest.json').read_text(encoding='utf-8'))
        if m['seed_comparison_identity'] != identity:raise ValueError('Seed export identity changed')
        core.verify_scale_export(destination,m,cache)
        core.atomic_json(run_dir/'export_index.json',{'manifest':(destination/'manifest.json').relative_to(run_dir).as_posix()})
        return destination
    if destination.exists():raise ValueError('Incomplete seed export; use another run name')
    maps = {'current_reference':reference['probabilities']['valid']}
    raw_mean = {}
    for rec in cache['valid']:
        raw_mean[rec['name']] = sum(pf.read_probability(e,'valid',rec)/len(exports) for e in exports)
    for i,export in enumerate(exports):
        maps[f'seed_{i+1}_selected'] = {r['name']:read_final_probability(export,'valid',r) for r in cache['valid']}
    for beta in (.25,.5,1.):
        maps[f'raw_seed_mean@{beta:g}'] = {r['name']:(1-beta)*old['probabilities']['valid'][r['name']]+beta*raw_mean[r['name']] for r in cache['valid']}
    summaries,rows = {},{}
    chosen = 'current_reference'
    best = reference['validation']['miou_'+absent_policy]
    for name,p in maps.items():
        summaries[name],rows[name] = metal.score_maps(cache['valid'],p)
        score = summaries[name]['miou_'+absent_policy]
        if score>best+1e-8:chosen,best = name,score
    ranking = [{'candidate':name,'selected':int(name==chosen),'miou_ignore':s['miou_ignore'],
                'miou_zero':s['miou_zero'],'delta_vs_current':s['miou_ignore']-reference['validation']['miou_ignore'],
                **{k+'_iou':v for k,v in s['class_iou_ignore'].items()}} for name,s in summaries.items()]
    ranking.sort(key=lambda r:r['miou_'+absent_policy],reverse=True)
    stage = destination.with_name(destination.name+'.pending')
    if stage.exists():raise ValueError('Incomplete pending export; inspect and change run name')
    stage.mkdir(parents=True)
    files = []
    for split in ('valid','test'):
        for rec in cache[split]:
            if chosen=='current_reference':p=reference['probabilities'][split][rec['name']]
            elif chosen.startswith('seed_'):
                index=int(chosen.split('_')[1])-1
                p=read_final_probability(exports[index],split,rec)
            else:
                beta=float(chosen.split('@')[1])
                mean=(raw_mean[rec['name']] if split=='valid' else sum(pf.read_probability(e,split,rec)/len(exports) for e in exports))
                p=(1-beta)*old['probabilities'][split][rec['name']]+beta*mean
            files.append(core.write_probability_pair(stage,split,rec,p,p,1.))
        core.zip_final_masks(stage,split)
    core.write_csv(stage/'candidate_ranking.csv',ranking)
    core.write_csv(stage/'validation_selected_per_image.csv',rows[chosen])
    core.write_csv(stage/'validation_candidates_per_image.csv',[{'candidate':k,**r} for k,v in rows.items() for r in v])
    core.atomic_json(stage/'validation_summary.json',{'selected_candidate':chosen,'selected':summaries[chosen],
        'current_reference':reference['validation'],'candidates':summaries,'absent_policy':absent_policy,
        'note':'Selected on valid20. This export stores final selected probabilities in both probability fields.'})
    core.atomic_json(stage/'export_settings.json',identity)
    with zipfile.ZipFile(stage/'analysis_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.glob('*.csv')):z.write(p,arcname=p.name)
        for name in ('validation_summary.json','export_settings.json'):z.write(stage/name,arcname=name)
        for p in sorted((stage/'selected_valid_masks').glob('*.png')):z.write(p,arcname='selected_valid_masks/'+p.name)
    artifacts=[{'path':p.relative_to(stage).as_posix(),'sha256':core.file_sha(p)} for p in sorted(stage.rglob('*')) if p.is_file()]
    manifest={'complete':True,'format_version':2,'model_family':core.MODEL_FAMILY,'target_class':-1,
              'architecture':'yolo_seed_selection','class_names':list(kd.CLASS_NAMES),'palette':kd.PALETTE.tolist(),
              'identity':{'data_sha256':report['data_sha256']},'tta_rotations':4,'dtype':'float32',
              'probability':'selected final four-class probabilities, not a raw checkpoint export',
              'seed_comparison_identity':identity,'selected_candidate':chosen,'files':files,'artifacts':artifacts}
    core.atomic_json(stage/'manifest.json',manifest)
    core.verify_scale_export(stage,manifest,cache)
    os.replace(stage,destination)
    core.atomic_json(run_dir/'export_index.json',{'manifest':(destination/'manifest.json').relative_to(run_dir).as_posix()})
    print('3 seed 평균/단일 seed/현재 조합 비교:',chosen,best)
    print('결과:',destination)
    return destination
