'YOLO03 best50 검증과 source 고정.'
from __future__ import annotations
import gc
import json
import math
import os
import shutil
from pathlib import Path
import torch
import komap_phase_fusion as pf
import komap_nieu_zoom_ram as core
import komap_metal_methods_ram as metal


def inspect_best50(run_dir, report, old_baseline_identity=None, absent_policy='ignore',
                   require_eu_balanced=True):
    run_dir=Path(run_dir)
    path=run_dir/'best50.pt'
    if not path.is_file():raise FileNotFoundError('best50.pt가 없습니다: '+str(path))
    digest=core.file_sha(path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    if core.file_sha(path)!=digest:
        raise ValueError('읽는 도중 best50이 바뀌었습니다. 잠시 뒤 다시 실행하세요.')
    if not isinstance(state,dict) or not isinstance(state.get('model'),dict) or not state['model']:
        raise ValueError('잘못된 best50 checkpoint 구조')
    if state.get('selection')!='fixed50':raise ValueError('50% 결합으로 선택한 checkpoint가 아닙니다.')
    identity=state.get('identity',{})
    if identity.get('family')!=metal.EXPERIMENT_FAMILY:
        raise ValueError('새 YOLO Eu 개선 노트북 계열이 아닙니다.')
    if identity.get('data_sha256')!=report['data_sha256']:
        raise ValueError('현재 실제 데이터의 SHA와 checkpoint가 다릅니다.')
    config=state.get('config',{})
    c=metal.Config(**config); c.validate()
    if c.architecture!='yolo11_edge_semantic':raise ValueError('YOLO11 semantic checkpoint가 아닙니다.')
    if require_eu_balanced and (not c.eu_component_balanced or c.decoder_p1 or c.detail_channels!=16):
        raise ValueError('03번 SizeBalanced 구조/Eu sampling과 다른 checkpoint입니다.')
    if c.absent_policy!=absent_policy:
        raise ValueError('best50 선택 당시 absent policy가 다릅니다.')
    if config!=identity.get('config'):
        raise ValueError('checkpoint config와 training identity가 다릅니다.')
    if old_baseline_identity is not None and identity.get('baseline_context')!=old_baseline_identity:
        raise ValueError('현재 준비한 DINO/specialist 조합과 source의 조합이 다릅니다.')
    manifest=run_dir/'data_manifest.json'
    if not manifest.is_file():raise FileNotFoundError('source data_manifest.json이 필요합니다: '+str(manifest))
    stored=json.loads(manifest.read_text(encoding='utf-8'))
    if stored.get('data_sha256')!=report['data_sha256'] or stored.get('files')!=report.get('files'):
        raise ValueError('source의 실제 파일 fingerprints와 현재 train/valid/test가 다릅니다.')
    score=state.get('score')
    if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
        raise ValueError('잘못된 best50 score')
    summary=state.get('validation',{}).get('fixed50',{})
    if abs(summary.get('miou_'+absent_policy,-1)-score)>1e-10:
        raise ValueError('best50 score와 해당 검증 summary가 다릅니다.')
    epoch=state.get('epoch')
    if type(epoch) is not int or not 0<=epoch<=c.epochs:raise ValueError('잘못된 source epoch')
    backend=state.get('execution_backend')
    if not isinstance(backend,dict) or backend.get('selected') not in metal.backend_candidates():
        raise ValueError('source에 검증된 YOLO GPU backend가 없습니다.')
    backend_file=run_dir/'execution_backend.json'
    if not backend_file.is_file() or json.loads(backend_file.read_text(encoding='utf-8'))!=backend:
        raise ValueError('source의 checkpoint와 실행 backend 기록이 다릅니다.')
    info={'path':str(path.resolve()),'run_dir':str(run_dir.resolve()),'sha256':digest,
          'config':config,'epoch':epoch,'source_score':float(score),'source_policy':absent_policy,
          'source_validation':summary,'backend':backend,'identity':identity}
    del state;gc.collect()
    return info


def resolve_sizebalanced_source(project_root, report, run_name=None,
                                old_baseline_identity=None, absent_policy='ignore'):
    runs=(Path(project_root)/'runs').resolve()
    if absent_policy not in ('ignore','zero'):raise ValueError('Invalid absent policy')
    if run_name:
        source=pf.contained_path(runs,run_name)
        info=inspect_best50(source,report,old_baseline_identity,absent_policy)
        info['run_name']=source.relative_to(runs).as_posix()
        return info
    candidates=[];rejected=[]
    for path in sorted(runs.rglob('best50.pt')):

        if 'source_snapshot' in path.parts:continue
        try:
            info=inspect_best50(path.parent,report,old_baseline_identity,absent_policy)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            rejected.append({'run':str(path.parent.relative_to(runs)),'reason':str(exc)})
            continue
        info['run_name']=path.parent.relative_to(runs).as_posix()
        candidates.append(info)
    if not candidates:
        raise ValueError('03번 SizeBalanced best50을 찾지 못했습니다. SOURCE_RUN_NAME에 실제 run 이름을 지정하세요. '
                         '학습 중이라도 best50.pt/data_manifest.json/execution_backend.json이 저장되면 실행 가능합니다. '
                         '반려 후보='+json.dumps(rejected[:8],ensure_ascii=False))

    candidates.sort(key=lambda item:(-item['source_score'],item['run_name']))
    selected=dict(candidates[0])
    selected['candidate_inventory']=[{'run_name':i['run_name'],'epoch':i['epoch'],
                                       'score':i['source_score'],'sha256':i['sha256']} for i in candidates]
    print('03번 best50 후보:',selected['candidate_inventory'])
    print('선택한 source:',selected['run_name'],'epoch',selected['epoch'],'score',selected['source_score'])
    return selected


def verify_source_checkpoint(info):
    path=Path(info['path'])
    if not path.is_file() or core.file_sha(path)!=info['sha256']:
        raise ValueError('고정한 source checkpoint 파일이 바뀌거나 없어졌습니다.')
    return path


def freeze_source(project_root,report,run_dir,run_name=None,
                  old_baseline_identity=None,absent_policy='ignore'):
    'Existing run resumes its snapshot even if active training improves later.'
    run_dir=Path(run_dir)
    snapshot=run_dir/'source_snapshot';metadata=snapshot/'source.json'
    if metadata.exists():
        info=json.loads(metadata.read_text(encoding='utf-8'))
        info['path']=str((snapshot/'best50.pt').resolve())
        if run_name and info.get('run_name')!=run_name:
            raise ValueError('고정 source run을 바꾸려면 새 RUN_NAME을 사용하세요.')
        if info.get('source_policy')!=absent_policy or info['identity'].get('data_sha256')!=report['data_sha256']:
            raise ValueError('고정 source의 policy/data가 다릅니다. 새 RUN_NAME을 사용하세요.')
        checked=inspect_best50(snapshot,report,old_baseline_identity,absent_policy)
        if any(checked[k]!=info[k] for k in ('sha256','config','epoch','source_score','backend','identity')):
            raise ValueError('source snapshot metadata/파일이 다릅니다.')
        verify_source_checkpoint(info)
        print('이미 고정한 source 재사용:',info['run_name'],info['sha256'],info['source_score'])
        return info
    if snapshot.exists():raise ValueError('미완성 source_snapshot이 있습니다. 상태를 확인하고 새 RUN_NAME을 사용하세요.')
    info=resolve_sizebalanced_source(project_root,report,run_name,old_baseline_identity,absent_policy)
    run_dir.mkdir(parents=True,exist_ok=True)
    stage=run_dir/'source_snapshot.pending'
    if stage.exists():raise ValueError('미완성 snapshot.pending이 있습니다. 새 RUN_NAME을 사용하세요.')
    stage.mkdir()
    path=verify_source_checkpoint(info)
    shutil.copyfile(path,stage/'best50.pt')
    if core.file_sha(stage/'best50.pt')!=info['sha256']:
        raise ValueError('복사 도중 source가 바뀌었습니다. 새 RUN_NAME으로 다시 실행하세요.')
    core.atomic_json(stage/'data_manifest.json',report)
    core.atomic_json(stage/'execution_backend.json',info['backend'])
    info['original_path']=info['path']
    info['path']=str((snapshot/'best50.pt').resolve())
    core.atomic_json(stage/'source.json',info)
    os.replace(stage,snapshot)
    verify_source_checkpoint(info)
    print('source best50을 별도 snapshot으로 고정했습니다:',info['path'])
    return info
