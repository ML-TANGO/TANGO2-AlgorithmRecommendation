"""Build Stage 4 from accepted local generations and independent request audits."""
import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from .request_checks import inspect
from .expression_utils import gpu_identity
from .request_design import NON_TEXT
from ..io_utils import read, write, sha, digest, write_jsonl, backup_existing
from ..paths import STAGE2, STAGE3, STAGE4, PACKAGE_DIR, rel, resolve

ARTIFACTS = ['input_2.json', 'label_2.json', 'stage4_cases.json', 'generation_summary.json', 'split_manifest.json',
             'input_ambiguity_report.json', 'training_assumptions.json', 'source_manifest.json', 'validation_report.json', 'diversity_report.json']


def read_requests(folder=STAGE4):
    records = {}
    folder = Path(folder)
    for p in sorted(folder.glob('requests_*.jsonl')):
        for line in p.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r.get('accepted'):
                records[r['case_id']] = r
    # Explicit successful repairs supersede original passes regardless of file order.
    for p in sorted(folder.glob('repairs_*.jsonl')):
        for line in p.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r.get('accepted'):
                records[r['case_id']] = r
    return records


def read_audits(folder=STAGE4):
    audits = {}
    for p in sorted(Path(folder).glob('audits_*.jsonl')):
        for line in p.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r.get('assessment') is not None:
                audits[(r['case_id'], r['request_sha256'])] = r
    return audits


def semantic_group(case):
    visible = dict(task=case['task'], gpu=gpu_identity(case['gpu']), language=case['language'])
    if case['task'] not in NON_TEXT:
        visible['context_length'] = case['context_length']
    return digest(json.dumps(visible, sort_keys=True, ensure_ascii=False))


def answer_for(row, request, audit, stage2, profiles, shared):
    source = row['input_1']; case = source['case']; p = profiles.get(source['label']['model']); a = audit['assessment']; ctx = request['metadata']['context']
    gpu_name = gpu_identity(case['gpu']); ambiguous = gpu_name == 'A100'
    thinking = dict(recommended=a['thinking_recommended'], explicitly_requested=False,
        reason=a['thinking_reason'], request_evidence=a['evidence'],
        decision_method=audit['method'],
        model_supports_thinking=p.get('native_thinking', False) if p else None,
        capability_source='stage_2/model_profiles.json; not a new benchmark',
        activation_configured=False,
        note='권장 여부와 선택 모델의 지원 여부는 별개입니다. 요청에 필수 조건이 없다고 모델의 사고 능력이 없다는 뜻은 아닙니다.')
    assumptions = {
      'gpu_capacity': dict(value_gb=case['gpu']['vram_gb'], origin='source_case_catalog_assumption', user_requested=False,
          explicit_in_input=False, requires_confirmation=ambiguous,
          note='GPU 표기에 용량을 쓰지 않았습니다. 특히 A100은 후보에 여러 용량이 있어 이름만으로 특정할 수 없습니다.'),
      'system_execution_policy': dict(origin='system_calculation_policy', user_requested=False, reference='training_assumptions.json', values=copy.deepcopy(shared['workload'])),
      'context_interpretation': dict(origin='system_interpretation', user_requested=False, applicable=ctx['applicable'],
          budget_tokens=ctx.get('budget_tokens'), surface_style=ctx['style'], exact_recoverable=ctx['exact_recoverable'], tokenizer_measurement_performed=False),
    }
    if ctx['style'] == 'pages':
        assumptions['context_interpretation'].update(assumed_tokens_per_page=ctx['assumed_tokens_per_page'], pages=ctx['pages'],
            note='입력·출력 총량의 페이지 표현을 페이지당 512토큰으로 가정한 합성 데이터입니다. 언어·문서·tokenizer에 따라 달라지며 실제 inference에서 정확한 복원을 보장하지 않습니다.')
    if ctx['style'] == 'k':
        assumptions['context_interpretation']['k_multiplier'] = 1024
    if case['task'] == 'Translation':
        assumptions['translation_direction'] = dict(origin='stage2_calculation_policy', user_requested=False, source_language='ko' if case['language'] == 'en' else 'en', target_language=case['language'], note='요청은 언어 하나만 제시했습니다. 이 방향은 추천 검토의 가정이며 사용자가 정한 방향이 아닙니다.')
    if p:
        assumptions['selected_model_workload'] = dict(origin='system_calculation_policy', user_requested=False, values=copy.deepcopy(p['workload']))
    recommendation = None
    status = row['stage3_status']
    if row['output_1'] is not None:
        output = row['output_1']
        recommendation = dict(**copy.deepcopy(source['label']), hyperparameters=copy.deepcopy(output['hyperparameters']),
            training_plan=copy.deepcopy(output['training_plan']), backend_requirements=copy.deepcopy(output['backend_requirements']),
            recommendation_reason=copy.deepcopy(stage2['rationale']), parameter_rationale=copy.deepcopy(output['parameter_rationale']))
        # Decision is about user-visible task; keep the source recommendation
        # conditional rather than silently swapping model or training scope.
        if thinking['recommended'] and not thinking['model_supports_thinking']:
            status = 'needs_thinking_model_review'
        elif ambiguous:
            status = 'needs_gpu_capacity_clarification'
        else:
            status = 'conditional_recommendation_draft'
    return dict(status=status, recommendation=recommendation, thinking=thinking,
        review_reasons=(["입력의 A100 이름만으로 GPU 용량을 구분할 수 없어 확인이 필요합니다."] if ambiguous else []) + (["thinking 권장 판단과 기존 선택 모델의 공개 지원 범위를 함께 재검토해야 합니다."] if thinking['recommended'] and p and not thinking['model_supports_thinking'] else []),
        recommendation_assumptions=assumptions, reason=row['reason'],
        source_stage3_status=row['stage3_status'], empirically_validated=False, ready_to_execute=False)


def build(output_dir=STAGE4, stage3=STAGE3, stage2=STAGE2):
    output_dir = Path(output_dir); stage3 = Path(stage3); stage2 = Path(stage2)
    rows = read(stage3 / 'hyperparameter_cases.json')
    report = read(stage3 / 'validation_report.json')
    if report.get('valid') is not True or report['artifact_sha256']['hyperparameter_cases.json'] != sha(stage3 / 'hyperparameter_cases.json'):
        raise ValueError('Stage 3 verification missing/stale')
    stage3_summary = read(stage3 / 'generation_summary.json')
    for path, digest_value in stage3_summary['source_sha256'].items():
        if sha(resolve(path)) != digest_value:
            raise ValueError('Stage 3 source changed: ' + path)
    profiles = {p['model_id']: p for p in read(stage2 / 'model_profiles.json')}
    stage2_rows = {r['case_id']: r for r in read(stage2 / 'labeled_cases.json')}
    shared = read(stage3 / 'training_assumptions.json'); reqs = read_requests(output_dir); audits = read_audits(output_dir)
    missing = []; combined = []
    for row in rows:
        cid = row['case_id']; request = reqs.get(cid)
        if request is None:
            missing.append((cid, 'missing_request')); continue
        errors = inspect(request['text'], row['input_1']['case'], request['metadata'])
        audit = audits.get((cid, digest(request['text'])))
        if errors or audit is None or not audit['assessment']['faithful']:
            missing.append((cid, errors or ('missing_audit' if audit is None else audit['assessment']['issues']))); continue
        answer = answer_for(row, request, audit, stage2_rows[cid], profiles, shared)
        group = semantic_group(row['input_1']['case'])
        # Stable semantic-group split prevents non-applicable context/capacity
        # variants of the same visible request from leaking across splits.
        bucket = int(group[:8], 16) % 100
        split = 'test' if bucket < 5 else 'validation' if bucket < 10 else 'train'
        eligible = answer['status'] == 'conditional_recommendation_draft'
        combined.append(dict(case_id=cid, input_1=copy.deepcopy(row['input_1']), input_2=request['text'],
            label_2=answer, source_output_1=row['output_1'], source_stage3_status=row['stage3_status'],
            expression_metadata=request['metadata'], request_audit=audit['assessment'], semantic_group=group, split=split,
            supervised_recommendation_eligible=eligible,
            generation_provenance=dict(generator=request['generator'], model_revision=request.get('revision'),
                method=request['method'], prompt_uses_case_only=True, per_case_GPT_API_called=False)))
    if missing:
        raise ValueError(f'{len(missing)} requests unresolved; first: {missing[:8]}')
    groups = defaultdict(list)
    for r in combined:
        groups[r['semantic_group']].append(r)
    ambiguities = []
    for g, members in groups.items():
        if len(members) < 2:
            continue
        labels = {json.dumps(r['input_1']['label'], sort_keys=True) for r in members}
        ambiguities.append(dict(semantic_group=g, case_ids=[r['case_id'] for r in members], distinct_source_labels=len(labels),
            note='동일한 관측 조건의 그룹. 비적용 길이 또는 입력에서 생략한 GPU 용량 등이 원본에서 다를 수 있음'))
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_existing(output_dir, ARTIFACTS + ['training_data'], 'before_v4')
    data = {'input_2.json': [dict(case_id=r['case_id'], input_2=r['input_2']) for r in combined],
            'label_2.json': [dict(case_id=r['case_id'], label_2=r['label_2']) for r in combined],
            'stage4_cases.json': combined, 'training_assumptions.json': shared,
            'split_manifest.json': [dict(case_id=r['case_id'], semantic_group=r['semantic_group'], split=r['split'], eligible=r['supervised_recommendation_eligible']) for r in combined],
            'input_ambiguity_report.json': dict(groups=ambiguities, distinct_semantic_groups=len(groups), capacity_omission_ambiguous_gpu='A100')}
    for name, value in data.items():
        write(output_dir / name, value)
    train_dir = output_dir / 'training_data'; train_dir.mkdir(exist_ok=True)
    for split in ['train', 'validation', 'test']:
        write_jsonl(train_dir / (split + '.jsonl'), [dict(case_id=r['case_id'], input=r['input_2'], output=r['label_2']) for r in combined if r['split'] == split and r['supervised_recommendation_eligible']])
    sources = [stage3 / n for n in ['hyperparameter_cases.json', 'validation_report.json', 'generation_summary.json', 'training_assumptions.json']]
    sources += [stage2 / n for n in ['labeled_cases.json', 'model_profiles.json']]
    sources += sorted(output_dir.glob('requests_*.jsonl')) + sorted(output_dir.glob('repairs_*.jsonl')) + sorted(output_dir.glob('audits_*.jsonl'))
    here = PACKAGE_DIR / 'stage4_requests'
    sources += [here / n for n in ['request_design.py', 'request_checks.py', 'local_writer.py', 'local_auditor.py', 'audit_contract.py', 'assemble.py', 'expression_utils.py', 'validate.py', 'inspect_diversity.py']]
    if (output_dir / 'generation_backend.json').exists():
        sources.append(output_dir / 'generation_backend.json')
    manifest = {rel(p): dict(sha256=sha(p)) for p in sources}
    write(output_dir / 'source_manifest.json', manifest)
    summary = dict(version='4.0', total_cases=len(combined), source_stage3_version='3.0',
       generation_method='per_case_local_gemma4_generation_and_separate_request_audit; Codex-authored validation and assembly',
       per_case_GPT_API_calls=0, source_cases_and_labels_preserved=True,
       request_generation_method_counts=dict(Counter(r['generation_provenance']['method'] for r in combined)),
       thinking_decision_method_counts=dict(Counter(r['label_2']['thinking']['decision_method'] for r in combined)),
       recommendation_status_counts=dict(Counter(r['label_2']['status'] for r in combined)),
       thinking_recommended_counts=dict(Counter(str(r['label_2']['thinking']['recommended']).lower() for r in combined)),
       expression_counts=dict(Counter(r['expression_metadata']['context']['style'] for r in combined)),
       supervised_recommendation_records=sum(r['supervised_recommendation_eligible'] for r in combined),
       training_split_counts=dict(Counter(r['split'] for r in combined if r['supervised_recommendation_eligible'])),
       all_case_split_counts=dict(Counter(r['split'] for r in combined)),
       gpu_memory_visible_in_user_input=False, unused_modality_context_not_rendered=True,
       exact_page_to_token_recovery_claimed=False, empirical_training_performed=False,
       downstream_regeneration_required=['stage_5'], artifact_sha256={n: sha(output_dir / n) for n in data},
       training_data_sha256={str(p.relative_to(output_dir)): sha(p) for p in sorted(train_dir.glob('*.jsonl'))})
    write(output_dir / 'generation_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith('sha256')}, ensure_ascii=False, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=STAGE4)
    p.add_argument('--stage3-dir', type=Path, default=STAGE3)
    p.add_argument('--stage2-dir', type=Path, default=STAGE2)
    args = p.parse_args(argv)
    return build(args.output_dir, args.stage3_dir, args.stage2_dir)


if __name__ == '__main__':
    main()
