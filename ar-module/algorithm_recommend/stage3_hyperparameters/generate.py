"""Stage 3 v3.0: strict catalog hyperparameters and modality-aware training plans."""
import argparse
import contextlib
import io
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from .training_scopes import build_training_plan
from .context_lengths import load_context_lengths, derive_sequence_length
from .architecture import module_spec, select_modules, memory_check
from ..io_utils import read, write, sha, canon, backup_existing
from ..paths import DATASET_DIR, SOURCE_CONFIGS, SOURCE_EVIDENCE, STAGE1, STAGE2, STAGE3, PACKAGE_DIR, rel

LEGACY = {'GTX 1060', 'GTX 1070', 'GTX 1080 Ti', 'RTX 2060', 'RTX 2070', 'RTX 2080 Ti', 'T4', 'V100'}
VERSION = '3.0'


def propose(row, catalog, profiles, contexts):
    status = row['training_status']
    out = dict(case_id=row['case_id'], output_1=None, stage3_status='pending_stage2_review' if status == 'needs_review' else 'skipped_stage2_not_possible',
               source_stage2_status=status, reason_codes=row['reason_codes'].copy(), reason=row.get('review_reason') or row['impossibility_reason'],
               training_assumptions_ref='training_assumptions.json', empirically_validated=False, ready_to_execute=False,
               generation_method='codex_authored_rules_no_per_case_llm_call')
    if status != 'possible':
        out['source_review_notes'] = row['review_notes']
        return out
    case = row['case']; label = row['label']; algo = label['algorithm']; p = profiles[label['model']]
    sequence = derive_sequence_length(case, p, contexts)
    seq = sequence['max_seq_length'] or 0
    specialist = p['profile_type'] == 'specialist'
    dtype = 'float16' if case['gpu']['gpu_name'] in LEGACY else 'bfloat16'
    lr = (5e-6 if case['task'] == 'TTS' else 1e-5 if sequence['applicable'] else 5e-5) if algo == 'SFT' else (5e-5 if specialist or seq >= 32768 else 1e-4)
    if algo != 'SFT' and case['task'] == 'Classification' and p['params_b_approx'] <= 4:
        lr = 2e-4
    accum = 8 if specialist else 16 if seq <= 4096 else 8 if seq <= 8192 else 4 if seq <= 32768 else 1
    hp = dict(learning_rate=lr, per_device_train_batch_size=1, gradient_accumulation_steps=accum, num_train_epochs=3,
              warmup_ratio=.05 if seq >= 32768 else .03, weight_decay=.01, max_grad_norm=.5 if case['task'] == 'TTS' else 1.,
              lr_scheduler_type='cosine', optimizer='AdamW', training_dtype=dtype)
    spec = None; attempts = []
    if algo in {'LoRA', 'QLoRA'}:
        rank = 8 if case['gpu']['vram_gb'] <= 16 or seq >= 32768 else 16
        targets = ['all-linear'] if algo == 'QLoRA' else ['q_proj', 'v_proj'] if rank == 8 else ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        hp.update(lora_r=rank, lora_alpha=2 * rank, lora_dropout=.05, target_modules=targets)
        spec = module_spec(label['model'])
    if algo == 'QLoRA':
        hp.update(bits=4, quant_type='nf4', double_quant=True, bnb_4bit_compute_dtype=dtype)
    if spec:
        for rank in sorted([r for r in catalog[algo]['lora_r'] if r <= hp['lora_r'] and 2 * r in catalog[algo]['lora_alpha']], reverse=True):
            hp.update(lora_r=rank, lora_alpha=2 * rank)
            memory = memory_check(row, p, hp, spec)
            attempts.append(dict(lora_r=rank, estimated_memory_gb=memory['estimated_memory_gb'], within_budget=memory['within_budget']))
            if memory['within_budget']:
                break
    else:
        memory = memory_check(row, p, hp)
    if set(hp) != set(catalog[algo]):
        raise ValueError('Hyperparameter schema drift')
    for k, v in hp.items():
        if canon(v) not in [canon(x) for x in catalog[algo][k]]:
            raise ValueError(f'{algo}.{k} outside catalog')
    plan = build_training_plan(case, p, algo)
    plan['context_length_applicable'] = sequence['applicable']; plan['sequence_length'] = sequence
    plan['assumptions_origin'] = 'system_calculation_policy_not_user_requirements'
    plan['adapter_targets'] = None
    if spec:
        chosen = select_modules(spec, hp['target_modules'])
        plan['adapter_targets'] = dict(scope_prefix=spec['scope_prefix'], target_suffixes=hp['target_modules'], expected_module_count=len(chosen),
             module_spec_ref='module_specs.json', module_spec_key=label['model'], sample_module_paths=[x['name'] for x in chosen[:7]],
             all_original_weights_frozen=True, runtime_verified=False)
    backend = dict(max_seq_length=sequence['max_seq_length'], sequence_length_source='case.context_length',
         context_candidate_source='dataset/context_length_list.json', effective_batch_size=accum,
         requires_fp16_loss_scaling=dtype == 'float16', gradient_checkpointing_required=True,
         gpu_count_assumption_ref='training_assumptions.json', packing=False, pad_to_max_length=False,
         non_text_token_allowance=p['extra_tokens'], training_objective=p['objective'],
         memory_efficient_attention_and_chunked_loss_required=sequence['applicable'],
         runtime_checks=['Verify dtype and memory-efficient attention support on actual GPU',
           'Compare named_modules and tensor shapes with module_specs before adapter injection',
           'Freeze base weights then inject adapters only inside approved scope; reject unexpected trainable parameters',
           'Check task-specific processor, targets and loss, and finite gradients in a forward/backward batch',
           'Measure memory at requested maximum workload without silent truncation'], launchable_configuration=False)
    if spec:
        backend['adapter_injection'] = 'Use resolve_and_verify_adapter_targets with the actual supported linear named_modules; use returned anchored PEFT regex, never global all-linear.'
    if algo == 'QLoRA':
        backend.update(prepare_model_for_kbit_training_required=True,
            quantization_policy='Quantize only linear weights in declared language blocks; verify converted names against module_specs. Embeddings, head and other modalities stay unquantized.',
            non_quantized_weight_budget_dtype='float32', quantized_base_weights_trainable=False,
            modules_to_save=plan['modules_to_save'])
    payload = dict(hyperparameters=hp, training_plan=plan, backend_requirements=backend,
         parameter_rationale=dict(learning_rate='Task·algorithm·길이에 따른 검증 전 시작값; 실험으로 최적화한 값이 아님',
            batch='단일 GPU microbatch 1의 계산 가정을 유지하고 누적으로 유효 배치 조절',
            length='독립 후보를 추가하지 않고 context_length_list에서 정한 case 길이 사용; 비텍스트 task는 비적용',
            precision='구형 GPU는 float16, 그 외 bfloat16 후보; 실제 지원·수치 안정성 확인 필요',
            epochs='데이터 규모 미제공 상태의 3 epoch 초안; 검증 성능으로 조정',
            adapter='허용된 scope 내부에서만 adapter 적용; 후보 rank를 낮추어 같은 label의 메모리 예산 재검토'))
    out['resource_check'] = memory; out['rank_search'] = attempts
    if not memory['within_budget']:
        out.update(stage3_status='needs_resource_review', reason_codes=['scoped_memory_estimate_exceeds_budget'],
            reason='양자화 제외 가중치·adapter 상태를 반영한 추정이 예산을 넘습니다. 원본 case·label은 유지하며 학습 불가능을 확정하지 않습니다.',
            review_candidate=payload)
    else:
        out.update(stage3_status='recommended_draft', output_1=payload, reason_codes=[],
                   reason='후보 목록과 모듈 범위·계산 예산을 통과한 초기 설정입니다. 실제 학습 성공이나 최적값을 검증한 것은 아닙니다.')
    return out


def sources(stage2=STAGE2, dataset_dir=DATASET_DIR):
    files = [Path(stage2) / n for n in ['labeled_cases.json', 'model_profiles.json', 'labeling_summary.json', 'labeling_policy.json', 'training_assumptions.json', 'validation_report.json']]
    files += [Path(dataset_dir) / n for n in ['hyper_parameter_list.json', 'context_length_list.json']]
    files += sorted(SOURCE_CONFIGS.glob('*.json'))
    here = PACKAGE_DIR / 'stage3_hyperparameters'
    files += [here / n for n in ['generate.py', 'training_scopes.py', 'architecture.py', 'context_lengths.py', 'validate.py']]
    files += sorted(p for p in SOURCE_EVIDENCE.glob('*') if p.is_file())
    return files


def generate(output_dir=STAGE3, stage2=STAGE2, stage1=STAGE1, dataset_dir=DATASET_DIR):
    # Validate Stage 2 and its provenance in memory; do not rewrite upstream reports.
    from ..stage2_labels.validate import validate as validate_stage2
    stage2 = Path(stage2)
    with tempfile.TemporaryDirectory(prefix='stage3_check_stage2_') as temp:
        for p in stage2.glob('*.json'):
            shutil.copy2(p, Path(temp) / p.name)
        with contextlib.redirect_stdout(io.StringIO()):
            validate_stage2(Path(temp), stage1, dataset_dir)
    paths = sources(stage2, dataset_dir); before = {rel(p): sha(p) for p in paths}
    rows = read(stage2 / 'labeled_cases.json'); profiles = {p['model_id']: p for p in read(stage2 / 'model_profiles.json')}
    catalog = read(Path(dataset_dir) / 'hyper_parameter_list.json'); contexts = load_context_lengths(Path(dataset_dir) / 'context_length_list.json')
    inputs = [dict(case_id=r['case_id'], input_1=dict(case=r['case'], label=r['label'])) for r in rows]
    outputs = [propose(r, catalog, profiles, contexts) for r in rows]
    combined = [dict(**out, input_1=inp['input_1']) for out, inp in zip(outputs, inputs)]
    selected_models = sorted({r['label']['model'] for r in rows if r['label']['algorithm'] in {'LoRA', 'QLoRA'}})
    assumptions = read(stage2 / 'training_assumptions.json')
    assumptions.update(source=rel(stage2 / 'training_assumptions.json'), source_sha256=sha(stage2 / 'training_assumptions.json'),
        stage3=dict(adapter_selection='System policy based on task and selected architecture, not explicit user module choices',
            qlora_non_quantized_weights_budget='FP32 after k-bit preparation',
            hyperparameters='Unmeasured starting points; training data size/distribution is not supplied'),
        downstream_contract=dict(render_as_user_requirements=False, keep_assumptions_separate_from_recommendation=True))
    drafts = [o for o in outputs if o['output_1'] is not None]
    summary = dict(version=VERSION, source_stage2_version=read(stage2 / 'labeling_summary.json')['version'], total_cases=len(rows),
        stage3_status_counts=dict(Counter(o['stage3_status'] for o in outputs)), recommended_drafts=len(drafts),
        algorithm_counts=dict(Counter(r['label']['algorithm'] for r, o in zip(rows, outputs) if o['output_1'])),
        source_cases_and_labels_preserved=True, catalog_violations=0, per_output_cpu_offload_fields=0,
        reasoning_decisions=0, empirically_validated_cases=0, ready_to_execute_cases=0,
        generation_method='codex_authored_rules_no_per_case_llm_call', source_sha256=before,
        downstream_regeneration_required=['stage_4', 'stage_5'],
        output_schema='input_1={case,label}; output_1={hyperparameters,training_plan,backend_requirements,parameter_rationale}; null if unresolved')
    data = {'input_1.json': inputs, 'output_1.json': outputs, 'hyperparameter_cases.json': combined,
            'review_cases.json': [o for o in outputs if o['stage3_status'] in {'pending_stage2_review', 'needs_resource_review'}],
            'not_trainable_cases.json': [o for o in outputs if o['stage3_status'] == 'skipped_stage2_not_possible'],
            'module_specs.json': {m: module_spec(m) for m in selected_models}, 'training_assumptions.json': assumptions,
            'hyper_parameter_candidates.json': catalog, 'source_manifest.json': {k: {'path': k, 'sha256': v} for k, v in before.items()},
            'selection_policy.json': dict(version=VERSION, strict_catalog=True, case_and_label_changes=False,
                sequence_source='dataset/context_length_list.json', scope_policy='Keep all task-required encoders, decoders, projectors and heads; distinguish frozen from unused',
                stage2_unresolved='Null output_1, preserve reason/status; no fabricated hyperparameters',
                memory_policy='max(Stage2 estimate, scope-aware parameter state + inherited activation/reserve); lower adapter rank within catalog, otherwise review',
                source_links=['https://huggingface.co/docs/peft/en/package_reference/lora', 'https://huggingface.co/docs/transformers/en/quantization/bitsandbytes', 'https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning', 'https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning'])}
    if before != {rel(p): sha(p) for p in paths}:
        raise RuntimeError('Sources changed')
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    backup_existing(output_dir, [p.name for p in output_dir.glob('*.json')], 'before_v3')
    for n in ['validation_report.json', 'reproducibility_report.json']:
        (output_dir / n).unlink(missing_ok=True)
    for name, value in data.items():
        write(output_dir / name, value)
    summary['artifact_sha256'] = {n: sha(output_dir / n) for n in data}
    write(output_dir / 'generation_summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['version', 'total_cases', 'stage3_status_counts', 'algorithm_counts']}, ensure_ascii=False, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=STAGE3)
    parser.add_argument('--stage2-dir', type=Path, default=STAGE2)
    parser.add_argument('--stage1-dir', type=Path, default=STAGE1)
    args = parser.parse_args(argv)
    return generate(args.output_dir, args.stage2_dir, args.stage1_dir)


if __name__ == '__main__':
    main()
