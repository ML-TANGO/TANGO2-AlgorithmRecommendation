"""Source, catalog, modality scope and resource audit for Stage 3 v3."""
import argparse
import json
from collections import Counter
from pathlib import Path

from .generate import propose
from .context_lengths import load_context_lengths
from .training_scopes import resolve_and_verify_adapter_targets
from ..io_utils import read, write, sha, canon, check
from ..paths import DATASET_DIR, STAGE2, STAGE3, resolve

NLP = {'Classification', 'Question Answering', 'Summarization', 'Translation'}
NON_TEXT = {'Image Classification', 'Detection', 'Segmentation', 'Audio Classification'}


def contains_key(obj, key):
    if isinstance(obj, dict):
        return key in obj or any(contains_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(contains_key(v, key) for v in obj)
    return False


def validate(output_dir=STAGE3, stage2=STAGE2, dataset_dir=DATASET_DIR):
    output_dir = Path(output_dir); stage2 = Path(stage2); dataset_dir = Path(dataset_dir)
    rows = read(stage2 / 'labeled_cases.json'); inputs = read(output_dir / 'input_1.json'); outputs = read(output_dir / 'output_1.json')
    joined = read(output_dir / 'hyperparameter_cases.json'); summary = read(output_dir / 'generation_summary.json')
    profiles = {p['model_id']: p for p in read(stage2 / 'model_profiles.json')}
    catalog = read(dataset_dir / 'hyper_parameter_list.json'); contexts = load_context_lengths(dataset_dir / 'context_length_list.json'); specs = read(output_dir / 'module_specs.json')
    check(len(rows) == len(inputs) == len(outputs) == len(joined) == 5000, 'Wrong count')
    check(len({x['case_id'] for x in outputs}) == 5000, 'Duplicate ID')
    for n, digest in summary['source_sha256'].items():
        check(sha(resolve(n)) == digest, 'Source changed: ' + n)
    for n, digest in summary['artifact_sha256'].items():
        check(sha(output_dir / n) == digest, 'Artifact changed: ' + n)
    check(read(output_dir / 'source_manifest.json') == {k: dict(path=k, sha256=v) for k, v in summary['source_sha256'].items()}, 'Manifest mismatch')
    check(read(output_dir / 'hyper_parameter_candidates.json') == catalog, 'Catalog snapshot drift')
    policy = read(output_dir / 'training_assumptions.json'); upstream = read(stage2 / 'training_assumptions.json')
    check(policy['workload'] == upstream['workload'] and policy['origin'] == 'system_calculation_policy' and policy['user_requested'] is False, 'Assumption origin or contents changed')
    check(policy['downstream_contract']['render_as_user_requirements'] is False, 'Assumptions as user requests')
    counts = Counter(); algos = Counter(); drafts = 0; adapters = 0; context_cases = 0
    for row, inp, out, full in zip(rows, inputs, outputs, joined):
        cid = row['case_id']; task = row['case']['task']; algo = row['label']['algorithm']
        check(cid == inp['case_id'] == out['case_id'] == full['case_id'], cid + ': ID drift')
        check(inp['input_1'] == dict(case=row['case'], label=row['label']), cid + ': changed case or label')
        check(full == dict(**out, input_1=inp['input_1']), cid + ': inconsistent join')
        check(not contains_key(out, 'cpu_offload') and not contains_key(out, 'reasoning') and not contains_key(out, 'thinking'), cid + ': removed field leaked')
        check(out['training_assumptions_ref'] == 'training_assumptions.json', cid + ': policy ref')
        check(out['ready_to_execute'] is False and out['empirically_validated'] is False, cid + ': execution claim')
        check(out == propose(row, catalog, profiles, contexts), cid + ': reproduction failure')
        counts[out['stage3_status']] += 1
        if row['training_status'] != 'possible':
            check(out['output_1'] is None and 'review_candidate' not in out, cid + ': fabricated settings for null label')
            check(out['source_stage2_status'] == row['training_status'] and out['reason'], cid + ': missing pending reason')
            continue
        payload = out['output_1'] or out.get('review_candidate'); check(payload is not None, cid + ': missing planning result')
        hp = payload['hyperparameters']; plan = payload['training_plan']; backend = payload['backend_requirements']; p = profiles[row['label']['model']]
        check(set(hp) == set(catalog[algo]), cid + ': hyperparameter schema')
        for k, v in hp.items():
            check(canon(v) in [canon(x) for x in catalog[algo][k]], cid + ': outside candidate ' + k)
        check('max_seq_length' not in hp and 'context_length' not in hp, cid + ': duplicate length candidate')
        check(hp['per_device_train_batch_size'] == 1 and hp['optimizer'] == 'AdamW', cid + ': invalid memory assumption')
        check(backend['effective_batch_size'] == hp['gradient_accumulation_steps'], cid + ': batch mismatch')
        sequence = plan['sequence_length']
        check(row['case']['context_length'] in contexts, cid + ': outside context catalog')
        check(sequence['applicable'] == p['context_applicable'] == (task not in NON_TEXT), cid + ': context applicability')
        check(backend['max_seq_length'] == sequence['max_seq_length'], cid + ': context mismatch')
        if sequence['applicable']:
            check(sequence['max_seq_length'] == row['case']['context_length'], cid + ': context clipped')
            check(sequence['max_seq_length'] + p['extra_tokens'] <= p['native_context'], cid + ': native context exceeded')
        else:
            check(sequence['max_seq_length'] is None, cid + ': non-text arbitrary length')
        check(not (set(plan['active_components']) & set(plan['components_to_not_load_if_present'])), cid + ': required component disabled')
        check(plan['applied_to_live_model'] is False and plan['requires_runtime_module_verification'], cid + ': false runtime claim')
        if algo in {'LoRA', 'QLoRA'}:
            spec = specs[row['label']['model']]
            check(plan['adapter_scope_prefixes'] == [spec['scope_prefix']], cid + ': scope mismatch')
            actual = {m['name']: (m['in_features'], m['out_features']) for m in spec['modules']}
            resolved = resolve_and_verify_adapter_targets(actual, spec, hp['target_modules'])
            check(len(resolved['module_names']) == plan['adapter_targets']['expected_module_count'], cid + ': adapter count')
            check(plan['modules_to_save'] == [] and plan['adapter_base_weights_frozen'] and plan['peft_bias'] == 'none', cid + ': unexpected base training')
            check(hp['lora_alpha'] == 2 * hp['lora_r'], cid + ': alpha/r')
            check(any(c['mode'] == 'adapter_only' for c in plan['component_decisions']), cid + ': missing trainable adapters')
            check(not any(c['mode'] == 'full_training' for c in plan['component_decisions']), cid + ': trains frozen weights')
            check(all('audio_tower' not in n and 'vision' not in n for n in resolved['module_names']), cid + ': adapter leakage')
        else:
            check(not plan['adapter_scope_prefixes'] and not plan['quantization_scope_prefixes'], cid + ': SFT has adapters')
            check(any(c['mode'] == 'full_training' for c in plan['component_decisions']), cid + ': no SFT weights')
        if algo == 'QLoRA':
            check(plan['quantization_scope_prefixes'] == plan['adapter_scope_prefixes'], cid + ': quantization scope')
            check(hp['bnb_4bit_compute_dtype'] == hp['training_dtype'], cid + ': compute dtype')
            check(backend['non_quantized_weight_budget_dtype'] == 'float32', cid + ': missed FP32 remainder')
        if task in NLP:
            check(plan['input_modalities'] == ['text'], cid + ': wrong NLP modality')
        if task == 'ASR' and algo == 'LoRA':
            check('audio_encoder' in plan['frozen_but_used_components'] and 'audio_encoder' in plan['active_components'], cid + ': ASR lost frozen encoder')
            check(plan['adapter_scope_prefixes'] == ['thinker.model.layers'], cid + ': unverified ASR scope')
        if task == 'TTS':
            check({'speaker_encoder', 'audio_codec'} <= set(plan['frozen_but_used_components']), cid + ': TTS freezing')
        if out['output_1'] is not None:
            check(out['stage3_status'] == 'recommended_draft' and out['resource_check']['within_budget'], cid + ': unsafe draft')
            drafts += 1; algos[algo] += 1; adapters += algo in {'LoRA', 'QLoRA'}; context_cases += sequence['applicable']
        else:
            check(out['stage3_status'] == 'needs_resource_review' and not out['resource_check']['within_budget'], cid + ': resource classification')
    check(summary['stage3_status_counts'] == dict(counts) and summary['algorithm_counts'] == dict(algos), 'Summary drift')
    check(read(output_dir / 'review_cases.json') == [o for o in outputs if o['stage3_status'] in {'pending_stage2_review', 'needs_resource_review'}], 'Review export')
    check(read(output_dir / 'not_trainable_cases.json') == [o for o in outputs if o['stage3_status'] == 'skipped_stage2_not_possible'], 'Unsupported export')
    report = dict(valid=True, version='3.0', cases_verified=5000, source_cases_and_labels_preserved=True,
                  recommended_drafts=drafts, adapter_plans=adapters, context_preserved_drafts=context_cases,
                  status_counts=dict(counts), algorithm_counts=dict(algos), catalog_violations=0,
                  per_output_cpu_offload_fields=0, reasoning_decisions=0, empirical_training_performed=False,
                  module_path_validation='Config/source-based inventory checks; actual loaded checkpoint verification remains required',
                  artifact_sha256={n: sha(output_dir / n) for n in ['input_1.json', 'output_1.json', 'hyperparameter_cases.json', 'generation_summary.json']})
    write(output_dir / 'validation_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=STAGE3)
    p.add_argument('--stage2-dir', type=Path, default=STAGE2)
    args = p.parse_args(argv)
    return validate(args.output_dir, args.stage2_dir)


if __name__ == '__main__':
    main()
