"""V7: four-field, task-specific feasibility estimates; preserve every Stage 1 case.

Unknown support or memory headroom produces needs_review, never a fabricated
negative label. All constraints are fixed before quota-based sampling.
"""
import copy
import math
from functools import lru_cache

from . import labeling_core as core
from ..io_utils import read
from ..paths import DATASET_DIR

NLP = {'Classification', 'Question Answering', 'Summarization', 'Translation'}
CV = {'Image Classification', 'Detection', 'Segmentation'}
POLICY = {
    'version': '7.0',
    'basis': 'prospective_estimate_under_explicit_policy_not_measured',
    'positive_memory_fraction': .75,
    'status_semantics': {
        'possible': 'A candidate passes explicit task/support and conservative planning estimates; not an empirical execution result.',
        'not_possible': 'Every considered candidate violates a documented constraint or explicit requested capability under this policy, not universal impossibility.',
        'needs_review': 'Incomplete evidence or uncertain memory fit; keep the case with null model/algorithm pending review.'},
    'workload': {'gpu_count': 1, 'cpu_offload': False, 'distributed_training': False,
        'microbatch': 1, 'gradient_checkpointing': True,
        'text_attention': 'Memory-efficient attention and chunked LM cross entropy; backend compatibility must be checked before execution.',
        'context_length': 'Text input/output budget only for language-generative models; not an image resolution or audio duration.',
        'image_count': 1, 'audio_duration_max_seconds': 10,
        'audio_classification_subtask': 'multi_label_sound_event_tagging unless explicitly specified',
        'tts_text_budget_reviewed_max': 256,
        'visual_qa_or_summary': 'Four-field Stage 1 cases do not specify visual inputs; QA and summary default to text. No image token workload is silently added.'},
    'memory': 'Uncalibrated estimates in decimal GB. SFT=18 B/parameter, LoRA=2.5, QLoRA=0.8 plus architecture-specific activation/reserve. No OOM inferred from an activation estimate alone.',
    'negative_memory_rule': 'Conservative weight-storage floor must exceed physical VRAM allowing GB/GiB conversion and 10% tolerance.',
    'algorithms': 'Text adapters require appropriate backend/targets. CV, Whisper and AST/keyword heads use task-specific supervised SFT, not text SFTTrainer.',
    'language': 'Documented ASR/TTS workflows constrain released-support scope; undocumented text-language support produces review. Signal/image class IDs do not depend on display-language codes.',
    'metrics': 'Primary metrics belong to the task catalog. LogicKor is supplementary for Korean generative QA using its official fixed benchmark and judge protocol.',
    'sampling': 'No sampling or resampling in Stage 2. Preserve all four-field Stage 1 cases and IDs.',
    'reasoning_policy': 'No reasoning requirement, preference, filter or enabled decision is used in Stage 2; any later decision belongs to the final recommendation stage.',
    'assumptions_origin': 'system_calculation_policy_not_user_requirements',
    'ranking': 'Text: target size 3B for Classification, otherwise 8B if GPU VRAM >=24GB else 4B; Qwen3 +3, Gemma3 +2 policy points. Specialists: prefer larger supported candidate if it fits. QLoRA score penalty 0.5; not empirical quality.',
    'algorithm_selection': 'Prefer LoRA then QLoRA then SFT among supported methods passing the budget; dedicated models may support only SFT.'}


REVIEW_MESSAGES = {
    'language_support_unverified': '요청 언어의 모델 지원 근거가 부족합니다.',
    'source_language_support_unverified': '가정한 번역 원문 언어의 지원 근거가 부족합니다.',
    'workload_outside_reviewed_text_length': '요청 길이가 현재 검토한 작업량 범위를 넘습니다. 모델의 절대 한계라는 뜻은 아닙니다.',
    'memory_estimate_requires_measurement': '보수적인 메모리 예산을 통과하지 못했지만 저장 하한만으로 불가능을 확정할 수 없어 실측이 필요합니다.',
}


def _saved_config(mid):
    return read(core.source_config_path(mid))['config']


def profiles(dataset_dir=DATASET_DIR):
    models = read(dataset_dir / 'model_list.json')['models']
    catalog = read(dataset_dir / 'model_task_capabilities.json')
    caps = {p['model_id']: p for p in catalog['models'] + catalog['text_model_capabilities']}
    text_ids = {p['model_id'] for p in catalog['text_model_capabilities']}
    text = {p['model_id']: p for p in core.make_profiles([m for m in models if m['model_id'] in text_ids])}
    result = []
    for m in models:
        mid = m['model_id']; c = caps[mid]
        p = {**m, **copy.deepcopy(c), **text.get(mid, {})}
        count = c['parameter_metadata'].get('hf_reported_tensor_elements') or c['parameter_metadata'].get('config_parameter_count')
        p.update(memory_params_b=count / 1e9 if count else m['params_b_approx'],
            profile_type='text' if mid in text else 'specialist',
            algorithms=['LoRA', 'QLoRA', 'SFT'] if mid in text else ['SFT'],
            native_context=text[mid]['max_context_tokens_used'] if mid in text else None,
            max_text_context=None, context_applicable=True, extra_tokens=0,
            hidden_size=p.get('hidden_size', 0), layers=p.get('layers', 0),
            vocab_size=p.get('vocab_size', 0), reserve_gb=2.,
            languages=text[mid]['languages_used'] if mid in text else c['language_support']['documented_codes'],
            language_rule='multilingual_claim' if mid in text and text[mid]['languages_used'] is None else 'review_outside_evidence',
            native_thinking=text[mid]['native_thinking'] if mid in text else False,
            objective='causal_language_model_supervision' if mid in text else 'task_specific_supervised_adaptation',
            training_sources=list(c['sources']), context_rule='native_limit',
            task_subtype=None, workload={})
        if mid in text:
            p['workload'] = {'input_modality': 'text', 'image_encoder_used': False,
                             'translation_source_language': 'en (ko when target is en)'}
            p['target_tasks'] = c['target_tasks']
        elif 'Qwen3-VL' in mid:
            cfg = _saved_config(mid)['text_config']
            p.update(algorithms=['LoRA', 'QLoRA', 'SFT'], native_context=cfg['max_position_embeddings'],
                     hidden_size=cfg['hidden_size'], layers=cfg['num_hidden_layers'], vocab_size=cfg['vocab_size'],
                     language_rule='multilingual_claim', reserve_gb=8., objective='vision_language_decoder_supervision',
                     workload={'input_modality': 'text for current four-field cases', 'vision_encoder_frozen': True, 'visual_inputs_verified': False})
            p['training_sources'].append('https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune')
        elif 'Qwen3-ASR' in mid:
            cfg = _saved_config(mid)['thinker_config']['text_config']
            p.update(algorithms=['LoRA', 'SFT'], native_context=cfg['max_position_embeddings'],
                     extra_tokens=500, hidden_size=cfg['hidden_size'], layers=cfg['num_hidden_layers'], vocab_size=cfg['vocab_size'],
                     language_rule='released_support_scope', reserve_gb=5., objective='audio_to_text_supervision',
                     workload={'audio_duration_max_seconds': 10, 'audio_token_allowance': 500, 'encoder_trainability': 'backend-specific; reserve assumes full encoder present'})
            p['training_sources'] += ['https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning', 'https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html']
        elif 'Qwen3-TTS' in mid:
            cfg = _saved_config(mid)['talker_config']
            p.update(max_text_context=256, native_context=cfg['max_position_embeddings'], hidden_size=cfg['hidden_size'], layers=cfg['num_hidden_layers'], extra_tokens=1000, reserve_gb=8.,
                language_rule='released_support_scope', objective='single_speaker_codec_SFT',
                workload={'audio_duration_max_seconds': 10, 'text_budget_reviewed_max': 256, 'single_speaker_adaptation': True, 'codec_token_allowance': 1000})
            p['training_sources'].append('https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning')
        elif mid == 'openai/whisper-small':
            cfg = _saved_config(mid)
            p.update(native_context=cfg['max_target_positions'], hidden_size=cfg['d_model'], layers=cfg['decoder_layers'],
                vocab_size=cfg['vocab_size'], reserve_gb=3., language_rule='released_support_scope', objective='encoder_decoder_ASR',
                workload={'audio_duration_max_seconds': 10, 'sampling_rate_hz': 16000, 'decoder_budget_includes_special_tokens': True})
            p['training_sources'].append('https://huggingface.co/blog/fine-tune-whisper')
        else:
            p.update(context_applicable=False, language_rule='signal_or_class_ids', languages=None)
            if m['family'] in {'ViT', 'ConvNeXT'}:
                p.update(reserve_gb=3., objective='single_label_image_classification', workload={'image_size': [224, 224], 'maximum_classes': 1000, 'loss': 'cross_entropy'})
            elif m['family'] in {'DETR', 'RT-DETR'}:
                p.update(reserve_gb=5. if m['family'] == 'DETR' else 3.5, objective='box_and_class_detection',
                    workload={'image_size': [640, 640], 'maximum_objects': 100, 'loss': 'matching plus class/box/GIoU', 'dataset_scope': 'COCO-style boxes'})
            elif m['family'] == 'SegFormer':
                p.update(reserve_gb=3. if '-b0-' in mid else 6., objective='semantic_segmentation',
                    workload={'image_size': [512, 512], 'maximum_classes': 150, 'loss': 'pixel_cross_entropy', 'ignore_index': 255})
            elif m['family'] == 'AST':
                p.update(reserve_gb=3., objective='multi_label_sound_event_tagging', task_subtype='sound_event_tagging',
                    workload={'audio_duration_max_seconds': 10, 'sampling_rate_hz': 16000, 'maximum_classes': 527, 'loss': 'binary_cross_entropy_with_logits'})
            elif m['family'] == 'Wav2Vec2':
                p.update(reserve_gb=3., objective='single_label_keyword_classification', languages=['en'],
                    language_rule='released_support_scope', task_subtype='keyword_spotting',
                    workload={'audio_duration_max_seconds': 1, 'sampling_rate_hz': 16000, 'maximum_classes': 12, 'loss': 'cross_entropy'})
            else:
                raise ValueError(f'No audited specialist profile for {mid}')
            p['training_sources'].append('https://huggingface.co/docs/transformers/tasks/' + {
                'Image Classification': 'image_classification', 'Detection': 'object_detection',
                'Segmentation': 'semantic_segmentation', 'Audio Classification': 'audio_classification'}[p['target_tasks'][0]])
        result.append(p)
    return result


class Evaluator:
    def __init__(self, dataset_dir=DATASET_DIR):
        self.profiles = profiles(dataset_dir)
        self.allowed = set(read(dataset_dir / 'training_algorithm_list.json'))
        self.metrics = read(dataset_dir / 'metric_list.json')
        self.modelmap = {p['model_id']: p for p in self.profiles}

    @lru_cache(maxsize=None)
    def memory(self, mid, seq, algorithm, vision=False):
        p = self.modelmap[mid]
        n = p['memory_params_b']
        effective = (seq if p['context_applicable'] else 0) + p['extra_tokens'] + (256 if vision else 0)
        state = n * {'SFT': 18., 'LoRA': 2.5, 'QLoRA': .8}[algorithm]
        activation = 4 * effective * p['hidden_size'] * p['layers'] / 1e9
        logits = 4 * min(effective, 1024) * p['vocab_size'] / 1e9
        return state + activation + logits + p['reserve_gb']

    def evaluate(self, case, details=True):
        task, lang, seq = case['task'], case['language'], case['context_length']
        if task not in self.metrics:
            raise ValueError(f'Task is outside current catalog: {task}')
        vram = case['gpu']['vram_gb']
        if set(case) - {'case_id', 'task', 'gpu', 'language', 'context_length'}:
            raise ValueError('Unexpected case fields; reasoning is not an input to Stage 2')
        checks = []; viable = []
        for p in self.profiles:
            if task not in p['target_tasks']:
                continue
            mid = p['model_id']; fail = []; review = []; why = []

            def reject(code, message):
                fail.append(code); why.append(message)
            if p['languages'] is not None and lang not in p['languages']:
                if p['language_rule'] == 'released_support_scope':
                    reject('language_outside_released_support_scope', f'요청 언어 {lang}가 공개 지원 목록 {p["languages"]} 밖임; 새 언어 확장 학습은 가정하지 않음')
                else:
                    review.append('language_support_unverified')
            if task == 'Translation' and p['languages'] is not None:
                source = 'ko' if lang == 'en' else 'en'
                if source not in p['languages']:
                    review.append('source_language_support_unverified')
            if p['context_applicable'] and p['native_context'] is not None and seq + p['extra_tokens'] > p['native_context']:
                reject('context_exceeds_model_policy_limit', f'요청 텍스트 {seq:,} + modality {p["extra_tokens"]:,} token > 정책 상한 {p["native_context"]:,}; 입력 분할·context 확장 미가정')
            if p['max_text_context'] is not None and seq > p['max_text_context']:
                review.append('workload_outside_reviewed_text_length')
            if p['task_subtype'] is not None:
                subtype = case.get('audio_subtask', 'sound_event_tagging')
                if subtype != p['task_subtype']:
                    reject('task_subtype_outside_checkpoint_scope', f'가정한 오디오 세부 작업 {subtype}과 checkpoint 범위 {p["task_subtype"]}가 다름')
            estimates = {a: self.memory(mid, seq, a, False) for a in p['algorithms'] if a in self.allowed}
            fitting = [a for a, n in estimates.items() if n <= POLICY['positive_memory_fraction'] * vram]
            floors = {a: p['params_b_approx'] * (.5 if a == 'QLoRA' else 2.) for a in estimates}
            if not fitting:
                physical_upper = vram * (1024 ** 3 / 1e9) * 1.10
                if floors and min(floors.values()) > physical_upper:
                    reject('minimum_weight_storage_exceeds_vram', f'가중치 저장 하한 {min(floors.values()):.3f} GB > GPU {vram} GB의 GiB·10% 여유 환산 {physical_upper:.3f} GB')
                else:
                    review.append('memory_estimate_requires_measurement')
            status = 'not_possible' if fail else 'needs_review' if review else 'possible'
            if details:
                checks.append(dict(model=mid, status=status, reason_codes=fail, review_codes=review, reasons=why,
                    fitting_algorithms=fitting,
                    algorithm_estimates_gb={a: round(n, 6) for a, n in estimates.items()}))
            else:
                checks.append(status)
            if status == 'possible':
                algo = next(a for a in ('LoRA', 'QLoRA', 'SFT') if a in fitting)
                if p['profile_type'] == 'text':
                    score, _ = core.model_score(p, case)
                elif 'Qwen3-VL' in mid:
                    # No visual requirement exists in Stage 1: prefer text-only models when comparable.
                    score = -8 - abs(math.log2(p['params_b_approx'] / 4))
                else:
                    score = math.log2(max(p['params_b_approx'], .001))
                if algo == 'QLoRA':
                    score -= .5
                viable.append((score, mid, algo, estimates[algo], p))
        statuses = [c['status'] for c in checks] if details else checks
        status = 'possible' if viable else 'needs_review' if not statuses or 'needs_review' in statuses else 'not_possible'
        if not details:
            return status
        metric, metric_why, requirements, refs = core.pick_metric(case, self.metrics)
        row = dict(case_id=case.get('case_id'), label=dict(model=None, algorithm=None, evaluation_metric=metric),
            training_possible=None if status == 'needs_review' else status == 'possible', training_status=status,
            feasibility_basis=POLICY['basis'], empirically_validated=False, impossibility_reason=None, reason_codes=[],
            evaluation_requirements=requirements, rationale=dict(model=None, algorithm=None, evaluation_metric=metric_why),
            assumptions_ref='training_assumptions.json', review_notes=[], sources=refs,
            label_method='codex_authored_rules_no_per_case_llm_call', policy_version='7.0', candidate_checks=checks,
            context_applicable=task in NLP | {'ASR', 'TTS'},
            supplementary_evaluation=[])
        if task == 'Question Answering' and lang == 'ko':
            row['supplementary_evaluation'] = [dict(metric='LogicKor Score', role='optional_official_benchmark',
                prerequisites='공식 질문·judge template·scoring script 사용; benchmark revision, judge model/version, single/multi-turn 점수 기록. 임의 QA 데이터에 이 점수 이름을 사용하지 않음.',
                uses_llm_judge=True, source=core.SOURCES['logickor'])]
            row['sources'].append(core.SOURCES['logickor'])
        if viable:
            score, mid, algo, mem, p = sorted(viable, key=lambda x: (-x[0], x[1]))[0]
            row['label'].update(model=mid, algorithm=algo)
            row['rationale']['model'] = f'{task} 지원 범위·언어·요청 조건을 만족하는 후보 중 정책 우선순위로 선택했습니다. {case["gpu"]["gpu_name"]} {vram:g}GB의 75% 예산 안에서 추정 메모리 {mem:.2f}GB입니다. 후보 순위는 실측 품질 순위가 아닙니다.'
            row['rationale']['algorithm'] = f'지원 방식 {", ".join(p["algorithms"])} 중 {algo}를 선택했습니다. 목적함수는 {p["objective"]}이며 해당 모델의 processor·loss·backend를 사용해야 합니다.'
            row['resource_check'] = dict(gpu_vram_gb=vram, positive_budget_gb=.75 * vram, estimated_memory_gb=round(mem, 6), empirical_measurement=False,
                algorithm_estimates_gb=next(c['algorithm_estimates_gb'] for c in checks if c['model'] == mid))
            row['context_applicable'] = p['context_applicable']; row['training_objective'] = p['objective']
            row['selected_workload_assumptions'] = dict(origin='system_calculation_policy', user_requested=False, values=copy.deepcopy(p['workload']))
            row['review_notes'] = ['가능은 명시된 가정 하의 추천 추정이며 실제 학습 성공·메모리 실측을 뜻하지 않음'] + p['requirements_and_limits']
            row['sources'] += p['training_sources']
            if task == 'Translation':
                row['review_notes'].append('language는 목표 언어. 출발 언어는 영어, 목표가 영어이면 한국어로 가정함; 실제 언어쌍을 확인해야 함')
        else:
            row['reason_codes'] = sorted({code for c in checks for code in c['reason_codes']})
            if status == 'not_possible':
                row['impossibility_reason'] = '현재 모델 목록과 단일 GPU·offload 없음·새 언어/context 확장 없음 조건에서 모든 후보가 제외되었습니다. 후보별 정확한 조건과 요청값은 candidate_checks의 reasons에 기록했습니다.'
                # Include concrete reasons in the top-level explanation as well.
                row['impossibility_reason'] += ' ' + ' / '.join(dict.fromkeys(reason for c in checks for reason in c['reasons']))
                row['rationale']['model'] = row['impossibility_reason']
            else:
                row['rationale']['model'] = '지원 근거 또는 메모리 추정의 불확실성 때문에 추가 검토가 필요합니다. 불가능으로 단정하지 않습니다.'
                row['review_notes'] = sorted({code for c in checks for code in c['review_codes']})
                row['review_reason'] = ' / '.join(REVIEW_MESSAGES.get(code, code) for code in row['review_notes'])
            row['sources'] += ['dataset/model_task_capabilities.json', 'stage_2/labeling_policy.json']
        row['sources'] = list(dict.fromkeys(row['sources']))
        return row
