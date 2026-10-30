"""Stage 4 revised input_2 -> label_2 contract, shared by preparation, training, inference and demo."""
import json
import re

from ..io_utils import read as read_json, read_jsonl, sha as file_hash  # noqa: F401 (re-exported)

SCHEMA_VERSION = 'stage4-label2-revised-v1'
MODEL_ALIASES = {'gemma': 'google/gemma-4-E4B-it', 'gemma4': 'google/gemma-4-E4B-it', 'gemma4-e4b-it': 'google/gemma-4-E4B-it', 'qwen': 'Qwen/Qwen3-8B', 'qwen3': 'Qwen/Qwen3-8B'}
TARGET_KEYS = {'status', 'recommendation', 'thinking', 'review_reasons', 'recommendation_assumptions', 'reason', 'source_stage3_status', 'empirically_validated', 'ready_to_execute'}
RECOMMENDATION_KEYS = {'model', 'algorithm', 'evaluation_metric', 'hyperparameters', 'training_plan', 'backend_requirements', 'recommendation_reason', 'parameter_rationale'}
# Verbatim training contract. Changing it invalidates every trained checkpoint.
SYSTEM_PROMPT = '''사용자의 작업 설명과 요구사항을 읽고 모델 학습 구성을 추천하세요. 응답은 코드 블록이나 부가 설명 없이 label_2 형식의 JSON 객체 하나로 작성하세요.
최상위 필드는 status, recommendation, thinking, review_reasons, recommendation_assumptions, reason, source_stage3_status, empirically_validated, ready_to_execute입니다.
recommendation에는 model, algorithm, evaluation_metric, hyperparameters, training_plan, backend_requirements, recommendation_reason, parameter_rationale를 포함하세요. training_plan에는 task에 필요한 모듈과 학습·동결 대상, LoRA/QLoRA 적용 범위, 사용하지 않는 모달리티를 구분하세요. 필요한 decoder·출력 head를 task 이름만으로 제거하지 마세요.
thinking의 recommended는 추천 대상 작업에서 별도 thinking 모델/모드가 권장되는지입니다. 추천을 고민하는 과정의 난이도와 혼동하지 마세요. model_supports_thinking은 선택한 모델의 지원 여부이며 별개입니다. 짧은 판단 사유를 적되 내부 사고 과정을 출력하지 마세요.
사용자가 지정하지 않은 GPU 용량·CPU offloading·작업량 등은 사용자 조건이라고 주장하지 말고 recommendation_assumptions에 시스템 가정으로 분리하세요. 페이지 수의 토큰 환산은 문서·언어·tokenizer에 따른 추정이며 정확한 측정값이라고 주장하지 마세요. TTS의 텍스트 분량은 지시문과 읽어 줄 문장의 양이며 음성 출력 길이가 아닙니다.
추천은 실측하지 않은 초안입니다. empirically_validated와 ready_to_execute는 false로 유지하세요. 근거가 부족하면 검토 사유를 표시하세요.'''
TAG_PATTERN = re.compile(r'^\s*<USER_TASK_DESCIRPTION>\s*(.*?)\s*</USER_TASK_DESCIRPTION>\s*<USER_REQUIREMENTS>\s*(.*?)\s*</USER_REQUIREMENTS>\s*$', re.S)


def write_json(path, value):
    from pathlib import Path
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def resolve_model(name):
    return MODEL_ALIASES.get(name.lower(), name)


def split_input(text):
    match = TAG_PATTERN.fullmatch(text)
    if not match or not all(v.strip() for v in match.groups()):
        raise ValueError('Expected non-empty USER_TASK_DESCIRPTION and USER_REQUIREMENTS sections')
    return match.group(1).strip(), match.group(2).strip()


def build_input(task_description, requirements):
    values = [task_description, requirements]
    if any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError('작업 설명과 요구사항을 모두 입력하세요.')
    if any(re.search(r'</?USER_[A-Z_]+>', v, re.I) for v in values):
        raise ValueError('각 항목의 내용만 입력하세요. USER 태그는 프로그램이 붙입니다.')
    # Preserve the historical Stage 4 spelling at the model boundary.
    return f'<USER_TASK_DESCIRPTION>\n{task_description.strip()}\n</USER_TASK_DESCIRPTION>\n<USER_REQUIREMENTS>\n{requirements.strip()}\n</USER_REQUIREMENTS>'


def target_errors(value, catalog=None):
    if not isinstance(value, dict):
        return ['label_2 must be a JSON object']
    errors = []
    if set(value) != TARGET_KEYS:
        errors.append('label_2 top-level keys differ from the Stage 4 contract')
    statuses = {'conditional_recommendation_draft', 'pending_stage2_review', 'needs_resource_review', 'skipped_stage2_not_possible', 'needs_gpu_capacity_clarification', 'needs_thinking_model_review'}
    if not isinstance(value.get('status'), str) or value['status'] not in statuses:
        errors.append('Unknown recommendation status')
    for key in ['empirically_validated', 'ready_to_execute']:
        if value.get(key) is not False:
            errors.append(key + ' must remain false; no empirical execution was performed')
    if not isinstance(value.get('reason'), str):
        errors.append('reason must be a string')
    if not isinstance(value.get('source_stage3_status'), str):
        errors.append('source_stage3_status must be a string')
    if not isinstance(value.get('review_reasons'), list) or any(not isinstance(x, str) for x in value.get('review_reasons', [])):
        errors.append('review_reasons must be a string list')
    thinking = value.get('thinking')
    if not isinstance(thinking, dict):
        errors.append('thinking must be an object')
    else:
        for k in ['recommended', 'explicitly_requested', 'activation_configured']:
            if type(thinking.get(k)) is not bool:
                errors.append('thinking.' + k + ' must be boolean')
        if thinking.get('model_supports_thinking') is not None and type(thinking['model_supports_thinking']) is not bool:
            errors.append('thinking support must be boolean or null')
        if not isinstance(thinking.get('reason'), str) or not thinking['reason'].strip():
            errors.append('thinking.reason is required')
        if thinking.get('recommended') is True and not isinstance(thinking.get('request_evidence'), str):
            errors.append('thinking=true needs request evidence')
    assumptions = value.get('recommendation_assumptions')
    if not isinstance(assumptions, dict):
        errors.append('recommendation_assumptions must be an object')
    else:
        for name, a in assumptions.items():
            if not isinstance(a, dict) or a.get('user_requested') is not False:
                errors.append('System assumptions must have user_requested=false: ' + name)
    rec = value.get('recommendation')
    if rec is None:
        if value.get('status') == 'conditional_recommendation_draft':
            errors.append('A recommendation draft cannot have null recommendation')
    elif not isinstance(rec, dict):
        errors.append('recommendation must be an object or null')
    else:
        if set(rec) != RECOMMENDATION_KEYS:
            errors.append('recommendation keys differ from Stage 4')
        for k in ['model', 'algorithm', 'evaluation_metric']:
            if not isinstance(rec.get(k), str) or not rec[k]:
                errors.append('recommendation.' + k + ' must be a string')
        for k in RECOMMENDATION_KEYS - {'model', 'algorithm', 'evaluation_metric'}:
            if not isinstance(rec.get(k), dict):
                errors.append('recommendation.' + k + ' must be an object')
        if catalog:
            for key, collection in [('model', 'models'), ('algorithm', 'algorithms'), ('evaluation_metric', 'metrics')]:
                if rec.get(key) not in catalog[collection]:
                    errors.append(key + ' is outside the saved catalog')
        plan = rec.get('training_plan')
        if isinstance(plan, dict) and isinstance(rec.get('algorithm'), str) and rec['algorithm'] in {'LoRA', 'QLoRA'}:
            if not plan.get('adapter_scope_prefixes'):
                errors.append('Adapter recommendation needs a non-empty module scope')
            if plan.get('adapter_base_weights_frozen') is not True:
                errors.append('Adapter base weights must remain frozen')
    return errors


def strict_json(text):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('Duplicate JSON key: ' + k)
            result[k] = v
        return result

    def bad_constant(s):
        raise ValueError('Invalid JSON constant: ' + s)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad_constant)


def get_contract(version):
    if version != SCHEMA_VERSION:
        raise ValueError('Unsupported data/checkpoint schema; prepare current Stage 4 data and train a new checkpoint')
    return SCHEMA_VERSION, SYSTEM_PROMPT, target_errors


def checkpoint_errors(checkpoint):
    """Static checks on a saved final/ or checkpoint-N/ folder (no model loading)."""
    from pathlib import Path
    checkpoint = Path(checkpoint)
    if not (checkpoint / 'run_config.json').is_file():
        return [f'checkpoint가 준비되지 않았습니다: {checkpoint} (run_config.json 없음). 학습 완료 후 final/ 또는 checkpoint-N/을 지정하세요.']
    manifest = read_json(checkpoint / 'run_config.json'); errors = []
    if manifest.get('schema_version') != SCHEMA_VERSION:
        errors.append(f'checkpoint schema {manifest.get("schema_version")!r} != {SCHEMA_VERSION!r}')
    if manifest.get('system_prompt') != SYSTEM_PROMPT or manifest.get('enable_thinking') is not False or manifest.get('text_only') is not True:
        errors.append('checkpoint prompt contract differs from this module')
    required = ['catalog.json', 'tokenizer_config.json', 'tokenizer.json']
    if manifest.get('finetuning') in {'lora', 'qlora'}:
        required += ['adapter_config.json', 'adapter_model.safetensors']
    else:
        required += ['config.json']
        if not any((checkpoint / n).is_file() for n in ['model.safetensors', 'model.safetensors.index.json']):
            errors.append('full-weight checkpoint is incomplete: ' + str(checkpoint))
    missing = [n for n in required if not (checkpoint / n).is_file() or (checkpoint / n).stat().st_size == 0]
    if missing:
        errors.append('checkpoint files missing or empty: ' + ', '.join(missing))
    if (checkpoint / 'catalog.json').is_file() and manifest.get('catalog_sha256') and file_hash(checkpoint / 'catalog.json') != manifest['catalog_sha256']:
        errors.append('saved catalog.json was modified')
    return errors
