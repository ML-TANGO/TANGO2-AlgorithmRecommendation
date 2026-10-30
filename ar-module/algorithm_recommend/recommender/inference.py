"""Generate label_2 using the actual fine-tuned Gemma text model."""
import argparse
import json
import sys
from pathlib import Path

from .common import SCHEMA_VERSION, SYSTEM_PROMPT, read_json, resolve_model, file_hash, build_input, strict_json, target_errors, checkpoint_errors
from .inference_output import save_prediction
from ..paths import DEFAULT_CHECKPOINT, INFERENCE_RESULTS


class Recommender:
    def __init__(self, checkpoint, *, device='cuda:0', dtype='auto', local_files_only=False, attention='sdpa', load_in_4bit=None, model=None, show_loading_report=False):
        import torch
        from .modeling import choose_dtype, generation_end_ids, load_text_model, load_tokenizer, render_chat, resolve_spec
        self._render_chat = render_chat
        self.path = Path(checkpoint).expanduser().resolve(); self.manifest = read_json(self.path / 'run_config.json')
        if self.manifest.get('schema_version') != SCHEMA_VERSION or self.manifest.get('text_only') is not True:
            raise ValueError('A checkpoint trained by this module (final/ or checkpoint-N/) is required; other schemas are incompatible')
        if self.manifest.get('system_prompt') != SYSTEM_PROMPT or self.manifest.get('enable_thinking') is not False:
            raise ValueError('Training/inference prompt contract differs')
        if model is not None and resolve_model(model) != self.manifest['base_model']:
            raise ValueError('--model does not match the trained base model')
        adapter = (self.path / 'adapter_config.json').exists()
        if adapter != (self.manifest['finetuning'] in {'lora', 'qlora'}):
            raise ValueError('Checkpoint format and training metadata disagree')
        if not adapter and not (self.path / 'config.json').exists():
            raise ValueError('Point to final/ or checkpoint-N/, not the run directory')
        source = self.manifest['base_model'] if adapter else str(self.path)
        spec = resolve_spec(source, self.manifest['base_revision'] if adapter else None, local_files_only)
        if spec.family != self.manifest['model_family']:
            raise ValueError('Model family mismatch')
        if str(device).startswith('cuda'):
            if not torch.cuda.is_available():
                raise ValueError('CUDA is unavailable; choose --device cpu')
            torch.cuda.set_device(torch.device(device))
        quantized = self.manifest['finetuning'] == 'qlora' if load_in_4bit is None else load_in_4bit
        self.tokenizer = load_tokenizer(spec, path=self.path, training=False, local_files_only=True)
        if self.tokenizer.chat_template != self.manifest['chat_template']:
            raise ValueError('Saved tokenizer template differs from training')
        if file_hash(self.path / 'catalog.json') != self.manifest['catalog_sha256']:
            raise ValueError('Saved catalog was modified')
        self.catalog = read_json(self.path / 'catalog.json')
        self.model, self.loading_report = load_text_model(spec, dtype=choose_dtype(dtype, device), device=device, quantized=quantized, attention=attention, local_files_only=local_files_only, quiet_expected_modalities=not show_loading_report)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(self.path), is_trainable=False)
        self.model.eval(); self.model.config.use_cache = True
        self.context_limit = spec.config.get_text_config().max_position_embeddings
        self.end_ids = generation_end_ids(self.tokenizer, spec.family)

    def predict(self, task_description, requirements, *, max_new_tokens=4096, temperature=0.0):
        import torch
        with torch.inference_mode():
            return self._predict(task_description, requirements, max_new_tokens=max_new_tokens, temperature=temperature)

    def _predict(self, task_description, requirements, *, max_new_tokens, temperature):
        prompt = build_input(task_description, requirements)
        if max_new_tokens <= 0 or temperature < 0:
            raise ValueError('Invalid max-new-tokens or temperature')
        rendered = self._render_chat(self.tokenizer, [{'role': 'system', 'content': self.manifest['system_prompt']}, {'role': 'user', 'content': prompt}], generation_prompt=True)
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_tensors='pt', return_token_type_ids=False)
        count = encoded['input_ids'].shape[1]
        if count + max_new_tokens > self.context_limit:
            raise ValueError(f'Prompt ({count}) + output budget ({max_new_tokens}) exceeds model context {self.context_limit}; no truncation was performed')
        encoded = {k: v.to(self.model.get_input_embeddings().weight.device) for k, v in encoded.items()}
        options = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0, eos_token_id=self.end_ids, pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
        options.update(temperature=temperature, top_p=.9, top_k=20) if temperature > 0 else options.update(temperature=None, top_p=None, top_k=None)
        ids = self.model.generate(**encoded, **options)[0, count:]
        raw = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        finished = bool(len(ids) and int(ids[-1]) in self.end_ids)
        try:
            parsed = strict_json(raw); errors = target_errors(parsed, self.catalog)
            if isinstance(parsed, dict):
                thinking = parsed.get('thinking')
                if isinstance(thinking, dict) and thinking.get('recommended') is True and (not isinstance(thinking.get('request_evidence'), str) or thinking['request_evidence'] not in prompt):
                    errors.append('thinking evidence is not a literal part of the user request')
        except (ValueError, TypeError) as exc:
            parsed = None; errors = ['Invalid JSON: ' + str(exc)]
        if not finished:
            errors.append('Generation ended without an end-of-turn token; increase --max-new-tokens')
        return {'valid': not errors, 'label_2': parsed, 'errors': errors, 'input': {'user_task_description': task_description, 'user_requirements': requirements},
                'input_2': prompt, 'raw_output': raw, 'input_tokens': count, 'generated_tokens': len(ids), 'finished': finished,
                'model_generation_performed': True, 'checkpoint': str(self.path),
                'validation_scope': 'JSON structure and saved catalog; not proof of task suitability, memory fit, or training success'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT, help=f'final/ or checkpoint-N/ (default: {DEFAULT_CHECKPOINT}; env AR_CHECKPOINT)')
    a = p.add_mutually_exclusive_group(); a.add_argument('--task-description', '--user-task-description', dest='task_description'); a.add_argument('--task-description-file', type=Path)
    b = p.add_mutually_exclusive_group(); b.add_argument('--requirements', '--user-requirements', dest='requirements'); b.add_argument('--requirements-file', type=Path)
    p.add_argument('--interactive', action='store_true', help='Ask for missing fields (also automatic when fields are omitted); one line each. Use file options for multiline input.')
    p.add_argument('--device', default='cuda:0'); p.add_argument('--dtype', choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto')
    p.add_argument('--model', help='Optional exact base-model consistency check')
    p.add_argument('--local-files-only', action='store_true'); p.add_argument('--attention', choices=['sdpa', 'eager'], default='sdpa')
    p.add_argument('--load-in-4bit', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--show-loading-report', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=4096); p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--output', type=Path, help=f'Default: automatically named JSON in {INFERENCE_RESULTS}/')
    return p


def missing_fields(args):
    return ((args.task_description is None and args.task_description_file is None)
            or (args.requirements is None and args.requirements_file is None))


def collect_fields(args):
    task = args.task_description_file.read_text(encoding='utf-8') if args.task_description_file else args.task_description
    req = args.requirements_file.read_text(encoding='utf-8') if args.requirements_file else args.requirements
    if args.interactive or missing_fields(args):
        try:
            if task is None:
                task = input('USER_TASK_DESCRIPTION — 작업 설명: ')
            if req is None:
                req = input('USER_REQUIREMENTS — 요구사항: ')
        except EOFError as exc:
            raise ValueError('입력이 완료되지 않았습니다. 두 인자 또는 UTF-8 파일 옵션을 사용하세요.') from exc
    build_input(task, req)  # Validate both fields before loading large model weights.
    return task, req


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if args.max_new_tokens <= 0 or args.temperature < 0:
        p.error('max-new-tokens는 양수, temperature는 0 이상이어야 합니다.')
    problems = checkpoint_errors(args.checkpoint)
    if problems:
        p.error('\n'.join(problems))
    try:
        task, req = collect_fields(args)
    except ValueError as exc:
        p.error(str(exc))
    print(f'Checkpoint: {Path(args.checkpoint).resolve()}', file=sys.stderr, flush=True)
    recommender = Recommender(args.checkpoint, device=args.device, dtype=args.dtype, model=args.model, local_files_only=args.local_files_only, attention=args.attention, load_in_4bit=args.load_in_4bit, show_loading_report=args.show_loading_report)
    result = recommender.predict(task, req, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    saved = save_prediction(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2)); print(f'결과 JSON 저장: {saved}', file=sys.stderr)
    if not result['valid']:
        raise SystemExit(2)
    return result


if __name__ == '__main__':
    main()
