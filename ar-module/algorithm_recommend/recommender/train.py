"""Supervised fine-tuning of the text recommendation model (full / LoRA / QLoRA)."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import os
import statistics
from pathlib import Path

from .common import file_hash, read_json, read_jsonl, write_json, get_contract
from ..paths import DATA_DIR, OUTPUTS_DIR, resolve, rel


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='gemma', help='gemma / qwen / Hugging Face ID / local model directory')
    p.add_argument('--revision', default='main')
    p.add_argument('--data-dir', type=Path, default=DATA_DIR)
    p.add_argument('--output-dir', type=Path, help=f'Default: {OUTPUTS_DIR}/<family>-<finetuning>')
    p.add_argument('--finetuning', choices=['full', 'lora', 'qlora'], default='lora', help='All three optimize the same supervised answer-only loss')
    p.add_argument('--max-seq-length', type=int, default=4096, help='Recommendation prompt + answer length; NOT the context_length inside a case')
    p.add_argument('--loss-chunk-size', type=int, default=128, help='Limit LM-head logits memory during SFT')
    p.add_argument('--epochs', type=float, default=3)
    p.add_argument('--learning-rate', type=float, help='Default: full=2e-5, adapters=2e-4')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--eval-batch-size', type=int, default=1)
    p.add_argument('--gradient-accumulation-steps', type=int, default=8)
    p.add_argument('--warmup-ratio', type=float, default=0.03)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--lora-rank', type=int, default=16)
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--lora-dropout', type=float, default=0.05)
    p.add_argument('--dtype', choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto')
    p.add_argument('--attention', choices=['sdpa', 'eager'], default='sdpa')
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--max-train-samples', type=int)
    p.add_argument('--max-eval-samples', type=int)
    p.add_argument('--logging-steps', type=int, default=10)
    p.add_argument('--save-steps', type=int, default=100)
    p.add_argument('--save-total-limit', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume-from-checkpoint', type=Path)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--require-sources', action='store_true', help='Fail when the upstream Stage 4 artifacts recorded in preparation_report.json are absent (default: verify those that exist)')
    p.add_argument('--dry-run', action='store_true', help='Validate/tokenize all three splits without loading model weights or training')
    return p


def check_prepared_sources(prepared, data_dir, require_sources):
    """Verify upstream hashes; shipped data may not carry its upstream artifacts."""
    absent = []
    for filename, digest in prepared['source_sha256'].items():
        path = resolve(filename)
        if not path.exists():
            absent.append(filename); continue
        if file_hash(path) != digest:
            raise ValueError(f'Data source changed: {filename}; regenerate data')
    if absent and require_sources:
        raise ValueError('Upstream sources missing: ' + ', '.join(absent))
    for filename, digest in prepared['data_sha256'].items():
        if file_hash(data_dir / filename) != digest:
            raise ValueError(f'{filename} changed after preparation; regenerate or review it')
    return absent


def main(argv=None):
    args = parser().parse_args(argv)
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from .sft_loss import answer_loss
    from .modeling import (CompletionCollator, EncodedDataset, choose_dtype, configure_adapters,
                           load_text_model, load_tokenizer, resolve_spec)

    class TextSFTTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            with self.accelerator.autocast():
                loss = answer_loss(model, inputs, chunk_size=self.loss_chunk_size, num_items_in_batch=num_items_in_batch)
            return (loss, {'loss': loss}) if return_outputs else loss

        def _save(self, output_dir=None, state_dict=None):
            # Gemma's source uses model.language_model.*, while this artifact is
            # gemma4_text with model.*. Do not reverse the loading conversion when
            # saving full weights, or a later text-only load would miss all weights.
            folder = Path(output_dir or self.args.output_dir)
            folder.mkdir(parents=True, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            model.save_pretrained(folder, state_dict=state_dict, save_original_format=False)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(folder)
            torch.save(self.args, folder / 'training_args.bin')

    class SaveRunMetadata(TrainerCallback):
        def __init__(self, manifest, catalog, tokenizer):
            self.manifest, self.catalog, self.tokenizer = manifest, catalog, tokenizer

        def save(self, folder):
            write_json(folder / 'run_config.json', self.manifest)
            write_json(folder / 'catalog.json', self.catalog)
            self.tokenizer.save_pretrained(folder)

        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                self.save(Path(args.output_dir) / f'checkpoint-{state.global_step}')

    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This verified training path supports one process/device; distributed training is not implemented')
    for name in ['loss_chunk_size', 'max_seq_length', 'batch_size', 'eval_batch_size', 'gradient_accumulation_steps', 'lora_rank', 'lora_alpha', 'logging_steps', 'save_steps', 'save_total_limit']:
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    for name in ['max_train_samples', 'max_eval_samples']:
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.lora_dropout < 1 or args.epochs <= 0:
        raise ValueError('Invalid warmup/dropout/epochs')
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError('max_steps must be -1 or positive')
    set_seed(args.seed)
    spec = resolve_spec(args.model, args.revision, args.local_files_only)
    output = (args.output_dir or OUTPUTS_DIR / f'{spec.family}-{args.finetuning}').resolve()
    if not args.dry_run and output.exists() and not args.resume_from_checkpoint:
        existing = {p.name for p in output.iterdir()}
        if existing - {'tokenization_report.json'}:
            raise ValueError(f'{output} is not empty; select a new --output-dir or --resume-from-checkpoint')
    tokenizer = load_tokenizer(spec, local_files_only=args.local_files_only)
    data_dir = Path(args.data_dir).resolve()
    catalog = read_json(data_dir / 'catalog.json')
    prepared = read_json(data_dir / 'preparation_report.json')
    schema_version, system_prompt, validate_target = get_contract(prepared['schema_version'])
    if prepared['system_prompt'] != system_prompt:
        raise ValueError('Prepared schema/system prompt differs; run prepare_data again')
    if prepared.get('catalog_sha256') and file_hash(data_dir / 'catalog.json') != prepared['catalog_sha256']:
        raise ValueError('Prepared catalog changed; regenerate data')
    absent_sources = check_prepared_sources(prepared, data_dir, args.require_sources)
    if absent_sources:
        print(f'Note: {len(absent_sources)} upstream Stage 4 artifacts are not present; verified prepared data hashes only.', flush=True)
    context_limit = spec.config.get_text_config().max_position_embeddings
    if args.max_seq_length > context_limit:
        raise ValueError(f'--max-seq-length exceeds model limit {context_limit}')
    datasets = {}; report = {'model': spec.name, 'family': spec.family, 'revision': spec.revision,
                             'max_seq_length': args.max_seq_length, 'enable_thinking': False, 'splits': {}}
    seen = set()
    for split in ['train', 'validation', 'test']:
        records = read_jsonl(data_dir / f'{split}.jsonl')
        all_ids = [r['case_id'] for r in records]
        if len(all_ids) != len(set(all_ids)) or seen.intersection(all_ids):
            raise ValueError('Case leakage or duplicate ID in prepared splits')
        seen.update(all_ids)
        if split == 'test' and not args.dry_run:
            continue  # Never pass held-out test examples to Trainer.
        limit = args.max_train_samples if split == 'train' else args.max_eval_samples
        if limit is not None:
            records = records[:limit]
        if not records:
            raise ValueError(f'{split} is empty')
        if any(r['messages'][0] != {'role': 'system', 'content': system_prompt} for r in records):
            raise ValueError('Unexpected system prompt in prepared data')
        for record in records:
            errors = validate_target(json.loads(record['messages'][2]['content']), catalog)
            if errors:
                raise ValueError(f'{record["case_id"]}: invalid training target: {errors}')
        datasets[split] = EncodedDataset(records, tokenizer, args.max_seq_length)
        lengths = [len(r['input_ids']) for r in datasets[split].records]
        answers = [sum(t != -100 for t in r['labels']) for r in datasets[split].records]
        report['splits'][split] = {'cases': len(lengths), 'min_tokens': min(lengths), 'max_tokens': max(lengths),
                                   'mean_tokens': statistics.mean(lengths), 'min_answer_tokens': min(answers)}
    write_json(output / 'tokenization_report.json', report)
    print(report, flush=True)
    if args.dry_run:
        return report
    if not args.cpu and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable. Use a CUDA PyTorch build or --cpu for small tests.')
    if not args.cpu and torch.cuda.device_count() > 1:
        raise ValueError('Select exactly one GPU with CUDA_VISIBLE_DEVICES; multi-GPU DataParallel is not supported')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = 'cpu' if args.cpu else f'cuda:{local_rank}'
    if not args.cpu:
        torch.cuda.set_device(local_rank)
    dtype = choose_dtype(args.dtype, device)
    load_dtype = torch.float32 if args.finetuning == 'full' and dtype == torch.float16 else dtype
    model, loading = load_text_model(spec, dtype=load_dtype, device=device, quantized=args.finetuning == 'qlora',
                                     attention=args.attention, local_files_only=args.local_files_only, quiet_expected_modalities=True)
    model = configure_adapters(model, args.finetuning, rank=args.lora_rank, alpha=args.lora_alpha,
                               dropout=args.lora_dropout, gradient_checkpointing=args.gradient_checkpointing)
    model.config.pad_token_id = tokenizer.pad_token_id
    lr = args.learning_rate if args.learning_rate is not None else 2e-5 if args.finetuning == 'full' else 2e-4
    if lr <= 0:
        raise ValueError('learning_rate must be positive')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    updates_per_epoch = math.ceil(math.ceil(len(datasets['train']) / (world * args.batch_size)) / args.gradient_accumulation_steps)
    planned_steps = args.max_steps if args.max_steps > 0 else math.ceil(updates_per_epoch * args.epochs)
    manifest = {
        'schema_version': schema_version, 'base_model': spec.name, 'base_revision': spec.revision, 'model_family': spec.family,
        'finetuning': args.finetuning, 'enable_thinking': False, 'text_only': True, 'system_prompt': system_prompt,
        'max_seq_length': args.max_seq_length, 'dtype': str(dtype), 'loading_report': loading,
        'tokenizer_class': type(tokenizer).__name__, 'pad_token_id': tokenizer.pad_token_id,
        'chat_template': tokenizer.chat_template,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'data_sha256': prepared['data_sha256'], 'source_sha256': prepared['source_sha256'], 'catalog_sha256': prepared['catalog_sha256'],
        'library_versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'peft', 'accelerate']},
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'loss_implementation': 'chunked_supervised_causal_cross_entropy',
        'artifact_role': 'training_checkpoint', 'data_dir': rel(data_dir),
        'learning_rate': lr, 'warmup_steps': int(planned_steps * args.warmup_ratio),
    }
    if args.resume_from_checkpoint:
        previous = read_json(args.resume_from_checkpoint / 'run_config.json')
        for key in ['base_model', 'base_revision', 'finetuning', 'data_sha256', 'max_seq_length', 'system_prompt', 'chat_template', 'catalog_sha256']:
            if previous[key] != manifest[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
    save_metadata = SaveRunMetadata(manifest, catalog, tokenizer)
    train_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, lr_scheduler_type='cosine',
        warmup_steps=manifest['warmup_steps'], weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16, use_cpu=args.cpu,
        gradient_checkpointing=args.gradient_checkpointing, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='epoch', save_strategy='steps', save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps, logging_first_step=True, report_to='none',
        optim='adamw_torch', seed=args.seed, data_seed=args.seed, remove_unused_columns=False,
        ddp_find_unused_parameters=False if world > 1 else None, prediction_loss_only=True,
    )
    trainer = TextSFTTrainer(model=model, args=train_args, train_dataset=datasets['train'], eval_dataset=datasets['validation'],
                             data_collator=CompletionCollator(tokenizer.pad_token_id), processing_class=tokenizer,
                             callbacks=[save_metadata])
    trainer.loss_chunk_size = args.loss_chunk_size
    trainer.model_accepts_loss_kwargs = True  # answer_loss uses the accumulated supervised-token count
    if trainer.is_world_process_zero():
        save_metadata.save(output)
    result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    final = output / 'final'
    trainer.save_model(str(final))
    if trainer.is_world_process_zero():
        save_metadata.save(final)
        write_json(output / 'train_metrics.json', result.metrics)
    trainer.save_state()
    print(f'Saved trained artifact: {final}', flush=True)
    return result.metrics


if __name__ == '__main__':
    main()
