"""Text-only Gemma 4 / Qwen3 loading and exact chat-template tokenization."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Gemma4ForCausalLM
from .common import resolve_model
from .inference_output import MODALITY_PREFIXES, quiet_expected_modality_report


@dataclass
class ModelSpec:
    name: str
    family: str
    revision: str | None
    config: object
    source_is_multimodal: bool


def resolve_spec(name, revision='main', local_files_only=False):
    name = resolve_model(name)
    if Path(name).is_dir():
        name = str(Path(name).resolve()); revision = None
    config = AutoConfig.from_pretrained(name, revision=revision, local_files_only=local_files_only, trust_remote_code=False)
    if config.model_type in {'gemma4','gemma4_text'}:
        family = 'gemma4'
    elif config.model_type == 'qwen3':
        family = 'qwen3'
    else:
        raise ValueError(f'Unsupported model_type={config.model_type!r}. Use Gemma 4 or Qwen3 (not Gemma 3n).')
    revision = getattr(config, '_commit_hash', None) or revision
    return ModelSpec(name,family,revision,config,config.model_type=='gemma4')


def load_tokenizer(spec, *, path=None, training=True, local_files_only=False):
    tokenizer = AutoTokenizer.from_pretrained(
        str(path) if path else spec.name, revision=None if path else spec.revision,
        local_files_only=local_files_only, use_fast=True, trust_remote_code=False,
    )
    if not tokenizer.chat_template:
        raise ValueError('The model tokenizer has no official chat template; refusing to guess one.')
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer must define EOS or PAD; no new tokens are added automatically.')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right' if training else 'left'
    return tokenizer


def render_chat(tokenizer, messages, *, generation_prompt):
    # The training data contains final recommendations, not thinking traces.
    # This flag is independent of the user's requested reasoning capability.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt, enable_thinking=False)


def encode_example(record, tokenizer, max_seq_length):
    messages = record['messages']
    if [m['role'] for m in messages] != ['system','user','assistant']:
        raise ValueError(f'{record["case_id"]}: expected one system/user/assistant conversation')
    prefix = render_chat(tokenizer, messages[:-1], generation_prompt=True)
    full = render_chat(tokenizer, messages, generation_prompt=False)
    if not full.startswith(prefix):
        raise ValueError(f'{record["case_id"]}: train/inference template prefixes differ')
    # The official template already supplies BOS and turn boundaries.
    prompt_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    ids = tokenizer(full, add_special_tokens=False)['input_ids']
    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f'{record["case_id"]}: tokenizer merges across answer boundary; refusing unsafe loss masking')
    if len(ids) > max_seq_length:
        raise ValueError(f'{record["case_id"]}: {len(ids)} tokens exceeds --max-seq-length {max_seq_length}; increase it. No answer is silently truncated.')
    if len(ids) <= len(prompt_ids):
        raise ValueError(f'{record["case_id"]}: no supervised answer tokens')
    return {'input_ids':ids, 'attention_mask':[1]*len(ids), 'labels':[-100]*len(prompt_ids)+ids[len(prompt_ids):]}


class CompletionCollator:
    """Right-pad by position; EOS supervision remains intact even if PAD == EOS."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        length = max(len(f['input_ids']) for f in features)
        data = {k:[] for k in ['input_ids','attention_mask','labels']}
        for feature in features:
            n = length-len(feature['input_ids'])
            for key, padding in [('input_ids',self.pad_token_id),('attention_mask',0),('labels',-100)]:
                data[key].append(feature[key]+[padding]*n)
        return {k:torch.tensor(v,dtype=torch.long) for k,v in data.items()}


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, records, tokenizer, max_seq_length):
        self.records = [encode_example(r,tokenizer,max_seq_length) for r in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def choose_dtype(name, device):
    if name == 'auto':
        name = 'bf16' if str(device).startswith('cuda') and torch.cuda.is_bf16_supported() else 'fp16' if str(device).startswith('cuda') else 'fp32'
    if name == 'bf16' and str(device).startswith('cuda') and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bfloat16; select --dtype fp16')
    if name == 'fp16' and str(device) == 'cpu':
        raise ValueError('Use --dtype fp32 for CPU execution')
    return {'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[name]


def is_unused_shared_kv_weight(key, spec):
    """Old Gemma 4 checkpoints include projections unused by KV-sharing layers."""
    if spec.family != 'gemma4':
        return False
    match = re.fullmatch(r'model\.(?:language_model\.)?layers\.(\d+)\.self_attn\.(?:k_proj|v_proj|k_norm)\.(?:weight|bias)', key)
    if not match:
        return False
    config = spec.config.get_text_config()
    first_shared = config.num_hidden_layers-config.num_kv_shared_layers
    return first_shared <= int(match.group(1)) < config.num_hidden_layers


def load_text_model(spec, *, dtype, device='cpu', quantized=False, attention='sdpa', local_files_only=False, quiet_expected_modalities=False):
    if quantized and not str(device).startswith('cuda'):
        raise ValueError('This QLoRA path requires a CUDA device')
    options = dict(revision=spec.revision, dtype=dtype, device_map={'':str(device)},
                   attn_implementation=attention, local_files_only=local_files_only,
                   trust_remote_code=False, output_loading_info=True)
    if quantized:
        options['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                                                           bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    if spec.family == 'gemma4':
        # Instantiate ONLY the text architecture. Neither visual/audio towers nor
        # their projectors are constructed. Keep E4B per-layer text embeddings.
        options['config'] = spec.config.get_text_config()
        if spec.source_is_multimodal:
            options['key_mapping'] = {r'^model\.language_model\.':'model.'}
        with quiet_expected_modality_report(quiet_expected_modalities):
            model, info = Gemma4ForCausalLM.from_pretrained(spec.name, **options)
    else:
        model, info = AutoModelForCausalLM.from_pretrained(spec.name, **options)
    missing = sorted(info.get('missing_keys', []))
    mismatched = list(info.get('mismatched_keys', []))
    if missing or mismatched or info.get('error_msgs'):
        raise RuntimeError(f'Incomplete pretrained text weights: missing={missing[:8]}, mismatched={mismatched[:8]}, errors={info.get("error_msgs")}')
    if any(any(term in name for term in ['vision_tower','audio_tower','embed_vision','embed_audio']) for name,_ in model.named_modules()):
        raise RuntimeError('A non-text encoder was unexpectedly instantiated')
    unexpected = sorted(info.get('unexpected_keys', []))
    allowed = MODALITY_PREFIXES
    unexplained = [k for k in unexpected if not k.startswith(allowed) and not is_unused_shared_kv_weight(k,spec)]
    if unexplained:
        raise RuntimeError(f'Unexpected non-modality weights: {unexplained[:8]}')
    model.config.use_cache = False
    return model, {'model_class':type(model).__name__, 'missing_keys':missing,
                   'ignored_modality_weight_count':sum(k.startswith(allowed) for k in unexpected),
                   'ignored_shared_kv_weight_count':sum(is_unused_shared_kv_weight(k,spec) for k in unexpected), 'text_only':True,
                   'parameter_count':sum(p.numel() for p in model.parameters())}


def configure_adapters(model, method, *, rank=16, alpha=32, dropout=0.05, gradient_checkpointing=True):
    if method == 'full':
        return model
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    if method == 'qlora':
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing,
                                               gradient_checkpointing_kwargs={'use_reentrant':False})
    # Both supported text architectures expose these attention projections.
    # In Gemma 4 some KV-sharing layers legitimately omit k_proj/v_proj.
    targets = ['q_proj','k_proj','v_proj','o_proj']
    present = {name.rsplit('.',1)[-1] for name,_ in model.named_modules()}
    if not set(targets).issubset(present):
        raise ValueError('Expected attention projection modules are missing')
    config = LoraConfig(task_type=TaskType.CAUSAL_LM,r=rank,lora_alpha=alpha,lora_dropout=dropout,
                        target_modules=targets,bias='none')
    model = get_peft_model(model,config)
    return model


def generation_end_ids(tokenizer, family):
    ids = [tokenizer.eos_token_id]
    turn_end = '<turn|>' if family == 'gemma4' else '<|im_end|>'
    vocab = tokenizer.get_vocab()
    if turn_end not in vocab:
        raise ValueError(f'Expected official turn terminator {turn_end!r} is missing')
    ids.append(vocab[turn_end])
    return list(dict.fromkeys(i for i in ids if i is not None))
