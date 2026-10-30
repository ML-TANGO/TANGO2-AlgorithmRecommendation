"""Config-derived module specifications; not a loaded-checkpoint verification."""
import json
from functools import lru_cache

from ..paths import SOURCE_CONFIGS, SOURCE_EVIDENCE, rel

FALLBACK = {
 'google/gemma-3-1b-it': dict(hidden_size=1152, intermediate_size=6912, num_hidden_layers=26, num_attention_heads=4, num_key_value_heads=1, head_dim=256, vocab_size=262144, tie_word_embeddings=True),
 'google/gemma-3-4b-it': dict(hidden_size=2560, intermediate_size=10240, num_hidden_layers=34, num_attention_heads=8, num_key_value_heads=4, head_dim=256, vocab_size=262144, tie_word_embeddings=True),
 'meta-llama/Llama-3.2-3B-Instruct': dict(hidden_size=3072, intermediate_size=8192, num_hidden_layers=28, num_attention_heads=24, num_key_value_heads=8, head_dim=128, vocab_size=128256, tie_word_embeddings=True),
}


@lru_cache(maxsize=None)
def language_config(mid):
    path = SOURCE_CONFIGS / (mid.replace('/', '__') + '.json')
    if path.exists():
        c = json.loads(path.read_text(encoding='utf-8'))['config']
        if 'thinker_config' in c:
            c = c['thinker_config']
        c = c.get('text_config', c)
        source = rel(path)
    else:
        c = FALLBACK[mid].copy()
        source = rel(SOURCE_EVIDENCE / ('gemma_config.py' if mid.startswith('google/') else 'llama_skus.py'))
    return c, source


@lru_cache(maxsize=None)
def module_spec(mid):
    c, source = language_config(mid)
    h = c['hidden_size']; d = c.get('head_dim') or h // c['num_attention_heads']
    q = d * c['num_attention_heads']; kv = d * c['num_key_value_heads']; ff = c['intermediate_size']
    prefix = 'thinker.model.layers' if 'Qwen3-ASR' in mid else 'model.language_model.layers' if 'Qwen3-VL' in mid else 'model.layers'
    shapes = {'self_attn.q_proj': (h, q), 'self_attn.k_proj': (h, kv), 'self_attn.v_proj': (h, kv), 'self_attn.o_proj': (q, h),
              'mlp.gate_proj': (h, ff), 'mlp.up_proj': (h, ff), 'mlp.down_proj': (ff, h)}
    modules = [dict(name=f'{prefix}.{i}.{leaf}', in_features=ins, out_features=outs) for i in range(c['num_hidden_layers']) for leaf, (ins, outs) in shapes.items()]
    return dict(model=mid, config_source=source, scope_prefix=prefix, layers=c['num_hidden_layers'], modules=modules,
                derivation='Published architecture/config and selected HF text wrapper; verify against actual named_modules before applying', runtime_verified=False)


def select_modules(spec, targets):
    chosen = [m for m in spec['modules'] if targets == ['all-linear'] or m['name'].rsplit('.', 1)[-1] in targets]
    if targets != ['all-linear'] and set(targets) != {m['name'].rsplit('.', 1)[-1] for m in chosen}:
        raise ValueError('Missing adapter target')
    return chosen


def memory_check(row, profile, hp, spec=None):
    algo = row['label']['algorithm']; n = profile['memory_params_b'] * 1e9
    old = row['resource_check']['estimated_memory_gb']; factor = {'SFT': 18., 'LoRA': 2.5, 'QLoRA': .8}[algo]
    overhead = max(0, old - factor * n / 1e9)
    adapter_count = 0; quantized = 0
    if spec:
        chosen = select_modules(spec, hp['target_modules'])
        adapter_count = sum(hp['lora_r'] * (m['in_features'] + m['out_features']) for m in chosen)
    if algo == 'SFT':
        state = 18 * n
    elif algo == 'LoRA':
        state = 2 * n + 18 * adapter_count
    else:
        quantized = sum(m['in_features'] * m['out_features'] for m in spec['modules'])
        if quantized > n:
            raise ValueError('Quantization parameter count exceeds total profile count')
        # prepare_model_for_kbit_training may upcast ALL non-quantized weights
        # to FP32. Budget those at 4 bytes, rather than pretending all weights
        # (especially large embeddings/head) are 4-bit.
        state = .625 * quantized + 4 * (n - quantized) + 18 * adapter_count
    estimate = max(old, state / 1e9 + overhead)
    budget = row['resource_check']['positive_budget_gb']
    return dict(estimated_memory_gb=round(estimate, 6), positive_budget_gb=budget, within_budget=estimate <= budget,
                stage2_estimated_memory_gb=old, adapter_parameter_count=adapter_count,
                quantized_linear_parameter_count=quantized, non_quantized_parameter_count=round(n - quantized),
                parameter_state_gb=round(state / 1e9, 6), activation_and_reserve_gb=round(overhead, 6),
                precision_accounting='QLoRA non-quantized weights conservatively FP32 after k-bit preparation; adapter state 18 bytes/parameter; LoRA frozen base 2 bytes/parameter',
                empirical_measurement=False, limitation='Uncalibrated planning estimate. Activation/backend/sequence differences require a real forward/backward memory check.')
