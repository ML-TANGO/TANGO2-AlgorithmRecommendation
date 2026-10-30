"""Verify language module paths/shapes on weight-free meta-device skeletons.

Requires torch/transformers. No checkpoint downloads, GPU allocation, pretrained
weights or training. Qwen ASR custom package is not installed; its inventory is
source-reviewed and explicitly not runtime checked.
"""
import argparse
import json
from pathlib import Path

from .architecture import language_config
from .training_scopes import resolve_and_verify_adapter_targets
from ..paths import STAGE3


def verify(output_dir=STAGE3):
    import torch
    import transformers
    output_dir = Path(output_dir)
    specs = json.loads((output_dir / 'module_specs.json').read_text(encoding='utf-8')); results = {}
    for mid, spec in specs.items():
        if 'Qwen3-ASR' in mid:
            results[mid] = {'status': 'source_review_only', 'reason': 'qwen_asr package not installed; official architecture snapshot retained'}
            continue
        c, _ = language_config(mid); c = dict(c)
        if mid.startswith('google/gemma-3'):
            cfg = transformers.Gemma3TextConfig(**c)
            cls = transformers.Gemma3ForCausalLM
        else:
            kind = c.pop('model_type', 'llama' if 'Llama' in mid else 'qwen2' if 'DeepSeek' in mid else 'qwen3')
            cfg = transformers.AutoConfig.for_model(kind, **c)
            cls = transformers.AutoModelForCausalLM
        with torch.device('meta'):
            model = cls.from_config(cfg) if cls is transformers.AutoModelForCausalLM else cls(cfg)
        linear = {n: (m.in_features, m.out_features) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
        resolved = resolve_and_verify_adapter_targets(linear, spec, ['all-linear'])
        results[mid] = {'status': 'meta_architecture_verified', 'linear_modules_in_scope': len(resolved['module_names']),
                        'pretrained_weights_loaded': False, 'forward_backward_run': False}
        del model
    report = {'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
              'results': results, 'checkpoint_weights_verified': False, 'actual_gpu_training_performed': False}
    (output_dir / 'architecture_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=STAGE3)
    return verify(p.parse_args(argv).output_dir)


if __name__ == '__main__':
    main()
