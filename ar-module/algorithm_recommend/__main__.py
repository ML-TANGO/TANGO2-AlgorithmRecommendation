"""Single entry point: python -m algorithm_recommend <command> [args...]"""
import importlib
import sys

COMMANDS = {
    'stage1': ('stage1_environments.generate', 'Stage 1: sample 5,000 four-field cases'),
    'stage1-validate': ('stage1_environments.validate', 'Stage 1: independent audit'),
    'stage2': ('stage2_labels.generate', 'Stage 2: rule-based model/algorithm/metric labels'),
    'stage2-validate': ('stage2_labels.validate', 'Stage 2: audit'),
    'stage3': ('stage3_hyperparameters.generate', 'Stage 3: hyperparameters and training plans'),
    'stage3-validate': ('stage3_hyperparameters.validate', 'Stage 3: audit'),
    'stage3-verify-arch': ('stage3_hyperparameters.verify_architecture', 'Stage 3: meta-device module path check (torch/transformers)'),
    'stage4-write': ('stage4_requests.local_writer', 'Stage 4: generate natural requests with the original Gemma (GPU)'),
    'stage4-audit': ('stage4_requests.local_auditor', 'Stage 4: label-blind request audit with the original Gemma (GPU)'),
    'stage4-assemble': ('stage4_requests.assemble', 'Stage 4: assemble input_2/label_2 from accepted requests + audits'),
    'stage4-validate': ('stage4_requests.validate', 'Stage 4: audit'),
    'stage4-diversity': ('stage4_requests.inspect_diversity', 'Stage 4: lexical diversity report'),
    'revise': ('revision.generate', 'Revised Stage 2/3: jointly feasible case/label/hyperparameter pairs'),
    'revise-requests': ('revision.requests', 'Revised Stage 4: fact-substituted requests + label_2 + splits'),
    'revise-validate': ('revision.validate', 'Revised Stage 2/3/4: full audit'),
    'revise-diversity': ('revision.check_diversity', 'Revised Stage 4: lexical diversity report'),
    'prepare': ('recommender.prepare_data', 'Stage 5: export chat-format SFT data'),
    'train': ('recommender.train', 'Stage 5: supervised fine-tuning (GPU)'),
    'infer': ('recommender.inference', 'Stage 5: recommendation from a trained checkpoint (GPU)'),
    'verify': ('recommender.verify', 'Stage 5: data/checkpoint integrity report'),
    'demo': ('recommender.demo.app', 'Web demo (FastAPI)'),
}


def usage():
    width = max(len(k) for k in COMMANDS)
    lines = ['usage: python -m algorithm_recommend <command> [options]', '', 'commands:']
    lines += [f'  {k.ljust(width)}  {v[1]}' for k, v in COMMANDS.items()]
    lines += ['', 'Each command accepts --help. Pipeline order: stage1 → stage2 → stage3 → (stage4-*) → revise → revise-requests → revise-validate → prepare → train → infer/demo']
    return '\n'.join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {'-h', '--help', 'help'}:
        print(usage()); return 0
    name, *rest = argv
    if name not in COMMANDS:
        print(f'Unknown command: {name}\n\n{usage()}', file=sys.stderr); return 2
    module = importlib.import_module(f'algorithm_recommend.{COMMANDS[name][0]}')
    result = module.main(rest)
    return result if isinstance(result, int) else 0


if __name__ == '__main__':
    raise SystemExit(main())
