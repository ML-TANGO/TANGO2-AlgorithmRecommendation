"""Read-only reuse of the audited Stage 2/3 policies for revised candidate search."""
import copy
import math

from ..io_utils import read, write, sha  # noqa: F401 (re-exported)
from ..paths import DATASET_DIR
from ..stage2_labels.feasibility_policy import Evaluator
from ..stage2_labels import labeling_core
from ..stage3_hyperparameters.generate import propose
from ..stage3_hyperparameters.context_lengths import load_context_lengths
from ..stage3_hyperparameters.architecture import module_spec  # noqa: F401 (re-exported)


def key(c):
    return (c['task'], c['gpu']['gpu_name'], c['language'], c['context_length'])


def gpu_key(g):
    return 'A100' if g['gpu_name'].startswith('A100 ') else g['gpu_name']


class JointPolicy:
    def __init__(self, dataset_dir=DATASET_DIR):
        self.evaluator = Evaluator(dataset_dir); self.profiles = self.evaluator.modelmap
        self.hp = read(dataset_dir / 'hyper_parameter_list.json'); self.contexts = load_context_lengths(dataset_dir / 'context_length_list.json')
        self.gpus = read(dataset_dir / 'gpu_list.json')['gpus']; self.gpu_map = {g['gpu_name']: g for g in self.gpus}
        self.languages = read(dataset_dir / 'language_list.json')
        self.cache = {}

    def effective_case(self, case):
        c = copy.deepcopy(case); c.pop('case_id', None)
        if gpu_key(c['gpu']) == 'A100':
            c['gpu'] = copy.deepcopy(self.gpu_map['A100 40GB'])
        return c

    def options(self, case):
        c = self.effective_case(case); k = key(c)
        if k in self.cache:
            return self.cache[k]
        base = self.evaluator.evaluate(c); out = []
        for check in base['candidate_checks']:
            if check['status'] != 'possible':
                continue
            mid = check['model']; p = self.profiles[mid]
            for algorithm in check['fitting_algorithms']:
                row = copy.deepcopy(base); row.update(case=c, training_possible=True, training_status='possible', impossibility_reason=None, reason_codes=[], review_notes=p['requirements_and_limits'].copy(), policy_version='revised-1.0', case_id='pending')
                row.pop('review_reason', None)
                mem = check['algorithm_estimates_gb'][algorithm]
                row['label'].update(model=mid, algorithm=algorithm)
                row['resource_check'] = dict(gpu_vram_gb=c['gpu']['vram_gb'], positive_budget_gb=.75 * c['gpu']['vram_gb'], estimated_memory_gb=mem, empirical_measurement=False, algorithm_estimates_gb=check['algorithm_estimates_gb'])
                row['training_objective'] = p['objective']; row['context_applicable'] = p['context_applicable']
                row['selected_workload_assumptions'] = dict(origin='system_calculation_policy', user_requested=False, values=copy.deepcopy(p['workload']))
                row['sources'] = list(dict.fromkeys(base['sources'] + p['training_sources']))
                draft = propose(row, self.hp, self.profiles, self.contexts)
                if draft['stage3_status'] != 'recommended_draft':
                    continue
                if p['profile_type'] == 'text':
                    score = labeling_core.model_score(p, c)[0]
                elif 'Qwen3-VL' in mid:
                    score = -8 - abs(math.log2(p['params_b_approx'] / 4))
                else:
                    score = math.log2(max(p['params_b_approx'], .001))
                if algorithm == 'QLoRA':
                    score -= .5
                row['rationale']['model'] = f"Task·언어·길이 지원 근거와 Stage 3 모듈별 메모리 계산을 통과한 후보입니다. 추정 {draft['resource_check']['estimated_memory_gb']:.3f}GB / 보수적 예산 {draft['resource_check']['positive_budget_gb']:.3f}GB. 실측 성공 판정은 아닙니다."
                row['rationale']['algorithm'] = f"{algorithm}: {p['objective']}에 필요한 모듈을 유지하고 해당 학습 범위·dtype·adapter 상태를 포함해 검토했습니다."
                out.append((score, row, draft))
        out.sort(key=lambda x: (-x[0], {'LoRA': 0, 'QLoRA': 1, 'SFT': 2}[x[1]['label']['algorithm']], x[1]['label']['model']))
        # Cache only viable compact choices; do not retain full failed enumerations.
        self.cache[k] = out
        return out

    def choose(self, case, preferred=None):
        options = self.options(case)
        if not options:
            return None
        chosen = next((o for o in options if preferred and o[1]['label'] == preferred), options[0]) if gpu_key(case['gpu']) != 'A100' else options[0]
        row = copy.deepcopy(chosen[1]); row['case'] = copy.deepcopy(case)
        row['gpu_resource_basis'] = {'visible_gpu_identity': gpu_key(case['gpu']), 'validated_vram_gb': [40, 80] if gpu_key(case['gpu']) == 'A100' else [case['gpu']['vram_gb']], 'budget_capacity_gb': self.effective_case(case)['gpu']['vram_gb'], 'rule': 'Fit every catalog capacity represented by the GPU name; no capacity is inferred from omitted input.'}
        # Actual A100 rank thresholds are identical at 40/80 GB. Recompute on the
        # actual case with the conservative shared resource budget to verify this.
        out = propose(row, self.hp, self.profiles, self.contexts)
        if out['stage3_status'] != 'recommended_draft':
            raise ValueError('Conservative GPU selection did not survive actual-case proposal')
        return row, out
