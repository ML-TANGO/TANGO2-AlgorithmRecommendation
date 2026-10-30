"""Fail-closed request checks and case-only briefs, using the bundled audited requests."""
import copy
import unittest

from algorithm_recommend.paths import AUDITED_REQUESTS
from algorithm_recommend.revision.sources import load_audited_requests
from algorithm_recommend.stage4_requests.request_checks import inspect
from algorithm_recommend.stage4_requests.request_design import brief_for
from algorithm_recommend.stage4_requests.assemble import semantic_group


@unittest.skipUnless(AUDITED_REQUESTS.exists(), 'bundled audited requests missing')
class RequestChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records = load_audited_requests(AUDITED_REQUESTS)
        cls.cases = {cid: r['case'] for cid, r in records.items()}
        cls.requests = {cid: dict(case_id=cid, text=r['input_2'], metadata=r['expression_metadata']) for cid, r in records.items()}
        cls.example = next(r for r in cls.requests.values() if not inspect(r['text'], cls.cases[r['case_id']], r['metadata']))

    def errors(self, text):
        r = self.example; return inspect(text, self.cases[r['case_id']], r['metadata'])

    def append(self, s):
        return self.example['text'].replace('</USER_REQUIREMENTS>', s + '</USER_REQUIREMENTS>')

    def test_valid_request(self):
        self.assertFalse(self.errors(self.example['text']))

    def test_all_bundled_requests_pass_surface_checks(self):
        bad = [cid for cid, r in self.requests.items() if inspect(r['text'], self.cases[cid], r['metadata'])]
        self.assertEqual(bad, [])

    def test_memory_units_with_no_space(self):
        for s in ['32GB입니다.', 'VRAM은 32기가입니다.', '32 GiB로 진행합니다.']:
            self.assertIn('memory', self.errors(self.append(s)))

    def test_answer_names_with_korean_particles(self):
        for s in ['LoRA를 사용할게요.', 'QLoRA로 정했습니다.', 'SFT를 원해요.', 'Gemma를 선택했습니다.']:
            self.assertIn('answer_leak', self.errors(self.append(s)))

    def test_fake_negative_constraints(self):
        self.assertIn('unsupported_negatives', self.errors(self.append('검색 기능은 필요 없습니다.')))

    def test_cpu_condition_not_user_fact(self):
        self.assertIn('new_resources', self.errors(self.append('CPU를 사용하지 않습니다.')))

    def test_gpu_is_required(self):
        self.assertIn('gpu_alias_missing', self.errors(self.example['text'].replace(self.example['metadata']['gpu_alias'], '')))

    def test_invented_data_history(self):
        self.assertIn('invented_data_history', self.errors(self.append('제가 가지고 있는 자료들을 활용하겠습니다.')))

    def test_tts_budget_excludes_generated_speech(self):
        r = next(r for r in self.requests.values() if self.cases[r['case_id']]['task'] == 'TTS' and not inspect(r['text'], self.cases[r['case_id']], r['metadata']))
        text = r['text'].replace('</USER_REQUIREMENTS>', '입력과 출력 결과를 합친 총량입니다.</USER_REQUIREMENTS>')
        self.assertIn('tts_text_budget_includes_output', inspect(text, self.cases[r['case_id']], r['metadata']))

    def test_exact_tags(self):
        self.assertIn('tag_format', self.errors(self.example['text'].replace('DESCIRPTION', 'DESCRIPTION')))

    def test_case_only_brief_does_not_leak_memory(self):
        for case in list(self.cases.values())[:50]:
            before = copy.deepcopy(case); brief, meta = brief_for(case, 'case_test')
            self.assertEqual(case, before); self.assertNotIn('vram_gb', brief); self.assertNotIn('GB', meta['gpu_alias'])
            self.assertIs(meta['answer_fields_given_to_generator'], False)

    def test_ambiguous_gpu_capacities_stay_same_group(self):
        case = copy.deepcopy(next(c for c in self.cases.values() if c['gpu']['gpu_name'] == 'A100 40GB'))
        other = copy.deepcopy(case); other['gpu'].update(gpu_name='A100 80GB', full_name='NVIDIA A100 80GB', vram_gb=80)
        self.assertEqual(semantic_group(case), semantic_group(other))

    def test_nontext_context_does_not_leak_across_groups(self):
        case = copy.deepcopy(next(c for c in self.cases.values() if c['task'] == 'Detection')); other = copy.deepcopy(case)
        other['context_length'] = 131072 if case['context_length'] != 131072 else 64
        self.assertEqual(semantic_group(case), semantic_group(other))


if __name__ == '__main__':
    unittest.main()
