"""Revised pipeline: fact substitution reproduces the generated revised requests."""
import unittest

from algorithm_recommend.io_utils import read
from algorithm_recommend.paths import REV2, REV4, AUDITED_REQUESTS, MANUAL_CORRECTIONS
from algorithm_recommend.revision.sources import load_audited_requests
from algorithm_recommend.revision.requests import revise_request
from algorithm_recommend.revision.thinking_policy import reason_for
from algorithm_recommend.stage4_requests.request_checks import inspect


class ThinkingPolicy(unittest.TestCase):
    def test_reason_describes_target_task(self):
        self.assertIn('번역', reason_for('Translation'))
        with self.assertRaises(KeyError):
            reason_for('Unknown')


@unittest.skipUnless((REV4 / 'stage4_cases.json').exists() and AUDITED_REQUESTS.exists(), 'revised artifacts not generated')
class RevisedRequests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = load_audited_requests(AUDITED_REQUESTS)
        cls.rev2 = read(REV2 / 'labeled_cases.json')
        cls.rev4 = {r['case_id']: r for r in read(REV4 / 'stage4_cases.json')}
        cls.corrections = read(MANUAL_CORRECTIONS) if MANUAL_CORRECTIONS.exists() else {}

    def test_changed_cases_reproduce_and_pass_checks(self):
        checked = 0
        for r2 in self.rev2:
            if checked >= 300:
                break
            cid = r2['case_id']
            if not r2['revision_trace']['changed_case_fields'] or cid in self.corrections:
                continue
            text, meta, edits, errors = revise_request(self.sources[r2['source_case_id']], r2['case'], cid)
            self.assertEqual(errors, [], cid)
            self.assertEqual(text, self.rev4[cid]['input_2'], cid)
            self.assertEqual(edits, self.rev4[cid]['transformation']['edits'], cid)
            self.assertFalse(inspect(text, r2['case'], meta))
            checked += 1
        self.assertGreater(checked, 0)

    def test_no_answer_leak_in_any_revised_request(self):
        for cid, r in list(self.rev4.items())[:1000]:
            self.assertFalse(inspect(r['input_2'], r['input_1']['case'], r['expression_metadata']), cid)


if __name__ == '__main__':
    unittest.main()
