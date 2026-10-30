"""Stage 5 contract, loss and CLI tests (torch tests skip when the ML stack is unavailable)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from algorithm_recommend.paths import DATA_DIR, DEFAULT_CHECKPOINT
from algorithm_recommend.recommender.common import build_input, split_input, read_jsonl, target_errors, strict_json, checkpoint_errors, SCHEMA_VERSION
from algorithm_recommend.recommender.inference import parser, collect_fields

try:
    import torch
    from transformers import Gemma4TextConfig, Gemma4ForCausalLM
    from algorithm_recommend.recommender.modeling import CompletionCollator, configure_adapters
    from algorithm_recommend.recommender.sft_loss import answer_loss
    ML = True
except Exception:  # pragma: no cover - environment dependent
    ML = False


class ContractTests(unittest.TestCase):
    def test_separate_input_roundtrip(self):
        task = '한국어 문서를 요약할 모델과 학습 방법을 추천해 주세요.'; req = '5090 그래픽 카드 환경입니다.\n보고서 5쪽 정도입니다.'
        self.assertEqual(split_input(build_input(task, req)), (task, req))
        self.assertIn('<USER_TASK_DESCIRPTION>', build_input(task, req))

    def test_missing_or_injected_fields_rejected(self):
        for task, req in [('', '요구사항'), ('설명', None), ('<USER_REQUIREMENTS>x</USER_REQUIREMENTS>', '요구')]:
            with self.assertRaises(ValueError):
                build_input(task, req)

    def test_two_cli_fields_and_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'task.txt'; p.write_text('요약 모델을 추천해 주세요.', encoding='utf-8')
            args = parser().parse_args(['--task-description-file', str(p), '--requirements', '한국어, RTX5090'])
            self.assertEqual(collect_fields(args), ('요약 모델을 추천해 주세요.', '한국어, RTX5090'))

    @unittest.skipUnless((DATA_DIR / 'train.jsonl').exists(), 'prepared data missing')
    def test_actual_prepared_target_and_false_empirical_claim(self):
        row = read_jsonl(DATA_DIR / 'train.jsonl')[0]; value = json.loads(row['messages'][2]['content'])
        self.assertFalse(target_errors(value)); value['ready_to_execute'] = True
        self.assertTrue(target_errors(value))

    def test_strict_json(self):
        for s in ['{"a":NaN}', '{"a":1,"a":2}', '```json\n{}\n```']:
            with self.assertRaises(ValueError):
                strict_json(s)

    @unittest.skipUnless((DEFAULT_CHECKPOINT / 'run_config.json').exists(), 'bundled checkpoint missing')
    def test_bundled_checkpoint_matches_contract(self):
        self.assertEqual(checkpoint_errors(DEFAULT_CHECKPOINT), [])
        self.assertEqual(json.loads((DEFAULT_CHECKPOINT / 'run_config.json').read_text())['schema_version'], SCHEMA_VERSION)

    def test_missing_checkpoint_is_reported_not_substituted(self):
        self.assertTrue(checkpoint_errors(Path('/nonexistent/checkpoint')))


@unittest.skipUnless(ML, 'torch/transformers with Gemma 4 support unavailable')
class LossTests(unittest.TestCase):
    @staticmethod
    def config():
        return Gemma4TextConfig(vocab_size=64, vocab_size_per_layer_input=64, hidden_size=32, hidden_size_per_layer_input=8, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16, layer_types=['sliding_attention', 'full_attention'], max_position_embeddings=128, sliding_window=32, attention_dropout=0.0)

    def test_padding_does_not_mask_real_eos(self):
        f = [dict(input_ids=[2, 3, 1], attention_mask=[1, 1, 1], labels=[-100, 3, 1]), dict(input_ids=[2, 1], attention_mask=[1, 1], labels=[-100, 1])]
        b = CompletionCollator(1)(f); self.assertEqual(b['labels'].tolist(), [[-100, 3, 1], [-100, 1, -100]])

    def test_chunked_loss_and_adapter_gradients_match_native(self):
        torch.set_num_threads(2); torch.manual_seed(5)
        a = configure_adapters(Gemma4ForCausalLM(self.config()), 'lora', rank=4, alpha=8, dropout=0)
        b = copy.deepcopy(a); a.train(); b.train()
        ids = torch.randint(3, 64, (2, 12)); mask = torch.ones_like(ids); mask[1, -2:] = 0; ids[1, -2:] = 0
        labels = ids.clone(); labels[:, :5] = -100; labels[1, -2:] = -100
        inputs = dict(input_ids=ids, attention_mask=mask, labels=labels)
        native = a(**inputs, use_cache=False).loss; native.backward()
        chunked = answer_loss(b, inputs, chunk_size=3); chunked.backward()
        torch.testing.assert_close(chunked, native, rtol=1e-5, atol=1e-5)
        gradients = 0
        for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
            self.assertEqual(na, nb)
            if pa.requires_grad:
                self.assertIsNotNone(pb.grad); torch.testing.assert_close(pa.grad, pb.grad, atol=1e-5, rtol=2e-4); gradients += 1
            else:
                self.assertIsNone(pb.grad)
        self.assertGreater(gradients, 0)

    def test_accumulated_token_denominator(self):
        torch.manual_seed(9); m = Gemma4ForCausalLM(self.config()); m.eval()
        x = torch.randint(3, 64, (1, 8)); y = x.clone(); y[:, :3] = -100; inputs = dict(input_ids=x, labels=y)
        normal = answer_loss(m, inputs, 3); scaled = answer_loss(m, inputs, 3, num_items_in_batch=10)
        torch.testing.assert_close(normal / 2, scaled)


if __name__ == '__main__':
    unittest.main()
