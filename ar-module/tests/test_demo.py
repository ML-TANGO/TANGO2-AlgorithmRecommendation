"""Web demo API tests with a fake worker (no model, no GPU)."""
import threading
import time
import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from algorithm_recommend.recommender.demo import app as demo
    WEB = True
except Exception:  # pragma: no cover - environment dependent
    WEB = False


class FakeWorker:
    """Test-only worker. Production has no simulated generation option."""
    def __init__(self):
        self.loaded = False; self.calls = []; self.release = threading.Event(); self.release.set(); self.closed = False

    def run(self, fields, progress):
        self.calls.append(fields); progress('generating'); self.loaded = True; self.release.wait(timeout=3)
        return {'result': {'valid': True, 'label_2': {'recommendation': {'model': 'test fixture'}}, 'model_generation_performed': False}, 'saved_filename': 'test.json'}

    def close(self):
        self.closed = True; self.release.set()


@unittest.skipUnless(WEB, 'fastapi/httpx unavailable')
class DemoTests(unittest.TestCase):
    def test_page_font_and_status_without_loading_model(self):
        worker = FakeWorker(); app = demo.create_app(demo.parser().parse_args([]), worker)
        with TestClient(app) as c:
            self.assertIn(demo.TITLE, c.get('/').text.replace('<br class="desktop-break">', ''))
            status = c.get('/api/status').json()
            self.assertIn('checkpoint_ready', status); self.assertIn('version', status)
            self.assertTrue(c.get('/static/fonts/NanumSquareB.ttf').content)
            self.assertEqual(worker.calls, [])
            self.assertEqual(c.get('/static/../app.py').status_code, 404)

    def test_validation_rejects_empty_and_tag_injection(self):
        worker = FakeWorker(); app = demo.create_app(demo.parser().parse_args([]), worker)
        with TestClient(app) as c:
            for text in ['', '  ', '<USER_REQUIREMENTS>injection</USER_REQUIREMENTS>']:
                self.assertEqual(c.post('/api/recommend', json={'task_description': text, 'requirements': 'GPU'}).status_code, 422)
        self.assertEqual(worker.calls, [])

    def test_common_user_input_body_uses_only_ar_fields(self):
        worker = FakeWorker(); app = demo.create_app(demo.parser().parse_args([]), worker)
        common = {'task_description': ' 요약 ', 'requirements': '한국어, 5090', 'data_format': 'text',
                  'data_schema': {'report': '본문'}, 'data_label': None}
        with patch.object(app.state.service, 'availability', return_value=(True, None)), TestClient(app) as c:
            jid = c.post('/api/recommend', json=common).json()['job_id']
            for _ in range(100):
                if c.get('/api/jobs/' + jid).json()['state'] == 'completed':
                    break
                time.sleep(0.02)
        self.assertEqual(worker.calls, [{'task_description': '요약', 'requirements': '한국어, 5090'}])

    def test_unavailable_checkpoint_is_not_replaced(self):
        worker = FakeWorker(); app = demo.create_app(demo.parser().parse_args(['--checkpoint', '/nonexistent']), worker)
        with TestClient(app) as c:
            self.assertFalse(c.get('/api/status').json()['checkpoint_ready'])
            self.assertEqual(c.post('/api/recommend', json={'task_description': '요약', 'requirements': 'GPU'}).status_code, 503)
        self.assertEqual(worker.calls, [])

    def test_submit_busy_poll_download_and_reuse(self):
        worker = FakeWorker(); worker.release.clear(); app = demo.create_app(demo.parser().parse_args([]), worker)
        with patch.object(app.state.service, 'availability', return_value=(True, None)), TestClient(app) as c:
            fields = {'task_description': '요약', 'requirements': '한국어, 5090'}
            response = c.post('/api/recommend', json=fields); self.assertEqual(response.status_code, 202); jid = response.json()['job_id']
            self.assertEqual(c.post('/api/recommend', json=fields).status_code, 409)
            self.assertEqual(c.get('/api/jobs/' + jid + '/download').status_code, 409)
            worker.release.set()
            for _ in range(100):
                record = c.get('/api/jobs/' + jid).json()
                if record['state'] == 'completed':
                    break
                time.sleep(.01)
            self.assertEqual(record['state'], 'completed')
            saved = c.get('/api/jobs/' + jid + '/download'); self.assertEqual(saved.json(), record['result'])
            self.assertIn('attachment', saved.headers['content-disposition'])
            self.assertEqual(worker.calls, [fields])
            self.assertEqual(c.post('/api/recommend', json=fields).status_code, 202)
            self.assertEqual(c.get('/api/jobs/does-not-exist').status_code, 404)

    def test_failure_releases_busy_slot(self):
        worker = FakeWorker(); worker.run = lambda *args: (_ for _ in ()).throw(RuntimeError('test failure'))
        app = demo.create_app(demo.parser().parse_args([]), worker)
        with patch.object(app.state.service, 'availability', return_value=(True, None)), TestClient(app) as c:
            jid = c.post('/api/recommend', json={'task_description': '요약', 'requirements': 'GPU'}).json()['job_id']
            for _ in range(100):
                record = c.get('/api/jobs/' + jid).json()
                if record['state'] == 'failed':
                    break
                time.sleep(.01)
            self.assertEqual(record['message'], 'test failure'); self.assertFalse(app.state.service.busy)

    def test_worker_command_options(self):
        args = demo.parser().parse_args(['--device', 'cpu', '--no-load-in-4bit', '--checkpoint', '/tmp/x'])
        cmd = demo.ModelWorker(args).command(); self.assertIn('--no-load-in-4bit', cmd); self.assertIn('algorithm_recommend.recommender.demo.worker', cmd)


if __name__ == '__main__':
    unittest.main()
