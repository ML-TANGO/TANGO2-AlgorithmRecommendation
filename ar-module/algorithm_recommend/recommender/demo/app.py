"""TANGO2 local web demo using the trained recommender (model loaded in a separate worker process)."""
import argparse
import json
import os
import selectors
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..common import checkpoint_errors
from ... import __version__
from ...paths import DEFAULT_CHECKPOINT, DEMO_LOGS, MODULE_ROOT

HERE = Path(__file__).resolve().parent
STATIC = HERE / 'static'
TITLE = 'TANGO2 Requirement-Driven LLM 모델 : 알고리즘 추천 모듈'


class RequestFields(BaseModel):
    """공통 사용자 입력 블록 중 AR이 쓰는 필드. PO 모듈과 같은 필드명·검증 규칙을 사용한다.

    공통 입력의 나머지 필드(data_format, data_schema, data_label 등)는 받되 무시하므로
    같은 요청 본문을 PO와 AR에 그대로 보낼 수 있다. 모델에는 아래 두 필드만 전달된다.
    """

    model_config = ConfigDict(extra='ignore')

    task_description: str = Field(min_length=1, max_length=20000)
    requirements: str = Field(min_length=1, max_length=20000)

    @field_validator('task_description', 'requirements')
    @classmethod
    def validate_field(cls, value):
        value = value.strip()
        if not value:
            raise ValueError('내용을 입력해 주세요.')
        if '<USER_' in value.upper() or '</USER_' in value.upper():
            raise ValueError('태그 없이 내용만 입력해 주세요.')
        return value


class ModelWorker:
    def __init__(self, args):
        self.args = args; self.process = None; self.log = None; self.loaded = False

    def command(self):
        a = self.args
        cmd = [sys.executable, '-u', '-m', 'algorithm_recommend.recommender.demo.worker', '--checkpoint', str(Path(a.checkpoint).resolve()),
               '--device', a.device, '--dtype', a.dtype, '--attention', a.attention,
               '--max-new-tokens', str(a.max_new_tokens), '--temperature', str(a.temperature)]
        if a.local_files_only:
            cmd += ['--local-files-only']
        if a.load_in_4bit is not None:
            cmd += ['--load-in-4bit' if a.load_in_4bit else '--no-load-in-4bit']
        return cmd

    def run(self, fields, progress):
        if self.process is None or self.process.poll() is not None:
            self.close(); DEMO_LOGS.mkdir(parents=True, exist_ok=True)
            self.log = open(DEMO_LOGS / (datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8] + '.log'), 'ab', buffering=0)
            self.process = subprocess.Popen(self.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                cwd=MODULE_ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false'}, bufsize=0)
        self.process.stdin.write((json.dumps(fields, ensure_ascii=False) + '\n').encode()); self.process.stdin.flush()
        deadline = time.monotonic() + self.args.request_timeout; pending = b''
        with selectors.DefaultSelector() as sel:
            sel.register(self.process.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                if not sel.select(timeout=1):
                    continue
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(f'모델 프로세스가 종료되었습니다. 서버의 {DEMO_LOGS}를 확인해 주세요.')
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1); event = json.loads(line)
                    if event['event'] in {'loading', 'generating'}:
                        self.loaded = event['event'] == 'generating'; progress(event['event'])
                    elif event['event'] == 'result':
                        return event
                    elif event['event'] == 'error':
                        raise RuntimeError(event['message'])
        raise TimeoutError('추천 생성 대기 시간을 초과했습니다. 다시 시도해 주세요.')

    def close(self):
        process = self.process; self.process = None; self.loaded = False
        if process:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=5)
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
        if self.log:
            self.log.close(); self.log = None


class DemoService:
    def __init__(self, args, worker=None):
        self.args = args; self.worker = worker or ModelWorker(args); self.jobs = OrderedDict(); self.guard = threading.Lock(); self.busy = False

    def availability(self):
        problems = checkpoint_errors(self.args.checkpoint)
        return (True, None) if not problems else (False, '\n'.join(problems))

    def submit(self, fields):
        ready, reason = self.availability()
        if not ready:
            raise HTTPException(503, reason)
        with self.guard:
            if self.busy:
                raise HTTPException(409, '다른 추천을 생성 중입니다. 완료 후 다시 시도해 주세요.')
            self.busy = True; jid = uuid.uuid4().hex
            self.jobs[jid] = {'id': jid, 'state': 'queued', 'created_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat()}
            while len(self.jobs) > 50:
                self.jobs.popitem(last=False)

        def run():
            try:
                def progress(state):
                    with self.guard:
                        self.jobs[jid]['state'] = state
                response = self.worker.run(fields, progress)
                with self.guard:
                    self.jobs[jid].update(state='completed', result=response['result'], saved_filename=response['saved_filename'])
            except Exception as exc:
                self.worker.close()
                with self.guard:
                    self.jobs[jid].update(state='failed', message=str(exc))
            finally:
                with self.guard:
                    self.busy = False
        threading.Thread(target=run, name='tango-inference', daemon=True).start()
        return jid

    def job(self, jid):
        with self.guard:
            if jid not in self.jobs:
                raise HTTPException(404, '요청 기록을 찾을 수 없습니다.')
            return dict(self.jobs[jid])


def create_app(args, worker=None):
    service = DemoService(args, worker)

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.worker.close()
    app = FastAPI(title=TITLE, docs_url=None, redoc_url=None, lifespan=lifespan); app.state.service = service

    @app.get('/')
    def home():
        return FileResponse(STATIC / 'index.html')

    @app.get('/api/status')
    def status():
        ready, reason = service.availability()
        return dict(title=TITLE, version=__version__, version_label=f'추천기 {__version__}', checkpoint=str(Path(args.checkpoint).resolve()),
                    checkpoint_ready=ready, message=reason, busy=service.busy, model_loaded=service.worker.loaded)

    @app.post('/api/recommend', status_code=202)
    def recommend(fields: RequestFields):
        return {'job_id': service.submit(fields.model_dump())}

    @app.get('/api/jobs/{jid}')
    def job(jid: str):
        return service.job(jid)

    @app.get('/api/jobs/{jid}/download')
    def download(jid: str):
        record = service.job(jid)
        if record['state'] != 'completed':
            raise HTTPException(409, '아직 저장할 결과가 없습니다.')
        return JSONResponse(record['result'], headers={'Content-Disposition': f'attachment; filename="tango2_{jid}.json"'})
    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    return app


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT, help=f'final/ or checkpoint-N/ (default: {DEFAULT_CHECKPOINT}; env AR_CHECKPOINT)')
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8004)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--dtype', choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto')
    p.add_argument('--attention', choices=['sdpa', 'eager'], default='sdpa')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--load-in-4bit', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--max-new-tokens', type=int, default=4096); p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--request-timeout', type=int, default=1800)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.max_new_tokens <= 0 or args.temperature < 0 or args.request_timeout <= 0:
        p.error('포트, 토큰 상한 또는 대기 시간 설정을 확인하세요.')
    print(f'TANGO2 demo: http://{args.host}:{args.port} | checkpoint={Path(args.checkpoint).resolve()}', flush=True)
    print('모델은 첫 추천 요청 시 로딩됩니다. 사용 가능한 GPU를 CUDA_VISIBLE_DEVICES로 지정하세요.', flush=True)
    import uvicorn
    uvicorn.run(create_app(args), host=args.host, port=args.port, workers=1)


if __name__ == '__main__':
    main()
