"""Scoped loading-log filtering and automatic inference JSON persistence."""
from contextlib import contextmanager
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import threading
from uuid import uuid4
from zoneinfo import ZoneInfo

MODALITY_PREFIXES = ('model.vision_tower.','model.audio_tower.','model.embed_vision.','model.embed_audio.')
from ..paths import INFERENCE_RESULTS as DEFAULT_RESULT_DIR
ANSI = re.compile(r'\x1b\[[0-9;]*m')


class ExpectedModalityReportFilter(logging.Filter):
    """Hide a report ONLY when every entry is an expected unused modality key.

    Actual missing/mismatched/conversion or unrelated unexpected keys remain
    visible. Does not change any loading information or logger verbosity.
    """
    def __init__(self):
        super().__init__()
        self.owner_thread = threading.get_ident()

    def filter(self, record):
        if record.thread != self.owner_thread:
            return True
        text = ANSI.sub('',record.getMessage())
        if not text.startswith('Gemma4ForCausalLM LOAD REPORT from:'):
            return True
        rows = []
        for line in text.split('\n\nNotes:',1)[0].splitlines()[1:]:
            if '|' not in line:
                continue
            columns = [column.strip() for column in line.split('|')]
            if columns[:2] == ['Key','Status']:
                continue
            if len(columns)<2 or columns[1]!='UNEXPECTED' or not columns[0].startswith(MODALITY_PREFIXES):
                return True
            rows.append(columns[0])
        return not rows


@contextmanager
def quiet_expected_modality_report(enabled):
    if not enabled:
        yield
        return
    filter_ = ExpectedModalityReportFilter()
    loggers = [logging.getLogger(name) for name in ['transformers.modeling_utils','transformers.utils.loading_report']]
    for logger in loggers:
        logger.addFilter(filter_)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(filter_)


def save_prediction(result, output=None):
    """Save the same JSON object printed by the CLI; default names never overwrite."""
    text = json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if output is None:
        stamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')
        path = DEFAULT_RESULT_DIR/f'prediction_{stamp}_{uuid4().hex[:8]}.json'
        mode = 'x'
    else:
        path = Path(output).expanduser().resolve()
        mode = 'w'  # --output retains the existing explicit-path behavior.
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open(mode,encoding='utf-8') as handle:
        handle.write(text)
    return path
