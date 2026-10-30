"""Audited original Stage 4 requests: from a full Stage 4 artifact folder or the bundled compact resource."""
from pathlib import Path

from ..io_utils import read
from ..paths import STAGE4, AUDITED_REQUESTS


def audited_requests_path(stage4=STAGE4):
    """Prefer a locally assembled Stage 4; fall back to the bundled audited requests."""
    full = Path(stage4) / 'stage4_cases.json'
    if full.exists():
        return full
    if AUDITED_REQUESTS.exists():
        return AUDITED_REQUESTS
    raise FileNotFoundError('No audited Stage 4 requests: run stage4 assembly or restore resources/audited_requests_v4.json')


def load_audited_requests(path=None):
    """{case_id: {case, input_2, expression_metadata, request_audit, semantic_group, label_2_status, label_2_reason, generation_provenance}}"""
    path = Path(path or audited_requests_path())
    data = read(path)
    if isinstance(data, dict) and 'records' in data:
        records = data['records']
    else:
        records = [dict(case_id=r['case_id'], case=r['input_1']['case'], input_2=r['input_2'], expression_metadata=r['expression_metadata'],
                        request_audit=r['request_audit'], semantic_group=r['semantic_group'],
                        label_2_status=r['label_2']['status'], label_2_reason=r['label_2']['reason'],
                        generation_provenance=r['generation_provenance']) for r in data]
    return {r['case_id']: r for r in records}
