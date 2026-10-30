"""Module-relative workspace layout. Every stage reads/writes through these paths.

Override with environment variables (see .env.example):
  AR_ARTIFACTS_DIR  generated stage artifacts (default: <module>/artifacts)
  AR_DATASET_DIR    candidate catalogs           (default: <module>/resources/dataset)
  AR_DATA_DIR       prepared SFT data            (default: <module>/data)
  AR_CHECKPOINT     trained recommender checkpoint (default: <module>/checkpoints/gemma4-lora-final)
"""
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
MODULE_ROOT = PACKAGE_DIR.parent
RESOURCES = MODULE_ROOT / 'resources'


def _env_path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else Path(default)


DATASET_DIR = _env_path('AR_DATASET_DIR', RESOURCES / 'dataset')
SOURCE_CONFIGS = RESOURCES / 'source_configs'
SOURCE_EVIDENCE = RESOURCES / 'source_evidence'
TTS_FEASIBLE_SPACE = RESOURCES / 'tts_feasible_space.json'
MANUAL_CORRECTIONS = RESOURCES / 'manual_corrections.json'
AUDITED_REQUESTS = RESOURCES / 'audited_requests_v4.json'

ARTIFACTS = _env_path('AR_ARTIFACTS_DIR', MODULE_ROOT / 'artifacts')
STAGE1 = ARTIFACTS / 'stage_1'
STAGE2 = ARTIFACTS / 'stage_2'
STAGE3 = ARTIFACTS / 'stage_3'
STAGE4 = ARTIFACTS / 'stage_4'
REV2 = ARTIFACTS / 'revised' / 'stage_2'
REV3 = ARTIFACTS / 'revised' / 'stage_3'
REV4 = ARTIFACTS / 'revised' / 'stage_4'

DATA_DIR = _env_path('AR_DATA_DIR', MODULE_ROOT / 'data')
OUTPUTS_DIR = ARTIFACTS / 'outputs'
INFERENCE_RESULTS = ARTIFACTS / 'inference_results'
DEMO_LOGS = ARTIFACTS / 'demo_logs'
DEFAULT_CHECKPOINT = _env_path('AR_CHECKPOINT', MODULE_ROOT / 'checkpoints' / 'gemma4-lora-final')
DEFAULT_BASE_MODEL = os.environ.get('AR_BASE_MODEL', 'google/gemma-4-E4B-it')
DEFAULT_BASE_REVISION = os.environ.get('AR_BASE_REVISION', 'main')


def rel(path):
    """Portable manifest key: relative to the module root when possible."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(MODULE_ROOT))
    except ValueError:
        return str(path)


def resolve(key):
    """Inverse of rel()."""
    path = Path(key)
    return path if path.is_absolute() else MODULE_ROOT / path


def package_files(*names):
    """Implementation files hashed into provenance manifests."""
    return [PACKAGE_DIR / n for n in names]
