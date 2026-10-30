"""Shared JSON, hashing, validation and backup helpers used by every stage."""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def canon(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def now_tag():
    return datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')


def backup_existing(output_dir, names, label):
    """Copy existing artifacts into output_dir/backup/<label>_<time>/ before regeneration."""
    output_dir = Path(output_dir)
    existing = [output_dir / n for n in names if (output_dir / n).exists()]
    if not existing:
        return None
    folder = output_dir / 'backup' / f'{label}_{now_tag()}'
    folder.mkdir(parents=True)
    for path in existing:
        if path.is_dir():
            shutil.copytree(path, folder / path.name)
        else:
            shutil.copy2(path, folder / path.name)
    write(folder / 'archive_manifest.json', {p.name: sha(p) for p in existing if p.is_file()})
    return folder


def manifest(paths):
    """{portable path: sha256} for provenance records."""
    from .paths import rel
    return {rel(p): sha(p) for p in paths}
