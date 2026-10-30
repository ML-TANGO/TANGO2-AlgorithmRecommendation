"""Resolve case length against the single candidate source (context_length_list.json)."""
import json
import re
from pathlib import Path

from ..paths import DATASET_DIR

SOURCE = DATASET_DIR / 'context_length_list.json'


def load_context_lengths(path=SOURCE):
    values = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(values, list) or not values:
        raise ValueError('Empty context length list')
    parsed = []
    for value in values:
        if type(value) is int:
            n = value
        elif isinstance(value, str):
            match = re.fullmatch(r'\s*([0-9][0-9,]*)(?:\s*\([^()]+\))?\s*', value)
            if not match:
                raise ValueError(f'Invalid context length: {value!r}')
            n = int(match[1].replace(',', ''))
        else:
            raise ValueError('Context candidates must be integers or numeric labels')
        if n <= 0 or n in parsed:
            raise ValueError('Nonpositive/duplicate context candidate')
        parsed.append(n)
    return sorted(parsed)


def derive_sequence_length(case, profile, candidates=None):
    candidates = load_context_lengths() if candidates is None else candidates
    n = case['context_length']; applies = profile['context_applicable']
    if type(n) is not int or n not in candidates:
        raise ValueError('case.context_length is outside context_length_list.json')
    if applies:
        if n + profile['extra_tokens'] > profile['native_context']:
            raise ValueError('Requested context exceeds the model native limit')
        if profile.get('max_text_context') is not None and n > profile['max_text_context']:
            raise ValueError('Requested context exceeds the reviewed text limit')
    return dict(candidate_source='dataset/context_length_list.json', value_source='case.context_length',
                applicable=applies, max_seq_length=n if applies else None,
                semantics='Text input plus supervised text output budget; non-text tokens must be handled separately. Not the recommendation LLM training sequence limit.' if applies else 'No text sequence limit for this task; omit the trainer option.')
