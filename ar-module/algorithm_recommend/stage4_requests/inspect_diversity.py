"""Measure repetition after removing case-specific names/numbers; no human-authorship claim."""
import argparse
import json
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from .assemble import read_requests
from .request_checks import TAGS
from .request_design import LANGUAGES
from ..io_utils import write
from ..paths import STAGE4


def diversity_report(rows):
    """rows: iterable of {'case_id','text','metadata'} records."""
    import numpy as np
    rows = list(rows); texts = []; starts = Counter(); sentences = Counter(); lengths = []
    for r in rows:
        match = TAGS.fullmatch(r['text'])
        if not match:
            continue
        desc, req = match.groups(); lengths.append(len(desc)); starts[' '.join(desc.split()[:4])] += 1
        for sent in re.split(r'(?<=[.!?])\s+', desc + ' ' + req):
            if len(sent) >= 35:
                sentences[sent] += 1
        text = r['text'].replace(r['metadata']['gpu_alias'], 'GPU')
        text = re.sub(r'</?USER_[A-Z_]+>', ' ', text)
        for language in LANGUAGES.values():
            text = text.replace(language, 'LANG')
        text = re.sub(r'\d[\d,]*(?:\.\d+)?', 'N', text)
        text = re.sub(r'\s+', ' ', text).strip(); texts.append(text)
    if len(texts) != len(rows):
        raise ValueError('Invalid tags before diversity inspection')
    # 64-permutation MinHash, 16 LSH bands, followed by exact shingle Jaccard.
    # Candidate search is approximate; reported pair scores are exact.
    shingle_sets = [{t[i:i + 5] for i in range(max(1, len(t) - 4))} for t in texts]
    random = np.random.default_rng(20261024)
    aa = random.integers(1, 2 ** 32, size=64, dtype=np.uint64)
    bb = random.integers(0, 2 ** 32, size=64, dtype=np.uint64)
    buckets = defaultdict(list); candidates = set()
    for i, parts in enumerate(shingle_sets):
        hashes = np.array([zlib.crc32(x.encode()) for x in parts], dtype=np.uint64)
        sig = ((hashes[:, None] * aa + bb) % np.uint64(4294967311)).min(axis=0)
        for band in range(16):
            key = (band, tuple(int(x) for x in sig[4 * band:4 * band + 4]))
            for j in buckets[key]:
                candidates.add((j, i))
            buckets[key].append(i)
    pairs = {}
    for i, j in candidates:
        left, right = shingle_sets[i], shingle_sets[j]
        similarity = len(left & right) / len(left | right)
        if similarity >= .68:
            key = tuple(sorted((rows[i]['case_id'], rows[j]['case_id'])))
            pairs[key] = round(similarity, 4)
    return dict(total_requests=len(rows), exact_duplicate_texts=len(rows) - len({r['text'] for r in rows}),
        normalized_exact_duplicates=len(texts) - len(set(texts)),
        description_character_lengths=dict(min=min(lengths), max=max(lengths), mean=round(sum(lengths) / len(lengths), 1)),
        most_common_four_word_openings=starts.most_common(12),
        repeated_long_sentences=[dict(sentence=s, count=n) for s, n in sentences.most_common(30) if n > 1],
        near_duplicate_pairs=[dict(case_ids=list(k), shingle_jaccard=v) for k, v in sorted(pairs.items(), key=lambda x: -x[1])],
        method='Character 5-gram MinHash LSH candidate search (64 permutations, 16 bands) after removing GPU alias, language and numeric values; exact Jaccard >=0.68 for candidates',
        limitations='Lexical diversity measure, not proof of zero rhetorical patterns or human authorship')


def run(rows, output):
    report = diversity_report(rows)
    write(output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['near_duplicate_pairs', 'repeated_long_sentences']}, ensure_ascii=False, indent=2))
    print('near_duplicate_pairs', len(report['near_duplicate_pairs']), 'repeated_sentences', len(report['repeated_long_sentences']))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage4-dir', type=Path, default=STAGE4)
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    return run(read_requests(args.stage4_dir).values(), args.output or args.stage4_dir / 'diversity_report.json')


if __name__ == '__main__':
    main()
