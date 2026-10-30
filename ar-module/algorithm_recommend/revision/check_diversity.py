"""Run the lexical diversity check on revised inputs only."""
import argparse
from pathlib import Path

from ..stage4_requests.inspect_diversity import run
from ..io_utils import read
from ..paths import REV4


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rev4-dir', type=Path, default=REV4); p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    rows = read(args.rev4_dir / 'stage4_cases.json')
    return run([dict(case_id=r['case_id'], text=r['input_2'], metadata=r['expression_metadata']) for r in rows], args.output or args.rev4_dir / 'diversity_report.json')


if __name__ == '__main__':
    main()
