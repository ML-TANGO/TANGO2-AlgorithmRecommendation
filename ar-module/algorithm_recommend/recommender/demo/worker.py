"""Persistent model worker: loads the recommender once, serves JSON-lines requests on stdin/stdout."""
import contextlib
import json
import sys
import traceback

from ..inference import Recommender, parser as inference_parser, checkpoint_errors
from ..inference_output import save_prediction


def main(argv=None):
    args = inference_parser().parse_args(argv)
    channel = sys.stdout

    def emit(value):
        channel.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n'); channel.flush()
    recommender = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if recommender is None:
                    emit({'event': 'loading'})
                    problems = checkpoint_errors(args.checkpoint)
                    if problems:
                        raise ValueError('\n'.join(problems))
                    recommender = Recommender(args.checkpoint, device=args.device, dtype=args.dtype, model=args.model,
                        local_files_only=args.local_files_only, attention=args.attention, load_in_4bit=args.load_in_4bit,
                        show_loading_report=args.show_loading_report)
                emit({'event': 'generating'})
                result = recommender.predict(request['task_description'], request['requirements'], max_new_tokens=args.max_new_tokens, temperature=args.temperature)
                saved = save_prediction(result, None)
            emit({'event': 'result', 'result': result, 'saved_filename': saved.name})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit({'event': 'error', 'message': str(exc)})
            # Release partial model / CUDA allocations and start clean on retry.
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
