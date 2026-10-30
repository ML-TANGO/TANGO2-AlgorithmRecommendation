"""Stage 2: reproducible, source-informed draft labels (no external LLM API calls)."""
import math

from ..io_utils import read, write  # noqa: F401  (re-exported for policy modules)
from ..paths import SOURCE_CONFIGS

VERSION = '7.0'
GEMMA_ARCH = 'https://github.com/google/gemma_pytorch/blob/main/gemma/config.py'
LLAMA_ARCH = 'https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py'
SOURCES = {
    'memory': 'https://huggingface.co/docs/transformers/v4.48.0/model_memory_anatomy',
    'checkpointing': 'https://huggingface.co/docs/transformers/grad_checkpointing',
    'chunked_loss': 'https://huggingface.co/docs/trl/reducing_memory_usage',
    'qlora': 'https://arxiv.org/abs/2305.14314',
    'lora': 'https://arxiv.org/abs/2106.09685',
    'logickor': 'https://github.com/instructkr/LogicKor',
    'errant': 'https://github.com/chrisjbryant/errant',
    'comet': 'https://github.com/Unbabel/COMET',
    'bertscore': 'https://github.com/Tiiiger/bert_score',
    'sacrebleu': 'https://github.com/mjpost/sacrebleu',
    'code': 'https://github.com/openai/human-eval',
    'sari': 'https://aclanthology.org/Q16-1029/',
}
VISION = {'Image Captioning', 'VQA', 'OCR', 'Document Understanding'}
AUDIO = {'ASR', 'Speech Translation', 'Speech Summarization', 'Speech Inpainting',
         'Speech Editing', 'Speech Enhancement', 'TTS', 'Voice Cloning',
         'Speaker Identification', 'Audio Captioning'}
SPATIAL = {'Grounding', 'Segmentation', 'Detection'}
IMAGE_OUTPUT = {'Image Editing', 'Image Generation'}
STRUCTURED = {'Information Extraction', 'Named Entity Recognition', 'Relation Extraction',
              'Classification', 'Sentiment Analysis', 'Topic Classification', 'Fact Verification'}
REASONING = {'Reasoning', 'Mathematical Reasoning', 'Logical Reasoning'}
CODE = {'Code Generation', 'Code Completion', 'Code Repair', 'Code Explanation', 'Code Translation'}
TOOLS = {'Agent', 'Tool Use', 'Function Calling'}
GROUNDED = {'Knowledge Grounded Generation', 'Retrieval Augmented Generation (RAG)', 'Closed-domain QA'}
TRANSLATION = {'Translation', 'Document-level Translation', 'Sentence-level Translation', 'Speech Translation'}
# Non-Qwen/Gemma profiles use conservative language subsets documented in model cards.
LLAMA_LANGS = 'en de fr it pt hi es th'.split()
PHI_LANGS = 'ar zh en fr de it ja ko pt ru es th tr'.split()
GRANITE_LANGS = 'en de es fr ja pt ar it ko zh'.split()
NEMO_LANGS = 'en fr de es it pt zh ja ko ar hi ru'.split()
QWEN25_LANGS = 'en zh fr es pt de it ru ja ko vi th ar'.split()


def source_config_path(model_id):
    return SOURCE_CONFIGS / (model_id.replace('/', '__') + '.json')


def make_profiles(models):
    # Architecture values for gated files come from the publishers' public code.
    fallback = {
        'meta-llama/Llama-3.2-1B-Instruct': (131072, 16, 2048, 128256),
        'meta-llama/Llama-3.2-3B-Instruct': (131072, 28, 3072, 128256),
        'meta-llama/Meta-Llama-3.1-8B-Instruct': (131072, 32, 4096, 128256),
        'google/gemma-3-1b-it': (32768, 26, 1152, 262144),
        'google/gemma-3-4b-it': (131072, 34, 2560, 262144),
        'google/gemma-3-12b-it': (131072, 48, 3840, 262144),
        'google/gemma-3-27b-it': (131072, 62, 5376, 262144),
    }
    profiles = []
    for model in models:
        mid = model['model_id']
        path = source_config_path(mid)
        config_source = None
        revision = None
        if path.is_file():
            saved = read(path)
            cfg = saved['config'].get('text_config', saved['config'])
            context = cfg['max_position_embeddings']
            layers = cfg.get('num_hidden_layers', cfg.get('num_layers'))
            hidden, vocab = cfg['hidden_size'], cfg['vocab_size']
            config_source, revision = saved['url'], saved['revision']
        else:
            context, layers, hidden, vocab = fallback[mid]
            config_source = GEMMA_ARCH if mid.startswith('google/') else LLAMA_ARCH
        family = model['family']
        if family == 'Qwen3':
            # Native trained context in the card is 32768 despite config maximum 40960.
            # YaRN extensions are not assumed by this policy.
            context = 32768
        if family in {'Qwen3', 'Gemma 3'}:
            languages = None
            language_basis = 'broad_multilingual_model_card_claim; per-language quality not verified'
        else:
            languages = {
                'Qwen2.5': QWEN25_LANGS, 'Llama 3.2': LLAMA_LANGS,
                'Llama 3.1': LLAMA_LANGS, 'Phi-4': PHI_LANGS,
                'Granite 3.3': GRANITE_LANGS, 'Mistral NeMo': NEMO_LANGS,
                'EXAONE 3.5': ['en', 'ko'], 'DeepSeek-R1-Distill': ['en', 'zh'],
                'Mistral': ['en'],
            }[family]
            language_basis = 'conservative documented/evaluated language subset; not a claim other languages are impossible'
        visual = family == 'Gemma 3' and model['params_b_approx'] >= 4
        profiles.append({
            **model, 'max_context_tokens_used': context,
            'layers': layers, 'hidden_size': hidden, 'vocab_size': vocab,
            'image_input': visual, 'output_modality': 'text',
            'languages_used': languages, 'language_basis': language_basis,
            'native_thinking': family in {'Qwen3', 'DeepSeek-R1-Distill', 'Granite 3.3'},
            'model_card_url': 'https://huggingface.co/' + mid,
            'architecture_source_url': config_source, 'config_revision': revision,
        })
    return profiles


# Each default is an explicit task-specific decision, not a random metric selection.
# The requirements below are evaluation prerequisites, not facts supplied by a case.
METRIC_POLICY = {
    'Classification': ('F1 Score', '클래스별 성능을 균등하게 평가합니다.', '정답 클래스, macro averaging, label 집합과 zero-division 규칙 필요'),
    'Question Answering': ('F1 Score', '참조 답변과 답변 토큰의 중첩을 평가합니다.', '짧은 정답 QA를 기본으로 가정; 언어별 정규화·토큰화와 복수 정답 처리 규칙 필요'),
    'Summarization': ('BERTScore', '참조 요약과 의미 유사성을 평가합니다.', '참조 요약, 해당 언어 지원 encoder와 버전, 장문 분할 규칙 필요; 사실성을 보장하지 않음'),
    'Translation': ('chrF', '문자 n-gram 중첩으로 참조 번역과 비교합니다.', '출발·도착 언어, 참조 번역, beta·n-gram·공백 규칙 필요; 언어별 neural evaluator 지원을 임의 가정하지 않음'),
    'Image Classification': ('F1 Score', '이미지 클래스별 성능을 균등하게 평가합니다.', '단일 라벨 이미지 분류, 정답 클래스, macro F1 규칙 필요'),
    'Detection': ('mAP', '객체 box와 class 검출 성능을 평가합니다.', '정답 box·class, COCO AP@[.50:.05:.95], one-to-one matching·ignore 규칙 필요'),
    'Segmentation': ('mIoU', '의미론적 분할의 클래스별 영역 중첩을 평가합니다.', '정답 semantic mask, 클래스 집합, background·ignore-index·빈 클래스 처리 필요'),
    'ASR': ('WER', '음성 전사의 단어 단위 오류를 평가합니다.', '참조 전사, 언어별 단어 분할·정규화·빈 참조 규칙 필요'),
    'TTS': ('MOS', '합성 음성의 자연스러움을 청취자가 평가합니다.', '대상 언어 청취자, blind 평가, 척도·평가 문항·표본 수·불확실성 보고 필요'),
    'Audio Classification': ('mAP', '소리 이벤트별 clip-level 분류 점수 순위를 평가합니다.', '기본은 multi-label sound-event tagging; 클래스별 점수와 정답 태그 필요, IoU·시간 구간 검출 지표가 아님'),
}


def pick_metric(case, metrics):
    task, lang = case['task'], case['language']
    name, why, requirements = METRIC_POLICY[task]
    refs = ['dataset/metric_list.json']
    if task == 'ASR' and lang in {'zh', 'ja', 'th'}:
        name, why = 'CER', '단어 분할 규칙에 덜 의존하도록 문자 단위 전사 오류를 평가합니다.'
        requirements = '참조 전사, Unicode·문자 분할·공백 정규화 규칙 필요'
    if task == 'Audio Classification' and case.get('audio_subtask') == 'keyword_spotting':
        name, why = 'Accuracy', '단일 라벨 키워드 분류의 정답 비율을 평가합니다.'
        requirements = '기본 영어 keyword set의 정답 클래스와 표준 테스트 분할 필요'
    if name == 'BERTScore':
        refs.append(SOURCES['bertscore'])
    if name == 'chrF':
        refs.append(SOURCES['sacrebleu'])
    assert name in {m['name'] for m in metrics[task].values()}, (task, name)
    return name, why, requirements, refs


def model_score(profile, case):
    """Explicit ranking heuristic; no hidden reasoning value or measured quality score."""
    target = 3.0 if case['task'] == 'Classification' else 8.0 if case['gpu']['vram_gb'] >= 24 else 4.0
    ratio = profile['params_b_approx'] / target
    score = 6 * math.log2(min(ratio, 1)) - 2 * max(math.log2(ratio), 0)
    score += {'Qwen3': 3, 'Gemma 3': 2}.get(profile['family'], 0)
    return score, [f'정책 목표 규모 {target:g}B와의 거리 및 다국어 범용 후보 우선순위; 실측 성능 아님']
