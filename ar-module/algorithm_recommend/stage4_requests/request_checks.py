"""Fail-closed surface checks; semantic quality still needs source-aware audit."""
import re,unicodedata
from difflib import SequenceMatcher
from .request_design import NON_TEXT
TAGS=re.compile(r'^\s*<USER_TASK_DESCIRPTION>\s*(.*?)\s*</USER_TASK_DESCIRPTION>\s*<USER_REQUIREMENTS>\s*(.*?)\s*</USER_REQUIREMENTS>\s*$',re.S)
FORBIDDEN={
'memory':r'(?i)(?<![a-z])(?:VRAM|RAM)(?![a-z])|\d+\s*(?:Gi?B|GB|MB|TB)|\d\s*(?:기가|메가|테라)|메모리\s*(?:용량|크기)|비디오\s*메모리',
'answer_leak':r'(?i)\b(?:Qwen\w*|Gemma\w*|Llama\w*|DeepSeek\w*|Whisper\w*|SegFormer\w*|ConvNeXT\w*|DETR|RT-DETR|ViT|LoRA|QLoRA|SFT|AdamW|Adam|BERTScore|BLEU|ROUGE|chrF|LogicKor|mIoU|mAP|MOS|WER|CER)\b|학습률|배치\s*크기|에포크|하이퍼파라미터\s*(?:값|는|:)|양자화|어댑터|adapter|동결|freeze|(?<![A-Za-z])(?:LoRA|QLoRA|SFT|AdamW|Qwen[0-9.]*|Gemma[0-9.]*|Llama[0-9.]*|Whisper|BERTScore|BLEU|ROUGE|LogicKor)(?![A-Za-z])|큐웬|젬마|딥시크|라마\s*모델',
'new_resources':r'(?i)(?<![a-z])CPU(?![a-z])|오프로딩|offload|멀티\s*GPU|다중\s*GPU|GPU\s*(?:한|두|세|네|\d+)\s*(?:장|대|개)|(?:한|두|세|네|\d+)\s*(?:장|대)의?\s*(?:GPU|그래픽\s*카드)',
'new_constraints':r'실시간|지연\s*시간|초당|샘플링\s*레이트|샘플링\s*속도|\d+\s*(?:Hz|fps|초|분|시간|원|달러|퍼센트|%)|해상도|화자\s*(?:복제|구분|분리|식별)|참조\s*녹음|레이블.*준비|라벨.*준비|데이터.*(?:확보했|가지고 있|준비되어|준비돼)',
'unsupported_negatives':r'(?:필요\s*없|필요하지\s*않|요구하지\s*않|사용하지\s*않|하지\s*않고|기능\s*없이|검색.*없이|제약.*없|조건.*없|제한.*없|목표.*없|정해지지|정하지\s*않|정하기\s*어려|지정되지|명시되지|요구하지\s*않|별도.*없이)',
'explicit_thinking':r'(?i)\bthinking\b|\breasoning\b|생각\s*과정|추론\s*(?:강화|모델)|chain.of.thought',
'style_as_requirement':r'논리적으로.*말투|말투로.*읽|미사여구|문체|격식|구어체|존댓말|상담처럼|간결하게\s*(?:진행|답변|출력)|데이터를[^.!?]*가지고\s*있|가지고\s*있는\s*(?:이미지|사진|녹음|데이터)',
'invented_data_history':r'(?:제가|저희가)\s*가지고\s*있는\s*(?![^.!?]*(?:GPU|그래픽|장비|지포스|GeForce|RTX|rtx|엔비디아|카드)).{0,12}(?:텍스트|문서|글|자료|데이터|음성|이미지|녹음)|기존에는[^.!?]*(?:했더니|분류했)|이미\s+(?:수집|확보|준비)',
'false_specifics':r'예를\s*들어|의료|법률|금융|고객\s*상담|상품\s*리뷰|뉴스\s*기사|쇼핑몰|회의록|회의\s*녹음|CCTV|자율\s*주행|악기\s*소리|자동차\s*소리',
}
TASK_MARKERS={
'Classification':[(r'텍스트|글|문서|문장'),r'분류|범주|카테고리'],
'Question Answering':[r'질문|질의',r'답변|응답|대답'],
'Summarization':[r'요약|줄이|간추|짧게.*정리'],
'Translation':[r'번역'],
'Image Classification':[r'이미지|사진|그림',r'분류|범주|카테고리'],
'Detection':[r'이미지|사진|그림',r'검출|위치|어디'],
'Segmentation':[r'이미지|사진|그림',r'분할|픽셀|마스크|영역'],
'ASR':[r'음성|말소리|녹음|말한|발화',r'인식|전사|받아쓰|글자|문자|텍스트'],
'TTS':[r'음성|말소리|소리\s*내|읽어',r'텍스트|글|문장|문자'],
'Audio Classification':[r'오디오|소리|음향|녹음',r'분류|범주|종류|카테고리']}

def norm(t):return re.sub(r'[\s,\-·]','',unicodedata.normalize('NFKC',t)).lower()
def inspect(text,case,meta):
    errors=[];m=TAGS.fullmatch(text)
    if not m:return ['tag_format']
    desc,req=m.groups()
    if not 170<=len(desc)<=620:errors.append('description_length')
    if len(req)<15:errors.append('requirements_too_short')
    if not re.search(r'학습|훈련|파인튜닝|미세\s*조정',desc):errors.append('description_missing_training_request')
    if not re.search(r'모델',desc):errors.append('description_missing_model_request')
    if not re.search(r'추천|알려|조언|궁금|어떤|어떻게|제안|골라|고르|맞을|찾고|선택',text):errors.append('missing_recommendation_request')
    if norm(meta['gpu_alias']) not in norm(text):errors.append('gpu_alias_missing')
    if meta['language_name'] not in text:errors.append('language_missing')
    for name,pattern in FORBIDDEN.items():
        if re.search(pattern,text):errors.append(name)
    if any(not re.search(p,text) for p in TASK_MARKERS[case['task']]):errors.append('task_missing')
    if case['task']=='TTS' and re.search(r'일상\s*대화처럼|목소리\s*톤|감정을\s*담|논리적으로',text):errors.append('invented_tts_style')
    if case['task']=='TTS' and re.search(r'답변|응답|입출력|입력(?:과|이랑|부터|되는).{0,20}(?:출력|결과)|(?:출력|결과물).{0,25}(?:총량|토큰|합친|합산)',text):errors.append('tts_text_budget_includes_output')
    ctx=meta['context']
    if ctx['applicable']:
        if ctx['style']=='pages':
            if not re.search(rf'(?<!\d){ctx["pages"]}\s*(?:페이지|쪽|page)',text,re.I):errors.append('page_quantity_missing')
            if re.search(r'토큰|tokens?|\d\s*[Kk]\b',text):errors.append('page_token_leak')
        elif norm(ctx['surface']) not in norm(text):errors.append('context_quantity_missing')
        if case['task'] not in {'TTS'} and not (re.search(r'입력|지시문|질문(?:과|을|이|까)|질의(?:와|를|부터)',text) and re.search(r'결과|출력|답변|응답|전사|대답|입출력',text)):
            errors.append('total_budget_scope_missing')
        if not re.search(r'합친|합쳐|합쳤|합계|합산|모두|전부|통틀|입출력|총(?:량|체|괄|\s)|전체.{0,12}(?:처리|분량)|처리량',text) and not (case['task']=='TTS' and '지시문' in text and re.search(r'읽|입력\s*텍스트',text)):errors.append('total_budget_scope_missing')
    elif re.search(r'토큰|tokens?|\d+\s*(?:페이지|쪽|[Kk]\b)|(?:입력|출력).{0,20}(?:분량|크기)|분량\s*관리|총량|전체\s*분량|총\s*분량',text):errors.append('nontext_invented_context')
    match=SequenceMatcher(None,norm(desc),norm(req),autojunk=False).find_longest_match()
    if match.size>90:errors.append('within_request_repetition')
    return sorted(set(errors))
