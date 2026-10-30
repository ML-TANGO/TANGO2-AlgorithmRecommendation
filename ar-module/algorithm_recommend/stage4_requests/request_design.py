"""Case-only generation briefs: labels and hardware capacity never enter prompts."""
import hashlib,json,random,re
from .expression_utils import gpu_aliases,gpu_identity,korean_number
LANGUAGES={'en':'영어','ko':'한국어','zh':'중국어','es':'스페인어','hi':'힌디어','ar':'아랍어','fr':'프랑스어','pt':'포르투갈어','bn':'벵골어','ru':'러시아어','ur':'우르두어','id':'인도네시아어','de':'독일어','ja':'일본어','tr':'튀르키예어','vi':'베트남어','it':'이탈리아어','fa':'페르시아어','th':'태국어','pa':'펀자브어'}
NON_TEXT={'Image Classification','Detection','Segmentation','Audio Classification'}
TASK_MEANING={
'Classification':'텍스트 내용을 범주별로 분류',
'Question Answering':'텍스트 질문에 적합한 답변을 작성하는 질의응답',
'Summarization':'원문의 중요한 내용을 추려 텍스트를 요약',
'Translation':'텍스트 번역',
'Image Classification':'이미지 전체의 내용을 보고 범주를 분류',
'Detection':'이미지 속 객체의 종류와 위치를 검출',
'Segmentation':'이미지의 픽셀/영역을 구분하는 분할',
'ASR':'음성을 텍스트로 받아쓰는 음성 인식·전사',
'TTS':'글을 말소리로 읽어 주는 텍스트 음성 합성',
'Audio Classification':'녹음에 담긴 소리의 종류를 분류'}

VOICES=['조심스럽게 상담하는 말투','일상 대화처럼 편하게 묻는 말투','차분한 존댓말','업무 메신저에 남기는 자연스러운 존댓말','기술 상담 게시판의 질문 글','경험이 많지 않은 사람이 목적부터 풀어 쓰는 말투','직접 부탁하는 간결하고 담백한 말투','고민하는 지점을 먼저 이야기하는 말투','작업 흐름을 설명하다가 추천을 부탁하는 말투','조금 더 격식 있는 문의 문장','혼잣말처럼 생각하다가 질문으로 이어지는 말투','상대에게 조건을 공유하며 함께 검토하려는 말투','간결한 구어체','담담한 일상 문장','목적과 기대하는 도움을 분리해 설명하는 문체','질문을 먼저 던지고 배경을 덧붙이는 문체']
ORDERS=['작업 목적 → 고민 → 추천 요청','질문 → 작업 방식 설명 → 조건 연결','현재 구상 → 도움이 필요한 점 → 질문','학습 방법에 대한 문의 → 작업 정의 → 마무리','원하는 결과 → 구현 구상 → 적합한 모델과 학습법 문의','두세 개의 연결된 질문으로 생각을 풀기','모델 선택을 고민하는 이유 → 작업 설명 → 요청','작업을 일상적인 말로 묘사한 다음 기술적 명칭 덧붙이기']
REQ_STYLES=['짧은 한 문단','앞 문장을 이어 받는 두 문장','조건 순서를 바꾼 메모','가벼운 줄바꿈을 넣은 서술','단순 목록과 한 문장의 혼합','질문 형태가 섞인 자연스러운 문장','한 문장 안에 조건들을 연결','두 개의 짧은 문단']
SYSTEM='''당신은 모델 추천 서비스를 이용하는 실제 사용자의 질문 글을 대신 작성합니다. 추천 답변이 아니라 사용자 입력만 작성하세요.
반드시 아래 XML 태그 두 쌍만 출력합니다. 철자를 그대로 지키고 설명/코드블록/JSON은 쓰지 마세요.
<USER_TASK_DESCIRPTION>사용자 요청</USER_TASK_DESCIRPTION>
<USER_REQUIREMENTS>조건에 대한 추가 설명</USER_REQUIREMENTS>
첫 항목은 한국어 4~6문장, 200~360자 정도의 충분한 서술입니다. 작업을 무엇 때문에 어떻게 하고 싶은지 풀어 쓰고 이 작업에 적합한 모델 추천 및 학습 방법을 직접 요청하세요. 같은 결론이나 문장 틀을 계속 쓰지 말고 도입·어순·문장 길이·질문 위치를 자유롭게 바꾸세요. 두 번째 항목은 반복 설명으로 채우지 말고 제공된 조건을 자연스럽게 전달하세요. GPU/언어/분량은 어느 섹션에 놓아도 좋습니다. 태그 순서만 유지하세요.
입력 사실을 바꾸거나 새 실행 조건을 만들지 마세요. 데이터가 이미 있다거나 라벨이 준비됐다거나 특정 분야라는 주장을 만들지 마세요. GPU 수, CPU offload, VRAM/메모리 용량, 학습 시간·비용·실시간·성능 목표·데이터 수·오디오 길이·해상도·화자·번역 방향·출력 비율은 지정되지 않았으므로 추가 금지입니다. 추가 상황은 '모델 선택을 고민한다/작업을 구성하려 한다' 수준의 비제약적인 배경만 허용합니다.
정답에 해당하는 구체적 모델명, LoRA/QLoRA/SFT 같은 알고리즘명, 평가 metric명, 하이퍼파라미터 값·명칭, adapter·동결·양자화 지시는 절대 쓰지 마세요. '모델과 학습 방법을 추천해 달라'는 일반적인 요청은 허용합니다. thinking 모델, 생각 과정 공개, 특별한 추론 능력을 새로 요구하지 마세요. '없다/필요 없다'는 조건도 임의로 추가하지 마세요.
불가능 여부를 판단하거나 모델을 추천하지 말고, 주어진 작업을 수행할 모델의 추천을 요청하는 사람처럼 쓰세요. 너무 기술적인 설명·중복 인사·정형적인 맺음말을 피하세요. 작성 지침에 언급한 금지 사항을 '필요 없다/하지 않는다/아직 정하지 않았다'라는 문장으로 바꿔 출력하는 것도 금지입니다. 글 안에서 '지정되지 않았다'를 설명하지 마세요. '예를 들어'라며 데이터 종류나 분야를 새로 정하지 마세요. 같은 조건을 두 항목에서 반복하지 말고 첫 항목에도 학습 방법을 묻는 질문을 포함하세요. 작성 방향은 이 질문 글을 쓰는 당신의 문체 지시일 뿐, 추천받을 모델의 말투·답변 방식·기능 조건이 아닙니다. 문체 지시를 사용자 요구사항의 내용에 옮기지 마세요. 첫 항목에서 학습하고 싶은 일과 모델·학습법에 대한 질문을 충분히 설명한 다음 두 번째 항목에는 GPU·언어·제공된 분량 중 아직 쓰지 않은 사실만 서술하세요. 없는 조건을 채우기 위해 문장을 늘리지 마세요.'''

def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def brief_for(case,cid):
    rng=random.Random(int(digest('stage4-v4/'+cid),16));task=case['task'];n=case['context_length']
    alias=rng.choice(gpu_aliases(case['gpu']))['text'];lang=LANGUAGES[case['language']]
    ctx=dict(source_context_length=n,applicable=task not in NON_TEXT,exact_recoverable=True,is_tokenizer_measurement=False)
    workload=None
    if task not in NON_TEXT:
        choices=['tokens','written','k'] if n>=1024 else ['tokens','written']
        if task in {'Classification','Question Answering','Summarization','Translation'} and n>=1024:choices+=['pages','pages']
        style=rng.choice(choices)
        unit='지시문·입력과 생성될 답변을 합친 총 텍스트 처리 분량'
        if task=='ASR':unit='지시문과 받아쓴 전사문을 합친 텍스트 분량(오디오 재생 시간이 아님)'
        if task=='TTS':unit='지시문과 읽어 줄 문장을 합친 텍스트 분량(음성 재생 시간이 아님)'
        surface=f'{n:,}토큰' if style=='tokens' else korean_number(n)+' 토큰' if style=='written' else f'{n//1024}K' if style=='k' else f'{rng.choice(["보고서","문서","책"])} {n//512}{rng.choice(["페이지","쪽"])}'
        ctx.update(style=style,surface=surface,budget_tokens=n,scope=unit)
        if style=='pages':ctx.update(exact_recoverable=False,assumed_tokens_per_page=512,pages=n//512,conversion_origin='synthetic_system_assumption_not_user_constraint')
        workload=f'{unit}: {surface} 정도'
        if style=='pages':workload+=' 사용자 문장에는 토큰 수나 페이지당 환산 수치를 쓰지 않는다.'
    else:ctx.update(style='not_applicable',budget_tokens=None,exact_recoverable=False)
    style=dict(voice=rng.choice(VOICES),order=rng.choice(ORDERS),requirements=rng.choice(REQ_STYLES),information_placement=rng.choice(['주요 조건을 첫 항목에도 자연스럽게 섞기','GPU와 분량은 두 번째 항목에 두고 첫 항목은 작업과 문의 중심','언어와 GPU를 첫 항목에 쓰고 두 번째에서 그 조건을 연결','조건 순서를 자유롭게 섞되 중복을 줄이기']))
    facts={'task':TASK_MEANING[task],'gpu_surface':alias,'language_surface':lang,'language_role':'발화 언어' if task in {'ASR','TTS'} else '작업에서 사용할 언어'  if task in NON_TEXT else '작업 언어',
           'workload':workload}
    if workload is None:del facts['workload']
    if workload:
        workload+= (' 분량은 지시문과 음성으로 읽을 입력 텍스트의 합입니다. 생성 음성이나 출력 답변을 토큰 분량에 더한다고 쓰지 마세요.' if task=='TTS' else ' 분량은 지시문과 받아쓴 전사문의 합이며, 입력 음성의 길이를 토큰 수로 표시하지 마세요.' if task=='ASR' else ' 분량은 원문만이 아니라 입력과 출력 결과를 합친 총량이라는 뜻을 분명히 밝혀주세요.')
        facts['workload']=workload
    prompt='주어진 사실(그 밖의 조건을 새로 만들지 않기):\n'+json.dumps(facts,ensure_ascii=False)+'\n작성 방향(문장 틀이 아니라 이번 글의 느낌):\n'+json.dumps(style,ensure_ascii=False)+'\nGPU 표기와 언어 이름, 분량 표현은 제공된 표현을 한 번 이상 그대로 포함하되 주변 문장과 배치는 자유롭게 작성하세요.'
    prompt+='\n이 문체 지시를 작업 내용으로 말하지 마세요. '+('처리 분량이 제공되지 않은 작업이므로 입력·출력 크기나 분량 관리라는 조건을 쓰지 마세요.' if not workload else '총량의 의미와 분량 수치를 모두 지키세요.')
    return prompt,dict(gpu_alias=alias,gpu_identity=gpu_identity(case['gpu']),language_name=lang,context=ctx,style_direction=style,answer_fields_given_to_generator=False)
