"""Explain the preserved negative judgment about the target task, not the recommender."""
TASK_REASONS={
 'Translation':'의미를 보존하는 번역',
 'Summarization':'원문의 핵심을 추리는 요약',
 'Question Answering':'질문에 대한 답변 생성',
 'Classification':'텍스트의 범주 분류',
 'Image Classification':'이미지의 범주 분류',
 'Detection':'이미지의 객체 검출',
 'Segmentation':'이미지 영역 분할',
 'ASR':'음성을 텍스트로 전사',
 'TTS':'텍스트를 음성으로 합성',
 'Audio Classification':'소리의 범주 분류',
}
def reason_for(task):
    return (f"추천 대상 작업은 {TASK_REASONS[task]}입니다. 기존 요청 검토에서 별도의 thinking 모델/모드를 요구할 근거가 확인되지 않았으므로 기본 task 모델을 우선합니다. "
            "이는 모델 추천 과정의 난이도나 모델 자체의 일반적 추론 능력에 대한 판단이 아닙니다. 복잡한 다단계 추론 요구가 추가되면 재검토합니다.")
