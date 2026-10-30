"""Component-aware training plans and exact adapter target resolution.

Plans describe required loading/forward behavior; producing a plan does not
load, remove or train a model. Only the scoped name resolver executes here.
"""
import re

NLP={'Classification','Question Answering','Summarization','Translation'}
CV={'Image Classification','Detection','Segmentation'}


def build_training_plan(case, profile, algorithm):
    mid=profile['model_id'];family=profile['family'];task=case['task']
    adapter=algorithm in {'LoRA','QLoRA'}
    if algorithm not in profile['algorithms']:
        raise ValueError(f'{algorithm} is not supported by the Stage 2 profile for {mid}')
    plan=dict(input_modalities=[],output_modalities=[],active_components=[],
        trainable_components=[],frozen_but_used_components=[],components_to_not_load_if_present=[],
        adapter_scope_prefixes=[],adapter_scope_semantics='Exact transformer block prefixes; resolve full module paths before PEFT injection',
        quantization_scope_prefixes=[],training_objective=profile['objective'],
        workload=profile['workload'],model_architecture=profile.get('architecture'),
        applied_to_live_model=False,requires_runtime_module_verification=True,
        module_path_warning='Module paths differ across backends; accept only enumerated matching prefixes, never fall back to all modules.')
    if task in NLP:
        plan.update(input_modalities=['text'],output_modalities=['text'],
            active_components=['language_backbone','token_embeddings','lm_head'],
            components_to_not_load_if_present=['vision_encoder','audio_encoder','modality_projectors'],
            adapter_scope_prefixes=['model.layers'],
            loading_strategy='Load the text causal-LM architecture only; do not allocate optional audio/vision towers or run their forward paths.')
        if family=='Gemma 3' and profile.get('image_input'):
            plan.update(text_only_model_class='Gemma3ForCausalLM',text_config_key='text_config',
                checkpoint_key_mapping='model.language_model.* -> model.*; retain lm_head; exclude model.vision_tower.* and model.multi_modal_projector.*',
                loading_strategy='Extract/load Gemma3 text_config and language weights with Gemma3ForCausalLM; reject missing text weights. Do not load the multimodal wrapper for these text cases.')
        elif 'Qwen3-VL' in mid:
            plan['adapter_scope_prefixes']=['model.language_model.layers']
            plan['loading_strategy']='Use a verified text-only extraction of the Qwen3-VL language backbone; language-weight mapping and wrapper must be implemented before execution.'
    elif task in CV:
        plan.update(input_modalities=['image'],output_modalities=['class_scores' if task=='Image Classification' else 'boxes_and_class_scores' if task=='Detection' else 'semantic_masks'],
            components_to_not_load_if_present=['audio_encoder','language_generation_backbone'],
            loading_strategy='Load the dedicated CV architecture with its image processor and required output head; no text/audio model is needed.')
        if family in {'ViT','ConvNeXT'}:
            plan['active_components']=['vision_encoder','classification_head']
        elif family in {'DETR','RT-DETR'}:
            plan['active_components']=['vision_backbone','detection_encoder_decoder','box_and_class_heads']
        elif family=='SegFormer':
            plan['active_components']=['vision_encoder','segmentation_decode_head']
        else:raise ValueError(f'Unreviewed CV family: {family}')
        if adapter:raise ValueError('Current CV Stage 2 profiles permit SFT only; do not silently change the label to LoRA')
    elif task=='Audio Classification':
        plan.update(input_modalities=['audio'],output_modalities=['class_scores'],
            active_components=['audio_encoder','classification_head'],
            components_to_not_load_if_present=['vision_encoder','language_generation_backbone'],
            loading_strategy='Load the dedicated audio classifier and feature extractor only.',
            loss_requirement='AST multi-label events require BCEWithLogitsLoss and float multi-hot targets; keyword spotting uses single-label cross entropy.')
        if adapter:raise ValueError('Current audio-classification profiles permit SFT only')
    elif task=='ASR' and 'Qwen3-ASR' in mid:
        plan.update(input_modalities=['audio'],output_modalities=['text'],
            active_components=['audio_encoder','audio_to_text_projection','language_decoder','lm_head'],
            components_to_not_load_if_present=['vision_encoder'],
            adapter_scope_prefixes=['thinker.model.layers'],
            loading_strategy='Retain audio features/encoder and the transcription decoder. These components are all needed for ASR; freezing the audio encoder does not disable its forward pass.')
        if adapter:plan['frozen_but_used_components']=['audio_encoder','audio_to_text_projection','base_language_decoder','lm_head']
    elif task=='ASR' and family=='Whisper':
        plan.update(input_modalities=['audio'],output_modalities=['text'],
            active_components=['audio_encoder','text_decoder','lm_head'],
            components_to_not_load_if_present=['vision_encoder'],
            loading_strategy='Use WhisperForConditionalGeneration with audio encoder and text decoder; keep both for transcription.')
    elif task=='TTS' and 'Qwen3-TTS' in mid:
        plan.update(input_modalities=['text','reference_audio'],output_modalities=['audio'],
            active_components=['text_embeddings','talker','code_predictor','speaker_encoder','audio_codec'],
            frozen_but_used_components=['speaker_encoder','audio_codec'],
            trainable_components=['talker_including_text_and_codec_embeddings','code_predictor'],
            components_to_not_load_if_present=['vision_encoder'],
            loading_strategy='Use the official single-speaker TTS path: retain speaker conditioning and codec; speaker embedding is detached and codec is preprocessing/decoding only.')
    else:raise ValueError(f'No component plan for {task}/{mid}')
    if adapter:
        plan['trainable_components']=['LoRA_adapter_weights_inside_declared_scope']
        if not plan['frozen_but_used_components']:
            plan['frozen_but_used_components']=['base_language_backbone','token_embeddings','lm_head']
        if not plan['adapter_scope_prefixes']:raise ValueError('Adapter scope cannot be empty')
        if algorithm=='QLoRA':plan['quantization_scope_prefixes']=plan['adapter_scope_prefixes'].copy()
    else:
        plan['adapter_scope_prefixes']=[]
        if not plan['trainable_components']:
            plan['trainable_components']=plan['active_components'].copy()
    plan['modules_to_save'] = []
    plan['unused_component_policy']='Do not load/execute unused modalities; merely setting requires_grad=False does not remove their VRAM allocation.'
    plan['required_component_policy']='Keep required heads, decoders, projectors and conditioning paths even when they are frozen.'
    plan['selection_reason'] = scope_reason(task, algorithm)
    plan['component_decisions'] = component_decisions(case, profile, algorithm)
    plan['trainable_base_weights'] = not adapter
    plan['base_parameter_policy'] = 'Freeze all base parameters; train only adapters, bias=none, modules_to_save=[]' if adapter else 'Train all parameters of the required model except explicitly frozen components; buffers and architecture-fixed statistics remain fixed'
    plan['component_decisions_exhaustive'] = False
    plan['unlisted_parameter_policy'] = 'frozen' if adapter else 'train unless explicitly frozen; required projections and query embeddings must be retained'
    plan['peft_bias'] = 'none' if adapter else None
    plan['adapter_base_weights_frozen'] = adapter
    return plan


def resolve_adapter_targets(linear_module_names, scope_prefixes, target_modules):
    """Resolve against names of actual supported linear modules, not all modules.

Returns full names and an anchored PEFT regex so a suffix shared with an audio
or vision encoder cannot cause adapters to be injected outside the scope.
"""
    if not scope_prefixes:raise ValueError('No approved adapter scope')
    names=sorted(set(linear_module_names))
    scoped=[n for n in names if any(n.startswith(p+'.') for p in scope_prefixes)]
    if not scoped:raise ValueError('No linear modules matched the approved component prefixes')
    if target_modules==['all-linear']:
        selected=scoped
    else:
        selected=[n for n in scoped if n.rsplit('.',1)[-1] in target_modules]
        missing=set(target_modules)-{n.rsplit('.',1)[-1] for n in selected}
        if missing:raise ValueError(f'Targets absent inside approved scope: {sorted(missing)}')
    if not selected:raise ValueError('No adapter targets; refusing a global fallback')
    return {'module_names':selected,'peft_target_modules_regex':r'^(?:'+ '|'.join(re.escape(n) for n in selected)+r')$'}


def scope_reason(task, algorithm):
    if task in NLP:
        return '텍스트 입력·출력에 필요한 언어 backbone을 사용합니다. Adapter 방식은 언어 블록의 추가 가중치만 학습하며 원본 가중치·embedding·LM head는 동결합니다. Classification도 현재 라벨은 텍스트로 class를 생성하는 방식입니다.'
    if task == 'ASR' and algorithm in {'LoRA', 'QLoRA'}:
        return '음성을 텍스트로 변환하려면 음성 encoder와 언어 decoder가 모두 필요합니다. 현재 정책은 공개 음성 표현을 동결·유지하고 전사 decoder의 adapter만 학습합니다. 음향 도메인 변화가 명시되지 않아 encoder 적응을 추가하지 않은 시작 구성이며 유일한 최적 구성은 아닙니다.'
    if task == 'TTS':
        return '공식 단일 화자 학습 경로를 기준으로 talker와 code predictor를 학습하고 speaker encoder와 audio codec은 동결합니다. 텍스트 입력이라는 이유로 오디오 생성 경로를 제거하지 않습니다.'
    return 'Stage 2의 전용 모델 SFT 라벨을 유지합니다. 입력 encoder뿐 아니라 해당 task의 decoder·출력 head도 학습 대상으로 보존합니다. 다른 모달리티가 실제 존재하는 경우에만 사용하지 않습니다.'


def component_decisions(case, profile, algorithm):
    task=case['task']; mid=profile['model_id']; family=profile['family']
    adapter=algorithm in {'LoRA','QLoRA'}
    def item(component, paths, mode, reason):
        return dict(component=component, module_paths=paths, mode=mode, used_in_forward=True,
                    reason=reason, runtime_path_verification_required=True)
    if task in NLP:
        prefix='model.language_model' if 'Qwen3-VL' in mid else 'model'
        return [item('language_decoder',[prefix+'.layers'],'adapter_only' if adapter else 'full_training','언어 task의 표현과 출력 분포 적응'),
                item('token_embeddings',[prefix+'.embed_tokens'],'frozen' if adapter else 'full_training','기존 어휘를 사용하며 새 토큰 추가는 가정하지 않음'),
                item('final_norm',[prefix+'.norm'],'frozen' if adapter else 'full_training','언어 출력 정규화 유지'),
                item('lm_head',['lm_head'],'frozen' if adapter else 'full_training','텍스트 출력에 필요; 원본 head를 유지')]
    if task=='ASR' and 'Qwen3-ASR' in mid:
        return [item('audio_encoder_including_projection',['thinker.audio_tower'],'frozen' if adapter else 'full_training','음성 특징과 proj1/proj2 변환을 유지'),
                item('language_decoder',['thinker.model.layers'],'adapter_only' if adapter else 'full_training','전사 문자열 생성 경로 적응'),
                item('text_embeddings_norm_and_head',['thinker.model.embed_tokens','thinker.model.norm','thinker.lm_head'],'frozen' if adapter else 'full_training','전사 입력·출력에 필요한 원본 언어 가중치')]
    if task=='ASR':
        return [item('audio_encoder',['model.encoder'],'full_training','음성 표현 적응'),item('text_decoder',['model.decoder'],'full_training','전사 생성 적응'),item('lm_head',['proj_out'],'full_training','전사 출력; tied weights는 한 번만 optimizer에 등록')]
    if task=='TTS':
        return [item('talker_including_code_predictor',['talker'],'full_training','text/codec embeddings 및 talker.code_predictor 포함'),item('speaker_encoder',['speaker_encoder'],'frozen','참조 화자 embedding은 detach'),dict(component='audio_codec',module_paths=[],mode='frozen',used_in_forward=False,used_in_preprocessing_and_decoding=True,reason='외부 speech tokenizer/codec을 전처리·출력 복원에 보존',runtime_path_verification_required=True)]
    paths={'ViT':[('vision_encoder','vit'),('classification_head','classifier')],
           'ConvNeXT':[('vision_encoder','convnext'),('classification_head','classifier')],
           'DETR':[('vision_backbone','model.backbone'),('detection_encoder','model.encoder'),('detection_decoder','model.decoder'),('class_head','class_labels_classifier'),('box_head','bbox_predictor')],
           'RT-DETR':[('vision_backbone','model.backbone'),('detection_encoder','model.encoder'),('detection_decoder','model.decoder'),('class_heads','class_embed'),('box_heads','bbox_embed')],
           'SegFormer':[('vision_encoder','segformer'),('segmentation_head','decode_head')],
           'AST':[('audio_encoder','audio_spectrogram_transformer'),('classification_head','classifier')],
           'Wav2Vec2':[('audio_encoder','wav2vec2'),('projection','projector'),('classification_head','classifier')]}[family]
    return [item(c,[path],'full_training','SFT 목적함수에 필요한 구성요소; 모델 내 고정 통계·buffer는 학습 파라미터가 아님') for c,path in paths]


def resolve_and_verify_adapter_targets(linear_modules, spec, target_modules):
    """linear_modules: {pre-PEFT full name: (in_features, out_features)}.

    Verify every expected block/shape, then generate an exact anchored regex.
    Caller must enumerate actual supported Linear/Linear4bit modules; do not
    pass a synthetic list in production. No automatic cross-backend fallback.
    """
    expected={m['name']:(m['in_features'],m['out_features']) for m in spec['modules']}
    scoped={n:tuple(shape) for n,shape in linear_modules.items() if n.startswith(spec['scope_prefix']+'.')}
    if scoped!=expected:
        missing=sorted(set(expected)-set(scoped));extra=sorted(set(scoped)-set(expected))
        bad=[n for n in expected.keys() & scoped.keys() if expected[n]!=scoped[n]]
        raise ValueError(f'Module inventory mismatch: missing={missing[:3]}, extra={extra[:3]}, wrong_shapes={bad[:3]}')
    return resolve_adapter_targets(list(linear_modules),[spec['scope_prefix']],target_modules)
