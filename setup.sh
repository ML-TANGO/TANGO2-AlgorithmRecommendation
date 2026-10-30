#!/usr/bin/env bash
# TANGO2 AR · PO · DA 통합 Docker 설치 스크립트
#
# 사용법:
#   bash setup.sh                     # 한 번에: 세 모듈 .env 준비 → Docker 이미지 빌드 → 세 서비스 기동
#   bash setup.sh --no-up             # 빌드까지만 하고 기동은 하지 않음
#   bash setup.sh --mock              # PO/DA를 mock 프로필로 (GPU 없이 API 흐름 확인; AR은 gpu 전용)
#   bash setup.sh --only ar,po        # 일부 모듈만 (ar / po / da, 쉼표 구분)
#   bash setup.sh --no-cache          # 캐시 없이 처음부터 빌드
#   bash setup.sh --with-conda        # 추가로 AR 로컬 conda 환경(etri)도 설치 (데이터 재생성 · 재학습용)
#
# 전제: NVIDIA 드라이버, Docker Engine + Compose v2, NVIDIA Container Toolkit.
#       베이스 모델(google/gemma-4-E4B-it 등)은 호스트 Hugging Face 캐시에 미리 받아 둡니다.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="gpu"; DO_UP=1; NO_CACHE=""; WITH_CONDA=0; ONLY="ar,po,da"; ENV_NAME="${ENV_NAME:-etri}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mock) PROFILE="mock" ;;
    --up) DO_UP=1 ;;
    --no-up) DO_UP=0 ;;
    --no-cache) NO_CACHE="--no-cache" ;;
    --with-conda) WITH_CONDA=1 ;;
    --only) ONLY="$2"; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done
want() { [[ ",$ONLY," == *",$1,"* ]]; }
log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1; }
dir_of() { case "$1" in ar) echo ar-module ;; po) echo prompt-module ;; da) echo da-module ;; esac; }
profile_of() { [[ "$1" == "ar" ]] && echo gpu || echo "$PROFILE"; }   # AR 은 gpu 프로필만 있음

# ---------- 1. 사전 점검 ----------
log "사전 점검"
need docker || { echo "docker 가 필요합니다." >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "docker compose v2 가 필요합니다." >&2; exit 1; }
if docker info >/dev/null 2>&1; then DOCKER="docker"
else warn "docker 소켓 권한이 없어 'sudo docker' 를 사용합니다. (영구 해결: sudo usermod -aG docker $USER 후 재로그인)"; DOCKER="sudo docker"; fi
if need nvidia-smi; then nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'
else warn "nvidia-smi 없음. gpu 프로필은 동작하지 않습니다 (--mock 권장)."; fi

# ---------- 2. 모듈별 .env ----------
log ".env 준비 (없을 때만 .env.example 복사)"
for m in ar po da; do
  want "$m" || continue; d="$ROOT/$(dir_of $m)"
  if [[ -f "$d/.env" ]]; then echo "  $(dir_of $m)/.env 있음 (유지)"; else cp "$d/.env.example" "$d/.env"; echo "  $(dir_of $m)/.env 생성"; fi
done

# ---------- 3. Hugging Face 캐시 위치 ----------
log "베이스 모델 캐시 확인"
HF_HUB=""; CANDIDATES=("${HF_HOME:-}/hub" "$HOME/.cache/huggingface/hub" "${SUDO_USER:+/home/$SUDO_USER/.cache/huggingface/hub}" /home/.cache/huggingface/hub)
# 1순위: 베이스 모델(google/gemma-4-E4B-it)이 실제로 들어 있는 캐시, 2순위: 아무 모델이나 있는 캐시
for c in "${CANDIDATES[@]}"; do
  [[ -n "$c" && -d "$c/models--google--gemma-4-E4B-it/snapshots" ]] && { HF_HUB="$(cd "$c" && pwd)"; break; }
done
if [[ -z "$HF_HUB" ]]; then
  for c in "${CANDIDATES[@]}"; do
    [[ -n "$c" && -d "$c" ]] && ls "$c" 2>/dev/null | grep -q '^models--' && { HF_HUB="$(cd "$c" && pwd)"; break; }
  done
fi
if [[ -n "$HF_HUB" ]]; then
  echo "  캐시: $HF_HUB"; ls "$HF_HUB" | grep '^models--' | sed 's/^/    /'
  ls -d "$HF_HUB/models--google--gemma-4-E4B-it" >/dev/null 2>&1 || warn "google/gemma-4-E4B-it 가 캐시에 없습니다: huggingface-cli download google/gemma-4-E4B-it"
  if want ar; then
    if ! grep -q '^HF_CACHE=' "$ROOT/ar-module/.env"; then
      printf '\n# Host HF hub cache mounted into the container (absolute path; "~" would resolve to /root under sudo)\nHF_CACHE=%s\n' "$HF_HUB" >> "$ROOT/ar-module/.env"
      echo "  ar-module/.env 에 HF_CACHE=$HF_HUB 기록"
    else
      cur="$(grep '^HF_CACHE=' "$ROOT/ar-module/.env" | cut -d= -f2-)"
      if [[ ! -d "$cur/models--google--gemma-4-E4B-it/snapshots" ]]; then
        sed -i "s#^HF_CACHE=.*#HF_CACHE=$HF_HUB#" "$ROOT/ar-module/.env"
        warn "ar-module/.env 의 HF_CACHE($cur)에 모델이 없어 $HF_HUB 로 변경했습니다."
      fi
    fi
  fi
  if want da && grep -q '^DA_MODELS_DIR=../prompt-module/models' "$ROOT/da-module/.env"; then
    sed -i "s#^DA_MODELS_DIR=.*#DA_MODELS_DIR=$HF_HUB#" "$ROOT/da-module/.env"; echo "  da-module/.env 의 DA_MODELS_DIR 를 $HF_HUB 로 변경"
  fi
  want po && echo "  prompt-module: docker-compose.yml 의 /models 볼륨(현재 /home/.cache/huggingface/hub)과 /adapters 경로를 확인하세요"
else
  warn "Hugging Face 캐시를 찾지 못했습니다. 모델을 받은 뒤 ar-module/.env 의 HF_CACHE, da-module/.env 의 DA_MODELS_DIR 를 절대경로로 지정하세요."
fi

# ---------- 4. Docker 이미지 빌드 ----------
for m in ar po da; do
  want "$m" || continue; d="$ROOT/$(dir_of $m)"; p="$(profile_of $m)"
  log "$(dir_of $m): docker compose --profile $p build $NO_CACHE"
  (cd "$d" && $DOCKER compose --profile "$p" build $NO_CACHE)
done

# ---------- 5. (선택) AR 로컬 conda 환경 ----------
if [[ $WITH_CONDA -eq 1 ]]; then
  log "AR 로컬 conda 환경 '$ENV_NAME' (python 3.12) — 데이터 재생성 · 재학습용"
  need conda || { echo "conda 가 필요합니다." >&2; exit 1; }
  conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" || conda create -n "$ENV_NAME" python=3.12 -y
  PIP=(conda run -n "$ENV_NAME" --no-capture-output python -m pip install)
  "${PIP[@]}" --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
  "${PIP[@]}" -r "$ROOT/ar-module/requirements.txt" -r "$ROOT/ar-module/requirements-demo.txt" -r "$ROOT/ar-module/requirements-dev.txt"
fi

# ---------- 6. 기동 또는 안내 ----------
if [[ $DO_UP -eq 1 ]]; then
  for m in po da ar; do
    want "$m" || continue; d="$ROOT/$(dir_of $m)"; p="$(profile_of $m)"
    [[ "$m" == "ar" ]] && mkdir -p "$d/artifacts"   # 볼륨 폴더를 사용자 소유로 미리 생성 (Docker가 만들면 root 소유)
    log "$(dir_of $m): docker compose --profile $p up -d"
    (cd "$d" && $DOCKER compose --profile "$p" up -d)
  done
fi
log "완료"
cat <<MSG
  실행 (기본 실행이면 이미 기동됨; --no-up 이었을 때만 필요):
    cd $ROOT/prompt-module && $DOCKER compose --profile $PROFILE up -d      # PO  http://localhost:8000/docs
    cd $ROOT/da-module     && $DOCKER compose --profile $PROFILE up -d      # DA  http://localhost:8002/docs , 수신기 :8003/last
    cd $ROOT/ar-module     && $DOCKER compose --profile gpu up -d           # AR  http://localhost:8004
  확인:
    curl -sS http://localhost:8000/health ; curl -sS http://localhost:8002/health ; curl -sS http://localhost:8004/api/status
  로그 / 중지:  <모듈 디렉터리>에서  $DOCKER compose --profile <gpu|mock> logs -f  /  ... down
  GPU 는 세 모듈이 기본으로 0번을 씁니다. 동시에 띄우면 메모리가 부족하므로
    ar-module/.env AR_GPU_ID, da-module/.env DA_GPU_ID, prompt-module/docker-compose.yml device_ids 로 나눠 주세요.
MSG
