# InternVL3.5-8B Video-Query Baseline 실험

## 개요

InternVL3.5-8B를 이용해 비디오 이해 파이프라인의 **FPS / num_frames 별 latency·정확도** 트레이드오프를 측정하는 실험 코드.

- **모델**: `OpenGVLab/InternVL3_5-8B` (bfloat16, transformers==4.51.3)
- **데이터셋**: MSRVTT-QA test split + MVBench (20개 sub-task)
- **절단점**: ViT → **Pixel Shuffle 직후** (프레임당 256 토큰) → `.pt` 캐시
- **GPU 타이밍**: `torch.cuda.Event` 전용 (CPU 타이머 사용 금지)
- **디버그 모드**: 모든 sweep 조건당 200개 영상 제한 (`--max_videos` / `--max_samples`)

---

## 파이프라인 구조

```
[Stage A: 오프라인 인제스트]
  video → decord FPS 다운샘플 → preprocess → ViT → Pixel Shuffle → .pt 저장

[Stage B: 온라인 쿼리]
  .pt 로드 → uniform subsample(num_frames) → mlp1 → model.chat() → <frame>K</frame> 파싱
```

### Stage B 핵심: extract_feature monkey-patch

ViT+Pixel Shuffle 결과를 캐시로 갖고 있으므로, `mlp1`만 적용한 뒤
`model.extract_feature`를 monkey-patch해서 `model.chat()`이 그대로 동작하게 함.
InternVL 코드를 fork/수정하지 않고 공식 모듈 호출 순서만 변경. ("변형 없음" 원칙)

```
embed_with_cache():
  visual_embeds = model.mlp1(cached_embeddings)   # [N, 256, LLM_dim]
  model.extract_feature = lambda pv: visual_embeds  # monkey-patch
  model.chat(dummy_pv, ..., num_patches_list=[1]*N)  # 공식 API 그대로
```

---

## 실험 매트릭스

| Sweep | 변수 | 값 | 고정 값 | 영상 수 (조건당) |
|-------|------|-----|---------|----------------|
| 1 (Stage A) | `target_fps` | 1, 2, 5, 10, 30 | `num_frames=8` | 200 |
| 2 (Stage B) | `num_frames` | 8, 16, 32 | `target_fps=5` | 200 |

각 조건은 독립 프로세스로 실행되므로 조건마다 200개씩 처리.

---

## 환경 설정

### Python 버전

Python **3.10** 권장 (decord 공식 wheel 기준)

```bash
conda create -n effvlm python=3.10
conda activate effvlm
pip install -r requirements.txt
```

> `transformers==4.51.3` 고정 필수.
> - 4.44.x 이하: Qwen3Config 없음 (InternVL3.5 로드 불가)
> - 4.52.x 이상: `all_tied_weights_keys` AttributeError 발생

---

## 데이터 준비

### MSRVTT-QA

```
data/msrvtt/
├── videos/           ← MP4 파일들 (videos/all/ 아님)
│   ├── video0.mp4
│   ├── video1.mp4
│   └── ...
├── test_qa.json      ← MSRVTT-QA.zip에서 압축 해제 (MEGA에서 수동 다운로드)
└── video_list.tsv    ← video_id TAB 절대경로 (스크립트가 자동 생성)
```

**다운로드**: MSRVTT 영상은 공식 라이선스 제한으로 수동 다운로드 필요.
`test_qa.json`은 MEGA에서 MSRVTT-QA.zip 다운로드 후 압축 해제.

```bash
bash scripts/01_download_msrvtt.sh data/msrvtt
```

### MVBench

```
data/mvbench/
├── json/
│   ├── action_antonym.json
│   ├── action_count.json
│   └── ... (20개 task JSON)
└── video/
    ├── ssv2_video/       ← source별 하위 폴더 (구조는 task마다 다름)
    ├── clevrer/
    ├── FunQA_test/
    └── ...               ← .webm, .mp4 혼재 가능
```

> MVBench JSON의 `video` 필드는 파일명만 포함 (경로 없음).
> 코드가 시작 시 `video/` 전체를 `os.walk()`로 재귀 스캔해서 `{filename → 절대경로}` 인덱스를 자동 구축.

```bash
bash scripts/02_download_mvbench.sh data/mvbench
```

---

## 실험 실행 순서

```bash
# 1. 환경변수 설정
export MODEL_PATH="OpenGVLab/InternVL3_5-8B"
export MSRVTT_ROOT="data/msrvtt"
export MVBENCH_ROOT="data/mvbench"

# 2. Gate 테스트 (반드시 통과 확인 후 실험 진행)
pytest tests/ -v

# 3. Sweep 1: Stage A FPS sweep (조건당 200개, tqdm 진행바 표시)
bash scripts/03_run_sweep1.sh

# 4. Sweep 2: Stage B num_frames sweep (조건당 200개, tqdm 진행바 표시)
bash scripts/04_run_sweep2.sh

# 5. 결과 집계 및 시각화
python scripts/05_aggregate.py
```

### GPU 클럭 고정 (권장)

```bash
LOCK_CLOCKS=1 bash scripts/03_run_sweep1.sh
```

---

## 진행 상황 확인 (tqdm)

실행 중 터미널에 아래와 같이 표시된다:

```
Stage A [msrvtt 5fps]:  45%|████████████       | 90/200 [02:13<02:41,  0.68video/s, ok=88, skip=2]
Stage B [mvbench nf=16]: 30%|██████             | 60/200 [05:22<12:10,  0.19item/s, ok=57, skip=2, na=1]
```

- `ok`: 정상 처리 수
- `skip`: 파일 없음 / decord 실패 등
- `na` (Stage B): F < num_frames → SubsampleSkipError (N/A 처리)

---

## 핵심 규칙

| 규칙 | 위치 |
|------|------|
| Pixel Shuffle 직후 절단 (scale=0.5, 256토큰/프레임) | `src/ingest/pixel_shuffle.py` |
| F < num_frames → SubsampleSkipError (절대 skip하지 않음, N/A로 기록) | `src/query/frame_subsample.py` |
| `<frame>K</frame>` 출력 파싱 | `src/query/grounding.py` |
| decord로 FPS 다운샘플 인덱싱 | `src/ingest/frame_indexing.py` |
| GPU timing: `torch.cuda.Event` 전용 | `src/timing/cuda_timer.py` |
| extract_feature monkey-patch (ViT bypass) | `src/query/injector.py` |
| MVBench 재귀 파일 인덱싱 | `src/data/mvbench.py` |

---

## 산출물

```
outputs/
├── embeddings/
│   ├── msrvtt/
│   │   ├── 1fps/        ← Stage A .pt 파일 (video_id.pt)
│   │   ├── 2fps/
│   │   ├── 5fps/
│   │   ├── 10fps/
│   │   └── 30fps/
│   └── mvbench/
│       ├── 1fps/
│       └── ...
├── responses/
│   └── sweep2_*.jsonl   ← Stage B 응답 (video_id, question, answer_gt, response, frame_k)
├── timings/
│   └── *.jsonl          ← JSONL 타이밍 로그 (Stage A/B 각 단계 ms)
└── report/
    ├── *.csv            ← 집계 테이블
    └── *.png            ← 그래프
```

---

## Gate 테스트 목록

| 테스트 파일 | 검증 내용 |
|------------|---------|
| `tests/test_frame_indexing.py` | FPS 다운샘플 인덱스 정확성 |
| `tests/test_pixel_shuffle_equivalence.py` | 분리 파이프라인 ≡ `model.chat()` 동등성 (CUDA 필요) |
| `tests/test_subsample_skip.py` | F < num_frames → SubsampleSkipError 발생 확인 |
| `tests/test_injector_smoke.py` | injector 기본 동작 (CPU 가능) |

---

## 한계 사항

- `<frame>K</frame>` ground truth 없음: 정량 평가 불가, 50개 qualitative 분석만 가능
- "실시간" 시뮬레이션은 파일 기반 (RTSP / 네트워크 jitter 미포함)
- MVBench × 30fps: context overflow 시 N/A
- GPU 절대 latency 수치는 A100/H100 80GB 기준
- 현재 `--max_videos 200` / `--max_samples 200` 제한 적용 중 (흐름 검증 후 제거)
