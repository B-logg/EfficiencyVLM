# InternVL3.5-8B Video-Query Baseline 실험

## 개요

InternVL3.5-8B를 이용해 비디오 이해 파이프라인의 **FPS / num_frames 별 latency·정확도** 트레이드오프를 측정하는 실험

- **모델**: `OpenGVLab/InternVL3_5-8B` (bfloat16, transformers==4.51.3)
- **데이터셋**: MSRVTT-QA test split + MVBench (20개 sub-task)
- **절단점**: ViT → **Pixel Shuffle 직후** (프레임당 256 토큰) → `.pt` 캐시
- **GPU 타이밍**: `torch.cuda.Event` 전용 (CPU 타이머 사용 금지)
- **디버그 모드**: 모든 sweep 조건당 100개 영상 제한 (`--max_videos` / `--max_samples`)

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
| 1 (Stage A) | `target_fps` | 1, 2, 5, 10, 30 | `num_frames=8` | 100 |
| 2 (Stage B) | `num_frames` | 4, 8, 16 | `target_fps=5` | 100 |

각 조건은 독립 프로세스로 실행되므로 조건마다 100개씩 처리.

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

