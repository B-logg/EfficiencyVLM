# InternVL3.5 Baseline Video-Query 실험

## 개요
- **모델**: OpenGVLab/InternVL3_5-8B (bfloat16)
- **데이터셋**: MSRVTT-QA test split + MVBench full
- **절단점**: Pixel Shuffle 직후 (256토큰/타일) 저장
- **FPS 다운샘플링**: decord 인덱싱

## 실험 순서

```bash
# 0. 의존성 설치
pip install -r requirements.txt

# 1. 데이터 준비
bash scripts/01_download_msrvtt.sh data/msrvtt
bash scripts/02_download_mvbench.sh data/mvbench

# 2. Gate 테스트 (반드시 통과 확인)
pytest tests/ -v

# 3. Sweep 1: Stage A FPS sweep
export MODEL_PATH="OpenGVLab/InternVL3_5-8B"
export MSRVTT_ROOT="data/msrvtt"
export MVBENCH_ROOT="data/mvbench"
bash scripts/03_run_sweep1.sh

# 4. Sweep 2: Stage B num_frames sweep
bash scripts/04_run_sweep2.sh

# 5. 표/그림 자동 생성
python scripts/05_aggregate.py
```

## 실험 매트릭스

| Sweep | 변수 | 값 | 고정 |
|-------|------|----|------|
| 1 | target_fps | 1,2,5,10,30 | num_frames=8 |
| 2 | num_frames | 8,16,32 | target_fps=5 |

## 핵심 규칙
- Pixel Shuffle 직후 절단: `src/ingest/pixel_shuffle.py`
- F < num_frames → N/A skip: `src/query/frame_subsample.py`
- `<frame>K</frame>` 출력 파싱: `src/query/grounding.py`
- decord 인덱싱 다운샘플링: `src/ingest/frame_indexing.py`
- GPU timing: `torch.cuda.Event` 전용 (`src/timing/cuda_timer.py`)

## 산출물
- `outputs/embeddings/`: Stage A .pt 파일
- `outputs/responses/`: Stage B JSONL 응답
- `outputs/timings/`: JSONL 타이밍 로그
- `outputs/report/`: 표(CSV) + 그림(PNG)

## 한계 (§8)
- 응답 프레임 ground truth 없음: `<frame>K</frame>` 정량 평가 불가, 50개 qualitative만
- "실시간"은 파일 기반 시뮬레이션 (RTSP/jitter 미포함)
- MVBench × 30FPS: context overflow 시 N/A
- GPU 절대 latency는 A100/H100 80GB 기준
