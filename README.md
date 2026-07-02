# UGround 논문 리뷰 및 재현 실험

**논문:** [Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents](https://arxiv.org/abs/2410.05243) (ICLR 2025 Oral)

## 실험 결과 (UGround-V1-7B, ScreenSpot 300샘플)

| 카테고리 | 정확도 |
|---------|-------|
| mobile_text | 100.0% |
| mobile_icon | 84.0% |
| desktop_text | 92.0% |
| desktop_icon | 84.0% |
| web_text | 94.0% |
| web_icon | 78.0% |
| **평균** | **88.67%** |

논문 보고 값(72B 기준 89.4%)과 거의 동일한 수치를 7B 모델로 재현.

## 구조

```
UGround/
├── grounding/          # 공식 추론 코드 (vllm 기반)
├── offline_evaluation/ # ScreenSpot 평가 코드
├── demo/
│   ├── run_demo.py          # 샘플 이미지 추론 데모
│   ├── run_screenspot.py    # ScreenSpot 벤치마크 실행
│   └── images/              # 데모용 샘플 UI 이미지
├── screenspot/
│   ├── questions.jsonl  # 평가 질문 (300개)
│   └── answers.jsonl    # 모델 예측 결과
└── train/              # 학습 코드
```

## 환경 설치

```bash
conda create -n uground python=3.10 -y
conda activate uground
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate qwen-vl-utils pillow huggingface_hub datasets
```

## 모델 다운로드

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='osunlp/UGround-V1-7B',
    local_dir='models/UGround-V1-7B'
)
```

## 추론 데모 실행

```bash
conda activate uground
python demo/run_demo.py
```

## ScreenSpot 평가 실행

```bash
python demo/run_screenspot.py
python offline_evaluation/ScreenSpot/eval.py \
  --ans_file screenspot/answers.jsonl \
  --image_dir screenspot/images
```
