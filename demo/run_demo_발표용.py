"""
[발표용] UGround GUI Grounding 데모
=====================================
UGround 모델이 화면 이미지를 보고
"이 버튼 클릭해줘" 같은 자연어 명령에서
정확한 픽셀 좌표를 찾아내는 과정을 보여줍니다.

사용 모델: UGround-V1-7B
"""

import torch
from PIL import Image, ImageDraw
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
모델_경로 = "/nas/home/lys0426/UGround/models/UGround-V1-7B"
이미지_경로 = "/nas/home/lys0426/UGround/demo/images/sample_ui.png"

# 찾고 싶은 UI 요소 목록 (자연어로 설명)
찾을_요소_목록 = [
    "the search button",       # 검색 버튼
    "the login button",        # 로그인 버튼
    "the like button",         # 좋아요 버튼
    "the search input field",  # 검색 입력창
]

# ──────────────────────────────────────────
# 1단계: 모델 로딩
# ──────────────────────────────────────────
print("=" * 50)
print("1단계: UGround-V1-7B 모델 로딩 중...")
print("=" * 50)

모델 = Qwen2VLForConditionalGeneration.from_pretrained(
    모델_경로,
    torch_dtype=torch.float16,  # 메모리 절약을 위해 float16 사용
    device_map="auto",          # GPU 자동 배분
)
전처리기 = AutoProcessor.from_pretrained(모델_경로)
print("모델 로딩 완료!\n")

# ──────────────────────────────────────────
# 2단계: 입력 이미지 확인
# ──────────────────────────────────────────
이미지 = Image.open(이미지_경로)
가로, 세로 = 이미지.size

print("=" * 50)
print(f"2단계: 입력 이미지 정보")
print("=" * 50)
print(f"  파일: {이미지_경로}")
print(f"  크기: {가로} x {세로} 픽셀\n")

# ──────────────────────────────────────────
# 3단계: 각 요소의 좌표 예측
# ──────────────────────────────────────────
print("=" * 50)
print("3단계: UI 요소 위치 예측")
print("=" * 50)
print("입력: 이미지 + 자연어 설명")
print("출력: 픽셀 좌표 (x, y)\n")

예측_결과 = []

for 설명 in 찾을_요소_목록:

    # 프롬프트 구성: 이미지 + 텍스트
    메시지 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": 이미지_경로},
                {"type": "text", "text": (
                    "Your task is to help the user identify the precise coordinates (x, y) "
                    "of a specific area/element/object on the screen based on a description.\n"
                    f"Description: {설명}\n"
                    "Answer:"
                )},
            ],
        }
    ]

    # 모델 입력 준비
    텍스트 = 전처리기.apply_chat_template(메시지, tokenize=False, add_generation_prompt=True)
    이미지_입력, _ = process_vision_info(메시지)
    입력값 = 전처리기(text=[텍스트], images=이미지_입력, return_tensors="pt").to(모델.device)

    # 모델 추론 (좌표 예측)
    with torch.no_grad():
        출력_토큰 = 모델.generate(**입력값, max_new_tokens=32, temperature=None, do_sample=False)

    # 출력 디코딩
    모델_출력 = 전처리기.batch_decode(
        출력_토큰[:, 입력값["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    # 비율 좌표(0~1000) → 실제 픽셀 좌표 변환
    try:
        x비율, y비율 = eval(모델_출력)
        x좌표 = int(x비율 / 1000 * 가로)
        y좌표 = int(y비율 / 1000 * 세로)
        print(f'  찾는 요소: "{설명}"')
        print(f"  모델 원본 출력: {모델_출력}  (0~1000 비율 좌표)")
        print(f"  실제 픽셀 좌표: ({x좌표}, {y좌표})\n")
        예측_결과.append((설명, x좌표, y좌표))
    except Exception as 오류:
        print(f'  "{설명}" → 파싱 실패: {오류}\n')

# ──────────────────────────────────────────
# 4단계: 결과 시각화
# ──────────────────────────────────────────
print("=" * 50)
print("4단계: 결과 이미지에 좌표 표시")
print("=" * 50)

결과_이미지 = Image.open(이미지_경로)
그리기 = ImageDraw.Draw(결과_이미지)
색상_목록 = ["red", "blue", "green", "orange"]

for 순번, (설명, x, y) in enumerate(예측_결과):
    반지름 = 15
    색 = 색상_목록[순번 % len(색상_목록)]
    # 원 표시
    그리기.ellipse([x - 반지름, y - 반지름, x + 반지름, y + 반지름], outline=색, width=4)
    # 텍스트 라벨
    그리기.text((x + 반지름 + 5, y - 10), 설명[:15], fill=색)

저장_경로 = "/nas/home/lys0426/UGround/demo/result_visualized_발표용.png"
결과_이미지.save(저장_경로)
print(f"결과 이미지 저장 완료: {저장_경로}")
print("\n[완료] 모든 UI 요소의 좌표 예측 성공!")
