"""
[발표용] UGround ScreenSpot 벤치마크 평가
==========================================
ScreenSpot은 GUI grounding 모델을 평가하는 표준 벤치마크입니다.
- 플랫폼: 모바일 / 데스크톱 / 웹
- 요소 유형: 텍스트 버튼 / 아이콘
- 총 1,272개 샘플 (본 실험: 300개 샘플링)

평가 방식:
  모델이 예측한 좌표가 정답 bounding box 안에 들어오면 정답
"""

import torch, json, os
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
모델_경로     = "/nas/home/lys0426/UGround/models/UGround-V1-7B"
질문_파일     = "/nas/home/lys0426/UGround/screenspot/questions.jsonl"
이미지_폴더   = "/nas/home/lys0426/UGround/screenspot/images"
답변_파일     = "/nas/home/lys0426/UGround/screenspot/answers_발표용.jsonl"

배치_크기 = 4  # GPU 메모리에 따라 조절

# ──────────────────────────────────────────
# 1단계: 모델 로딩
# ──────────────────────────────────────────
print("=" * 55)
print("1단계: UGround-V1-7B 모델 로딩")
print("=" * 55)
모델 = Qwen2VLForConditionalGeneration.from_pretrained(
    모델_경로, torch_dtype=torch.float16, device_map="auto"
)
전처리기 = AutoProcessor.from_pretrained(모델_경로)
print("로딩 완료!\n")

# ──────────────────────────────────────────
# 2단계: 질문 파일 로딩
# ──────────────────────────────────────────
with open(질문_파일) as f:
    질문_목록 = [json.loads(줄) for 줄 in f]

print("=" * 55)
print(f"2단계: 평가 데이터 로딩")
print("=" * 55)
print(f"  총 질문 수: {len(질문_목록)}개")

from collections import Counter
카테고리별 = Counter(f"{q['platform']}_{q['data_type']}" for q in 질문_목록)
for 카테고리, 수 in sorted(카테고리별.items()):
    print(f"  {카테고리}: {수}개")
print()

# ──────────────────────────────────────────
# 3단계: 추론 실행
# ──────────────────────────────────────────
print("=" * 55)
print("3단계: 모델 추론 시작")
print("  입력: 화면 이미지 + UI 요소 설명")
print("  출력: 픽셀 좌표 (x, y)")
print("=" * 55)

if os.path.exists(답변_파일):
    os.remove(답변_파일)

def 배치_추론(배치):
    결과_목록 = []
    for 질문 in 배치:
        이미지_경로 = os.path.join(이미지_폴더, 질문["img_filename"])
        메시지 = [{
            "role": "user",
            "content": [
                {"type": "image", "image": 이미지_경로},
                {"type": "text", "text": (
                    "Your task is to help the user identify the precise coordinates (x, y) "
                    "of a specific area/element/object on the screen based on a description.\n"
                    f"Description: {질문['description']}\nAnswer:"
                )},
            ],
        }]

        텍스트 = 전처리기.apply_chat_template(메시지, tokenize=False, add_generation_prompt=True)
        이미지_입력, _ = process_vision_info(메시지)
        입력값 = 전처리기(text=[텍스트], images=이미지_입력, return_tensors="pt").to(모델.device)

        with torch.no_grad():
            출력_토큰 = 모델.generate(**입력값, max_new_tokens=32, temperature=None, do_sample=False)

        모델_출력 = 전처리기.batch_decode(
            출력_토큰[:, 입력값["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

        # 비율 좌표 → 실제 픽셀 좌표 변환
        with Image.open(이미지_경로) as 이미지:
            가로, 세로 = 이미지.size

        try:
            x비율, y비율 = eval(모델_출력)
            x좌표 = int(x비율 / 1000 * 가로)
            y좌표 = int(y비율 / 1000 * 세로)
            좌표_출력 = f"({x좌표}, {y좌표})"
        except:
            좌표_출력 = ""

        결과_목록.append({**질문, "output": 좌표_출력, "scale": 1.0})
    return 결과_목록

with open(답변_파일, "w") as f:
    for i in tqdm(range(0, len(질문_목록), 배치_크기), desc="추론 진행"):
        배치 = 질문_목록[i:i + 배치_크기]
        for 결과 in 배치_추론(배치):
            f.write(json.dumps(결과, ensure_ascii=False) + "\n")

print(f"\n추론 완료! 결과 저장: {답변_파일}")

# ──────────────────────────────────────────
# 4단계: 정확도 계산
# ──────────────────────────────────────────
print("\n" + "=" * 55)
print("4단계: 카테고리별 정확도 계산")
print("  정답 기준: 예측 좌표가 정답 bbox 안에 있으면 정답")
print("=" * 55)

카테고리별_성적 = {}

with open(답변_파일) as f:
    for 줄 in f:
        데이터 = json.loads(줄)
        키 = f"{데이터['platform']}_{데이터['data_type']}"
        if 키 not in 카테고리별_성적:
            카테고리별_성적[키] = {"맞춤": 0, "전체": 0}

        카테고리별_성적[키]["전체"] += 1

        출력 = 데이터.get("output", "")
        if not 출력:
            continue

        # bbox 안에 좌표가 있는지 확인
        try:
            x, y = map(int, 출력.strip("()").split(", "))
            bx, by, bw, bh = 데이터["bbox"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                카테고리별_성적[키]["맞춤"] += 1
        except:
            pass

전체_정확도_합 = 0
for 키, 성적 in sorted(카테고리별_성적.items()):
    정확도 = 성적["맞춤"] / 성적["전체"] if 성적["전체"] > 0 else 0
    전체_정확도_합 += 정확도
    print(f"  {키:20s}: {정확도:.1%}  ({성적['맞춤']}/{성적['전체']})")

평균_정확도 = 전체_정확도_합 / len(카테고리별_성적)
print(f"\n  {'평균 정확도':20s}: {평균_정확도:.2%}")
print("\n[완료] ScreenSpot 벤치마크 평가 종료!")
