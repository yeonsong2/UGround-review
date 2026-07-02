"""
UGround 데모: transformers만 사용한 GUI grounding 추론
입력: 화면 이미지 + 자연어 설명
출력: 해당 UI 요소의 픽셀 좌표 (x, y)
"""
import torch
from PIL import Image, ImageDraw
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/nas/home/lys0426/UGround/models/UGround-V1-7B"
IMAGE_PATH = "/nas/home/lys0426/UGround/demo/images/sample_ui.png"

QUERIES = [
    "the search button",
    "the login button",
    "the like button",
    "the search input field",
]

print("모델 로딩 중...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("모델 로딩 완료!\n")

image = Image.open(IMAGE_PATH)
W, H = image.size
print(f"이미지 크기: {W} x {H}\n")

results = []
for description in QUERIES:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_PATH},
                {"type": "text", "text": f"""Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.
Description: {description}
Answer:"""},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32, temperature=None, do_sample=False)

    generated = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    try:
        ratio_coords = eval(generated)
        x_ratio, y_ratio = ratio_coords
        x_abs = int(x_ratio / 1000 * W)
        y_abs = int(y_ratio / 1000 * H)
        print(f'질문: "{description}"')
        print(f'  모델 출력: {generated}  →  실제 좌표: ({x_abs}, {y_abs})')
        results.append((description, x_abs, y_abs))
    except Exception as e:
        print(f'질문: "{description}"')
        print(f'  모델 출력: {generated}  [파싱 실패: {e}]')

# 결과를 이미지에 시각화
draw = ImageDraw.Draw(image)
colors = ["red", "blue", "green", "orange"]
for i, (desc, x, y) in enumerate(results):
    r = 15
    draw.ellipse([x-r, y-r, x+r, y+r], outline=colors[i % len(colors)], width=4)
    draw.text((x+r+5, y-10), desc[:15], fill=colors[i % len(colors)])

output_path = "/nas/home/lys0426/UGround/demo/result_visualized.png"
image.save(output_path)
print(f"\n결과 이미지 저장: {output_path}")
