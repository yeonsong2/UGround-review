"""
UGround ScreenSpot 벤치마크 평가 스크립트 (transformers 기반)
"""
import torch, json, os
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

MODEL_PATH = "/nas/home/lys0426/UGround/models/UGround-V1-7B"
QUESTION_FILE = "/nas/home/lys0426/UGround/screenspot/questions.jsonl"
IMAGE_DIR = "/nas/home/lys0426/UGround/screenspot/images"
ANSWERS_FILE = "/nas/home/lys0426/UGround/screenspot/answers.jsonl"

print("모델 로딩 중...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("로딩 완료!\n")

with open(QUESTION_FILE) as f:
    questions = [json.loads(l) for l in f]

if os.path.exists(ANSWERS_FILE):
    os.remove(ANSWERS_FILE)

BATCH_SIZE = 4

def run_batch(batch):
    all_messages = []
    for q in batch:
        img_path = os.path.join(IMAGE_DIR, q["img_filename"])
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": f"Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.\nDescription: {q['description']}\nAnswer:"},
            ],
        }]
        all_messages.append(messages)

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in all_messages]
    image_inputs = [process_vision_info(m)[0] for m in all_messages]

    results = []
    for i, (text, imgs, q) in enumerate(zip(texts, image_inputs, batch)):
        inputs = processor(text=[text], images=imgs, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, temperature=None, do_sample=False)
        generated = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

        img_path = os.path.join(IMAGE_DIR, q["img_filename"])
        with Image.open(img_path) as img:
            W, H = img.size

        try:
            rx, ry = eval(generated)
            x_abs = int(rx / 1000 * W)
            y_abs = int(ry / 1000 * H)
            output = f"({x_abs}, {y_abs})"
        except:
            output = ""

        result = {**q, "output": output, "scale": 1.0}
        results.append(result)
    return results

with open(ANSWERS_FILE, "w") as f:
    for i in tqdm(range(0, len(questions), BATCH_SIZE), desc="추론 중"):
        batch = questions[i:i+BATCH_SIZE]
        for r in run_batch(batch):
            f.write(json.dumps(r) + "\n")

print(f"\n완료! 결과 저장: {ANSWERS_FILE}")
