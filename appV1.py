import os
import time
import json
import re
import uuid
from datetime import datetime
import cv2
import requests
import openai
import numpy as np

# 기본 경로 설정
BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, "asset", "images", "input")
OUTPUT_JSON_DIR = os.path.join(BASE_DIR, "asset", "json")

# 각 디렉토리가 존재하지 않을 경우 생성
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

PROCESSED_FILES = set(os.listdir(INPUT_DIR))

# API 키 파일 경로 설정 및 로드 함수
def load_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

CLOVA_API_URL = load_api_key("/Users/apple/Desktop/Python/2nd_Grade/Competition/TEAM-CLICK/lib/CLOVA_API_URL.txt")
CLOVA_SECRET_KEY = load_api_key("/Users/apple/Desktop/Python/2nd_Grade/Competition/TEAM-CLICK/lib/CLOVA_SECRET_KEY.txt")
openai.api_key = load_api_key("/Users/apple/Desktop/Python/2nd_Grade/Competition/TEAM-CLICK/lib/openai_api_key.txt")

def process_image(image_path):
    """이미지 전처리 후 에지 강조 처리"""
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=1, threshold2=50)
    gray_image[edges == 255] = 0
    return np.where(gray_image == 0, 255, 0).astype(np.uint8)

def extract_text_with_clova(image_data):
    """클로바 OCR을 사용하여 텍스트 추출"""
    files = [('file', ('image.jpg', image_data, 'application/octet-stream'))]
    request_json = {
        'images': [{'format': 'jpg', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }
    headers = {'X-OCR-SECRET': CLOVA_SECRET_KEY}
    response = requests.post(CLOVA_API_URL, headers=headers, data={'message': json.dumps(request_json)}, files=files)
    
    if response.status_code == 200:
        ocr_result = response.json()
        fields = ocr_result.get("images", [])[0].get("fields", [])
        return " ".join(field["inferText"] for field in fields)
    else:
        print(f"[{datetime.now()}] OCR extraction failed with status code: {response.status_code}")
        return None

def analyze_schedule_with_openai(ocr_text, output_json_path, student_name, student_id):
    """GPT-4 API를 사용하여 시간표 분석 및 JSON 저장"""
    prompt = f"""
    Analyze the following OCR data and organize it as a class schedule in JSON with fields: 'class_name', 'class_days', 'start_time', 'end_time', and 'location'.
    OCR data: {ocr_text}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in OCR data processing and JSON structuring."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.2,
    )
    response_text = response['choices'][0]['message']['content'].strip()
    schedule_data = json.loads(re.search(r'\{.*\}|\[.*\]', response_text, re.DOTALL).group(0))
    
    final_data = {"info": {"name": student_name, "number": student_id}, "schedule": schedule_data}
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(final_data, json_file, ensure_ascii=False, indent=4)
    print(f"[{datetime.now()}] Schedule JSON saved: {output_json_path}")

def monitor_and_process_images():
    """새로운 이미지를 감지하고 처리"""
    while True:
        current_files = set(os.listdir(INPUT_DIR))
        new_files = current_files - PROCESSED_FILES

        for filename in new_files:
            match = re.match(r"^(\d{8})_([\w가-힣]+)\.(jpg|png|jpeg)$", filename)
            if not match:
                print(f"[{datetime.now()}] Ignored file: {filename} (does not match '학번_이름.확장자' format)")
                continue

            student_id, student_name = match.groups()
            print(f"[{datetime.now()}] New file detected: {filename} (Student ID: {student_id}, Name: {student_name})")
            file_path = os.path.join(INPUT_DIR, filename)

            processed_image = process_image(file_path)
            ocr_text = extract_text_with_clova(cv2.imencode('.jpg', processed_image)[1].tobytes())
            if ocr_text:
                output_json_path = os.path.join(OUTPUT_JSON_DIR, filename.replace(".png", ".json").replace(".jpg", ".json"))
                analyze_schedule_with_openai(ocr_text, output_json_path, student_name, student_id)

            PROCESSED_FILES.add(filename)
        
        # time.sleep(5) # 주석처리된 대기 시간

if __name__ == "__main__":
    monitor_and_process_images()
