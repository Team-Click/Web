import os
import time
import json
import uuid
import re
from datetime import datetime
import numpy as np
import cv2
import requests
import openai

# 기본 경로 설정을 실행 파일의 위치를 기준으로 상대 경로로 설정
BASE_DIR = os.path.dirname(__file__)  # 현재 파일이 있는 디렉토리(Web/)를 기준으로 함

INPUT_DIR = os.path.join(BASE_DIR, "asset", "images", "input")
OUTPUT_IMAGE_DIR = os.path.join(BASE_DIR, "asset", "images", "output")
TEMP_JSON_DIR = os.path.join(BASE_DIR, "asset", "temp")
OUTPUT_JSON_DIR = os.path.join(BASE_DIR, "asset", "json")

# 각 디렉토리가 존재하지 않을 경우 생성
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(TEMP_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

PROCESSED_FILES = set(os.listdir(INPUT_DIR))  # 초기 실행 시 input 폴더에 있는 파일을 기록하여 무시

# 파일에서 API 정보를 로드하는 함수
def load_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

# API 키 파일 경로 설정
CLOVA_API_URL_PATH = "/Users/apple/Desktop/Python/2nd_Grade/Competition/TEAM-CLICK/lib/CLOVA_API_URL.txt"
CLOVA_SECRET_KEY_PATH = "//Users/apple/Desktop/Python/2nd_Grade/Competition/TEAM-CLICK/lib/CLOVA_SECRET_KEY.txt"
OPENAI_API_KEY_PATH = "/Users/apple/Desktop/Python/2nd_Grade/Competition/TEAM-CLICK/lib/openai_api_key.txt"

# API 키 파일에서 정보를 로드
CLOVA_API_URL = load_api_key(CLOVA_API_URL_PATH)
CLOVA_SECRET_KEY = load_api_key(CLOVA_SECRET_KEY_PATH)
openai.api_key = load_api_key(OPENAI_API_KEY_PATH)

def process_image(image_path, output_dir):
    """1_Image preprocessing.py에서 가져온 이미지 전처리 함수"""
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, file_name)

    # Step 1: 이미지 로드 및 흑백 변환
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: 흑백 이미지에서 에지 강조 처리
    edges = cv2.Canny(gray_image, threshold1=1, threshold2=50)
    kernel = np.ones((3, 3), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)
    gray_image_with_edges = gray_image.copy()
    gray_image_with_edges[edges == 255] = 0  # 에지 부분을 검정색으로 강조

    # Step 3: 검정색을 흰색으로 반전
    inverted_image = np.where(gray_image_with_edges == 0, 255, 0).astype(np.uint8)

    # Step 4: 이미지 재반전 (원래 색상 복구)
    re_inverted_image = cv2.bitwise_not(inverted_image)

    # 최종 저장 경로를 설정
    output_path = os.path.join(OUTPUT_IMAGE_DIR, os.path.basename(image_path))
    cv2.imwrite(output_path, re_inverted_image)
    print(f"[{datetime.now()}] Final processed image saved: {output_path}")
    return output_path

def extract_text_with_clova(file_path, temp_dir):
    """클로바 OCR을 사용하여 텍스트 추출 및 JSON 저장"""
    os.makedirs(temp_dir, exist_ok=True)
    file_name = os.path.basename(file_path)
    temp_json_path = os.path.join(temp_dir, file_name.replace(".png", ".json").replace(".jpg", ".json"))

    files = [('file', open(file_path, 'rb'))]
    request_json = {
        'images': [{'format': 'jpg', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }
    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    headers = {'X-OCR-SECRET': CLOVA_SECRET_KEY}

    response = requests.post(CLOVA_API_URL, headers=headers, data=payload, files=files)

    if response.status_code == 200:
        ocr_result = response.json()
        with open(temp_json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_result, f, ensure_ascii=False, indent=4)
        print(f"[{datetime.now()}] OCR data saved: {temp_json_path}")
        return temp_json_path
    else:
        print(f"[{datetime.now()}] OCR extraction failed with status code: {response.status_code}")
        return None

def extract_ocr_text(ocr_data):
    """OCR 데이터에서 주요 텍스트 추출"""
    fields = ocr_data.get("images", [])[0].get("fields", [])
    text_data = " ".join(field["inferText"] for field in fields)
    return text_data

def analyze_schedule_with_openai(ocr_text, output_json_path, student_name, student_id):
    """GPT-4 API를 사용하여 시간표 분석 및 JSON 저장"""
    prompt = f"""
    Analyze the following OCR data and organize it as a class schedule. Each schedule entry should contain:
    - Class name (without spaces),
    - Class days (represented as 1 for Monday to 5 for Friday),
    - Class start and end times (rounded to the nearest 30-minute interval),
    - Location (classroom or hall).

    Here is the OCR data: {ocr_text}
    
    Please respond in JSON format only, including the fields 'class_name', 'class_days', 'start_time', 'end_time', and 'location'.
    """
    
    retries = 3
    for attempt in range(retries):
        try:
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
            # JSON 부분만 추출
            json_match = re.search(r'\{.*\}|\[.*\]', response_text, re.DOTALL)
            if json_match:
                schedule_json = json_match.group(0)
                schedule_data = json.loads(schedule_json)
                
                # 사용자 정보 추가
                final_data = {
                    "info": {
                        "name": student_name,
                        "number": student_id
                    },
                    "schedule": schedule_data  # GPT-4 응답 내용
                }
                
                # 최종 JSON 저장 (포맷팅 포함)
                with open(output_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(final_data, json_file, ensure_ascii=False, indent=4)  # indent=4로 포맷팅
                    json_file.flush()  # 즉각적 저장 확인
                print(f"[{datetime.now()}] Schedule JSON saved: {output_json_path}")
                return output_json_path
            else:
                print(f"Unexpected response format on attempt {attempt + 1}/{retries}. Retrying...")
                time.sleep(2)

        except openai.error.APIError as e:
            print(f"APIError: {e}. Retrying {attempt + 1}/{retries}...")
            time.sleep(2)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}. Ensure the response format is JSON.")

    print("Failed to retrieve a valid response from OpenAI API after retries.")
    return None



def monitor_and_process_images():
    """지정된 폴더에서 새로운 이미지를 감지하고 처리"""
    while True:
        current_files = set(os.listdir(INPUT_DIR))
        new_files = current_files - PROCESSED_FILES

        for filename in new_files:
            # "학번_이름.확장자" 형식 검사 (8자리 학번만 허용)
            match = re.match(r"^(\d{8})_([\w가-힣]+)\.(jpg|png|jpeg)$", filename)
            if not match:
                print(f"[{datetime.now()}] Ignored file: {filename} (does not match '학번_이름.확장자' format)")
                continue

            student_id = match.group(1)
            student_name = match.group(2)
            print(f"[{datetime.now()}] New file detected: {filename} (Student ID: {student_id}, Name: {student_name})")

            file_path = os.path.join(INPUT_DIR, filename)

            # 전처리 단계
            processed_image_path = process_image(file_path, OUTPUT_IMAGE_DIR)
            if processed_image_path is None:
                continue

            # OCR 처리 및 임시 JSON 저장
            temp_json_path = extract_text_with_clova(processed_image_path, TEMP_JSON_DIR)
            if temp_json_path is None:
                continue

            # 시간표 추출 단계 (OpenAI API 사용)
            with open(temp_json_path, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)
            ocr_text = extract_ocr_text(ocr_data)
            output_json_path = os.path.join(OUTPUT_JSON_DIR, filename.replace(".png", ".json").replace(".jpg", ".json"))
            analyze_schedule_with_openai(ocr_text, output_json_path, student_name, student_id)

            # 처리된 파일 추가
            PROCESSED_FILES.add(filename)

        # # 5초 대기
        # time.sleep(5)


if __name__ == "__main__":
    monitor_and_process_images()
