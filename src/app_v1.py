import os
import time
import json
import uuid
import re
from datetime import datetime
import numpy as np
import cv2
from PIL import Image  # Pillow 사용
import requests
import openai
import pandas as pd
from rapidfuzz import process, fuzz

# 기본 경로 설정을 실행 파일의 위치를 기준으로 상대 경로로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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


def load_api_key(file_path):
    """API 키 파일에서 정보를 로드"""
    with open(file_path, 'r') as file:
        return file.read().strip()


# API 키 파일 경로 설정
CLOVA_API_URL_PATH = os.path.join(BASE_DIR, "lib", "CLOVA_API_URL.txt")
CLOVA_SECRET_KEY_PATH = os.path.join(BASE_DIR, "lib", "CLOVA_SECRET_KEY.txt")
OPENAI_API_KEY_PATH = os.path.join(BASE_DIR, "lib", "openai_api_key.txt")

# API 키 파일에서 정보를 로드
CLOVA_API_URL = load_api_key(CLOVA_API_URL_PATH)
CLOVA_SECRET_KEY = load_api_key(CLOVA_SECRET_KEY_PATH)
openai.api_key = load_api_key(OPENAI_API_KEY_PATH)

from PIL import Image

def process_image(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, file_name)

    print(f"[DEBUG] Reading image from: {image_path}")

    try:
        # Pillow로 이미지 읽기
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image = np.array(img)

        # OpenCV로 처리
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, threshold1=1, threshold2=50)
        kernel = np.ones((3, 3), np.uint8)
        thick_edges = cv2.dilate(edges, kernel, iterations=1)
        gray_image_with_edges = gray_image.copy()
        gray_image_with_edges[edges == 255] = 0
        inverted_image = np.where(gray_image_with_edges == 0, 255, 0).astype(np.uint8)
        re_inverted_image = cv2.bitwise_not(inverted_image)

        # Pillow로 이미지 저장
        try:
            Image.fromarray(re_inverted_image).save(output_path)
            print(f"[{datetime.now()}] Final processed image saved: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save processed image with Pillow: {e}")
            return None

        # 저장된 파일 확인
        if not os.path.exists(output_path):
            print(f"[ERROR] File not found after save attempt: {output_path}")
            return None

        return output_path
    except Exception as e:
        print(f"[ERROR] Failed to process image: {e}")
        return None


def extract_text_with_clova(file_path, temp_dir):
    """클로바 OCR을 사용하여 텍스트 추출"""
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
    """GPT-4 API를 사용하여 시간표 분석"""
    prompt = f"""
    Analyze the following OCR data and organize it into a class schedule in JSON format. The output should match this exact structure:
    - Class name (without spaces),
    - Class days (represented as 1 for Monday to 5 for Friday),
    - Class start and end times (in hh:mm format, rounded to the nearest 30-minute interval),
    - Location (classroom or hall).

    {{
        "class_name",
        "class_days",
        "class_start_time",
        "class_end_time",
        "location"
    }}

    Here is the OCR data: {ocr_text}
    """

    retries = 3  # API 호출 재시도 횟수
    for attempt in range(retries):
        try:
            # 최신 OpenAI ChatCompletion API 호출
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in OCR data processing and JSON structuring."},
                    {"role": "user", "content": prompt}
                ]
            )

            # 응답에서 메시지 텍스트 추출
            response_text = response['choices'][0]['message']['content'].strip()

            # JSON 형식만 추출
            json_match = re.search(r'\{.*\}|\[.*\]', response_text, re.DOTALL)
            if json_match:
                schedule_json = json_match.group(0)
                schedule_data = json.loads(schedule_json)

                # 결과 데이터에 사용자 정보 추가
                final_data = {
                    "info": {
                        "name": student_name,
                        "number": student_id
                    },
                    "schedule": schedule_data
                }

                # JSON 파일로 저장
                with open(output_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(final_data, json_file, ensure_ascii=False, indent=4)
                print(f"[{datetime.now()}] Schedule JSON saved: {output_json_path}")
                return output_json_path
            else:
                print(f"[ERROR] Unexpected response format on attempt {attempt + 1}/{retries}. Retrying...")
                time.sleep(2)

        except openai.OpenAIError as e:
            print(f"[ERROR] OpenAI API call failed: {e}. Retrying {attempt + 1}/{retries}...")
            time.sleep(2)

    print("[ERROR] Failed to retrieve a valid response from OpenAI API after retries.")
    return None





def monitor_and_process_images():
    """OCR 처리 및 JSON 업데이트를 한 루프에서 반복"""
    valid_extensions = {"png", "jpg", "jpeg"}
    processed_files = PROCESSED_FILES.copy()  # 이전에 처리된 파일 기록
    excel_path = "Web/2024-2시간표.xlsx"  # 엑셀 파일 경로
    input_json_dir = OUTPUT_JSON_DIR  # OCR 결과가 저장된 JSON 디렉토리
    output_json_dir = os.path.join(BASE_DIR, "asset", "json")  # 업데이트된 JSON 저장 디렉토리

    os.makedirs(output_json_dir, exist_ok=True)

    while True:
        # 1. 새로운 이미지 파일 감지
        current_files = set(os.listdir(INPUT_DIR))
        new_files = current_files - processed_files

        for filename in new_files:
            file_extension = filename.split(".")[-1].lower()
            if file_extension not in valid_extensions:
                print(f"[{datetime.now()}] Error: Invalid file extension for '{filename}'. Only PNG, JPG, JPEG files are allowed.")
                continue

            match = re.match(r"^(\d{8})_([\w가-힣]+)\.(png|jpg|jpeg)$", filename, re.IGNORECASE)
            if not match:
                print(f"[{datetime.now()}] Ignored file: {filename} (does not match '학번_이름.확장자' format)")
                continue

            student_id = match.group(1)
            student_name = match.group(2)
            print(f"[{datetime.now()}] New file detected: {filename} (Student ID: {student_id}, Name: {student_name})")

            # 2. 이미지 처리 및 OCR
            file_path = os.path.join(INPUT_DIR, filename)
            processed_image_path = process_image(file_path, OUTPUT_IMAGE_DIR)
            if processed_image_path is None or not os.path.exists(processed_image_path):
                print(f"[ERROR] Processed image file missing: {processed_image_path}")
                continue

            temp_json_path = extract_text_with_clova(processed_image_path, TEMP_JSON_DIR)
            if temp_json_path is None:
                continue

            with open(temp_json_path, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)
            ocr_text = extract_ocr_text(ocr_data)

            # 3. 시간표 분석 및 JSON 생성
            output_json_path = os.path.join(OUTPUT_JSON_DIR, filename.replace(".png", ".json").replace(".jpg", ".json"))
            analyze_schedule_with_openai(ocr_text, output_json_path, student_name, student_id)

            processed_files.add(filename)

        # 4. JSON 업데이트 수행
        print(f"[{datetime.now()}] Updating JSON files with Excel data...")
        process_final_json(input_json_dir, excel_path, output_json_dir)

        # 5. 루프 대기
        print(f"[{datetime.now()}] Waiting for new files...")
        time.sleep(10)  # 10초 간격으로 새로운 파일 확인




def update_json_with_excel_data(json_data, df):
    def convert_days_to_numbers(days):
        day_mapping = {"월": 1, "화": 2, "수": 3, "목": 4, "금": 5}
        return [day_mapping[day] for day in days if day in day_mapping]

    def find_best_match(ocr_name, ocr_start_time, ocr_end_time, df):
        best_match = None
        best_score = 0
        for _, row in df.iterrows():
            name_score = fuzz.token_sort_ratio(ocr_name, row["class_name"]) * 0.5
            start_time_score = fuzz.ratio(str(ocr_start_time), str(row["class_start_time"])) * 0.25 if not pd.isnull(row["class_start_time"]) else 0
            end_time_score = fuzz.ratio(str(ocr_end_time), str(row["class_end_time"])) * 0.25 if not pd.isnull(row["class_end_time"]) else 0
            total_score = name_score + start_time_score + end_time_score
            if total_score > best_score:
                best_match = row.to_dict()
                best_score = total_score
        return best_match, best_score

    schedule = json_data["schedule"]
    for item in schedule:
        ocr_name = item["class_name"]
        ocr_start_time = item.get("class_start_time", "")
        ocr_end_time = item.get("class_end_time", "")
        matched_row, score = find_best_match(ocr_name, ocr_start_time, ocr_end_time, df)
        item["originalOCR"] = ocr_name
        if matched_row and score >= 80:
            item["class_name"] = matched_row["class_name"]
            item["class_days"] = convert_days_to_numbers(str(matched_row["class_days"]))
            item["class_start_time"] = matched_row["class_start_time"]
            item["class_end_time"] = matched_row["class_end_time"]
            item["matchScore"] = score
        else:
            item["matchScore"] = score
    return json_data

def process_final_json(input_json_dir, excel_path, output_json_dir):
    os.makedirs(output_json_dir, exist_ok=True)
    for filename in os.listdir(input_json_dir):
        if not filename.endswith(".json"):
            continue
        input_json_path = os.path.join(input_json_dir, filename)
        with open(input_json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        df = pd.read_excel(excel_path)
        updated_json_data = update_json_with_excel_data(json_data, df)

        output_json_path = os.path.join(output_json_dir, filename)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(updated_json_data, f, ensure_ascii=False, indent=4)
        print(f"[INFO] Updated JSON saved: {output_json_path}")




if __name__ == "__main__":
    monitor_and_process_images()
