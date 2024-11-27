import os
import time
import json
import uuid
import re
from datetime import datetime, timedelta
import numpy as np
import cv2
import requests
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import jwt

# Flask 설정
app = Flask(__name__)
CORS(app)

# JWT 시크릿 키
SECRET_KEY = "your_secure_secret_key"

# MongoDB 클라이언트 설정
client = MongoClient("mongodb+srv://kylhs0705:smtI18Nl4WqtRUXX@team-click.s8hg5.mongodb.net/?retryWrites=true&w=majority&appName=Team-Click")
db = client.OurTime
collection = db.timetable

# 상대 경로 설정
BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, "asset", "images", "input")
OUTPUT_IMAGE_DIR = os.path.join(BASE_DIR, "asset", "images", "output")
TEMP_JSON_DIR = os.path.join(BASE_DIR, "asset", "temp")
OUTPUT_JSON_DIR = os.path.join(BASE_DIR, "asset", "json")

# 디렉토리 생성
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(TEMP_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# API 키 파일 상대 경로 설정
CLOVA_API_URL_PATH = os.path.join(current_dir, "key", "CLOVA_API_URL.txt")
CLOVA_SECRET_KEY_PATH = os.path.join(current_dir, "key", "CLOVA_SECRET_KEY.txt")
OPENAI_API_KEY_PATH = os.path.join(current_dir, "key", "openai_api_key.txt")

# API 키 파일 로드 함수
def load_api_key(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()

# API 키 로드
CLOVA_API_URL = load_api_key(CLOVA_API_URL_PATH)
CLOVA_SECRET_KEY = load_api_key(CLOVA_SECRET_KEY_PATH)
openai.api_key = load_api_key(OPENAI_API_KEY_PATH)

# 로그인 API
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        user_id = data.get("id")
        user_pw = data.get("pw")

        # 간단한 로그인 검증 (DB 연동 가능)
        if user_id == "21011898" and user_pw == "password":
            expiration_time = datetime.utcnow() + timedelta(minutes=30)
            token = jwt.encode({"id": user_id, "exp": expiration_time}, SECRET_KEY, algorithm="HS256")
            return jsonify({"status": "success", "token": token})
        else:
            return jsonify({"status": "error", "message": "아이디 또는 비밀번호가 올바르지 않습니다."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"서버 오류 발생: {str(e)}"})

# 이미지 업로드 및 시간표 분석 API
@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        # 토큰 검증
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"status": "error", "message": "로그인이 필요합니다."})
        try:
            decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            user_id = decoded_token["id"]
        except jwt.ExpiredSignatureError:
            return jsonify({"status": "error", "message": "토큰이 만료되었습니다."})
        except jwt.InvalidTokenError:
            return jsonify({"status": "error", "message": "유효하지 않은 토큰입니다."})

        # 파일 처리
        if "image" not in request.files:
            return jsonify({"status": "error", "message": "이미지 파일이 필요합니다."})
        file = request.files["image"]
        filename = secure_filename(f"{user_id}.png")  # ID를 파일 이름으로 사용
        file_path = os.path.join(INPUT_DIR, filename)
        file.save(file_path)

        # 이미지 전처리
        processed_image_path = process_image(file_path, OUTPUT_IMAGE_DIR)

        # 클로바 OCR 처리
        temp_json_path = extract_text_with_clova(processed_image_path, TEMP_JSON_DIR)
        if not temp_json_path:
            return jsonify({"status": "error", "message": "OCR 처리 중 오류 발생."})

        # OCR 데이터 로드
        with open(temp_json_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
        ocr_text = extract_ocr_text(ocr_data)

        # OpenAI API로 시간표 분석
        output_json_path = os.path.join(OUTPUT_JSON_DIR, f"{user_id}.json")
        final_data = analyze_schedule_with_openai(ocr_text, output_json_path, student_name=user_id, student_id=user_id)

        # MongoDB에 저장
        if final_data:
            collection.replace_one({"_id": user_id}, final_data, upsert=True)  # upsert로 데이터 저장

        return jsonify({"status": "success", "message": "이미지 업로드 및 분석 완료.", "data": final_data})
    except Exception as e:
        return jsonify({"status": "error", "message": f"서버 오류 발생: {str(e)}"})

# 이미지 전처리 함수
def process_image(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, file_name)

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=1, threshold2=50)
    kernel = np.ones((3, 3), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)
    gray_image_with_edges = gray_image.copy()
    gray_image_with_edges[edges == 255] = 0
    inverted_image = np.where(gray_image_with_edges == 0, 255, 0).astype(np.uint8)
    re_inverted_image = cv2.bitwise_not(inverted_image)
    cv2.imwrite(output_path, re_inverted_image)
    return output_path

# 클로바 OCR 처리
def extract_text_with_clova(file_path, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    files = [('file', open(file_path, 'rb'))]
    headers = {'X-OCR-SECRET': CLOVA_SECRET_KEY}
    response = requests.post(CLOVA_API_URL, headers=headers, files=files)
    if response.status_code == 200:
        temp_json_path = os.path.join(temp_dir, f"{os.path.basename(file_path)}.json")
        with open(temp_json_path, "w", encoding="utf-8") as f:
            json.dump(response.json(), f, ensure_ascii=False, indent=4)
        return temp_json_path
    return None

# OpenAI API를 사용한 시간표 분석
def analyze_schedule_with_openai(ocr_text, output_json_path, student_name, student_id):
    prompt = f"""
    Analyze the following OCR data and organize it as a class schedule. Include fields:
    'class_name', 'class_days', 'start_time', 'end_time', and 'location'.
    Data: {ocr_text}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in OCR data analysis."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.2,
        )
        schedule_json = json.loads(response['choices'][0]['message']['content'])
        final_data = {
            "_id": student_id,
            "info": {"name": student_name, "number": student_id},
            "schedule": schedule_json,
        }
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        return final_data
    except Exception as e:
        print(f"OpenAI 분석 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
