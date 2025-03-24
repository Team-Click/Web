import os
import time
import json
import uuid
import numpy as np
import cv2
import requests
import openai
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import jwt
from werkzeug.utils import secure_filename
import re

# Flask 설정
app = Flask(__name__)
CORS(app)

# JWT 시크릿 키
SECRET_KEY = "your_secure_secret_key"

# MongoDB 클라이언트 설정
client = MongoClient(
    "mongodb+srv://kylhs0705:smtI18Nl4WqtRUXX@team-click.s8hg5.mongodb.net/?retryWrites=true&w=majority&appName=Team-Click",
    tls=True,
    tlsAllowInvalidCertificates=True
)
db = client.OurTime
collection_timetable = db.timetable
collection_user = db.User

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

# API 키 파일 로드 함수
def load_api_key(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()

# API 키 로드
CLOVA_API_URL = load_api_key(os.path.join(BASE_DIR, "key", "CLOVA_API_URL.txt"))
CLOVA_SECRET_KEY = load_api_key(os.path.join(BASE_DIR, "key", "CLOVA_SECRET_KEY.txt"))
openai.api_key = load_api_key(os.path.join(BASE_DIR, "key", "openai_api_key.txt"))

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        user_id = data.get("id")
        user_pw = data.get("pw")

        # 사용자 정보 확인
        user = collection_user.find_one({"_id": user_id})
        if not user:
            return jsonify({"status": "error", "message": "아이디가 올바르지 않습니다."})
        elif user["info"]["pw"] != user_pw:
            return jsonify({"status": "error", "message": "비밀번호를 확인해주세요."})

        # JWT 생성 (유효시간: 30분)
        expiration_time = datetime.utcnow() + timedelta(minutes=30)
        payload = {"id": user_id, "exp": expiration_time}
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

        return jsonify({"status": "success", "message": "로그인 성공", "token": token, "user_name": user["info"]["name"]})
    except Exception as e:
        return jsonify({"status": "error", "message": f"서버 오류 발생: {str(e)}"})


@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        # 1. 토큰 검증
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

        # 2. 사용자 정보 가져오기
        user = collection_user.find_one({"_id": user_id})
        if not user:
            return jsonify({"status": "error", "message": "사용자 정보가 존재하지 않습니다."})

        # 3. 이미지 파일 처리
        if "image" not in request.files:
            return jsonify({"status": "error", "message": "이미지 파일이 필요합니다."})
        file = request.files["image"]
        filename = secure_filename(f"{user_id}.png")  # ID를 파일 이름으로 사용
        file_path = os.path.join(INPUT_DIR, filename)
        file.save(file_path)

        # 4. 이미지 전처리
        processed_image_path = process_image(file_path, OUTPUT_IMAGE_DIR)
        print(f"[DEBUG] 전처리된 이미지 경로: {processed_image_path}")

        # 5. 클로바 OCR 처리 후 JSON 파일 저장
        temp_json_path = os.path.join(TEMP_JSON_DIR, f"{user_id}_ocr.json")
        ocr_data_path = extract_text_with_clova(processed_image_path, TEMP_JSON_DIR)
        if not ocr_data_path:
            return jsonify({"status": "error", "message": "OCR 처리 중 오류 발생."})

        print(f"[DEBUG] OCR JSON 저장 경로: {ocr_data_path}")

        # 6. OpenAI GPT API로 시간표 문서화
        ocr_text = extract_ocr_text(ocr_data_path)  # OCR 데이터에서 텍스트만 추출
        if not ocr_text:
            return jsonify({"status": "error", "message": "OCR 데이터에서 텍스트를 추출하지 못했습니다."})

        output_json_path = os.path.join(OUTPUT_JSON_DIR, f"{user_id}.json")
        final_data = analyze_schedule_with_openai(
            ocr_text,
            output_json_path,
            student_name=user["info"]["name"],
            student_id=user_id
        )

        # 7. MongoDB에 데이터 저장
        if final_data:
            collection_timetable.replace_one({"_id": user_id}, final_data, upsert=True)
            print(f"[DEBUG] MongoDB 저장 성공: ID {user_id}")

        return jsonify({"status": "success", "message": "이미지 업로드 및 분석 완료.", "data": final_data})

    except Exception as e:
        print(f"[ERROR] 서버 오류 발생: {e}")
        return jsonify({"status": "error", "message": f"서버 오류 발생: {str(e)}"})


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
    print(f"[DEBUG] 전처리 완료: {output_path}")
    return output_path


def extract_text_with_clova(file_path, temp_dir):
    try:
        os.makedirs(temp_dir, exist_ok=True)
        files = [('file', open(file_path, 'rb'))]
        headers = {'X-OCR-SECRET': CLOVA_SECRET_KEY}
        payload = {
            'message': json.dumps({
                'version': 'V2',
                'requestId': str(uuid.uuid4()),
                'timestamp': int(round(time.time() * 1000)),
                'images': [{'format': 'jpg', 'name': os.path.basename(file_path)}]
            })
        }
        response = requests.post(CLOVA_API_URL, headers=headers, data=payload, files=files)
        if response.status_code == 200:
            temp_json_path = os.path.join(temp_dir, f"{os.path.basename(file_path)}.json")
            with open(temp_json_path, "w", encoding="utf-8") as f:
                json.dump(response.json(), f, ensure_ascii=False, indent=4)
            print(f"[DEBUG] OCR JSON 저장 완료: {temp_json_path}")
            return temp_json_path
        else:
            print(f"[ERROR] 클로바 OCR API 오류: 상태 코드 {response.status_code}, 응답: {response.text}")
            return None
    except Exception as e:
        print(f"[ERROR] OCR 호출 중 오류 발생: {e}")
        return None


def extract_ocr_text(ocr_data_path):
    try:
        with open(ocr_data_path, "r", encoding="utf-8") as file:
            ocr_data = json.load(file)

        if "images" not in ocr_data:
            print("[ERROR] OCR JSON에 'images' 필드가 없습니다.")
            return ""

        extracted_texts = []
        for image in ocr_data.get("images", []):
            if "fields" in image:
                for field in image["fields"]:
                    extracted_texts.append(field.get("inferText", ""))

        combined_text = " ".join(extracted_texts)
        # print(f"[DEBUG] OCR에서 추출된 텍스트: {combined_text}")
        return combined_text
    except Exception as e:
        print(f"[ERROR] OCR 텍스트 추출 중 오류 발생: {e}")
        return ""


def analyze_schedule_with_openai(ocr_text, output_json_path, student_name, student_id):
    """GPT-4 API를 사용하여 시간표 분석 및 JSON 저장"""
    prompt = f"""
    Analyze the following OCR data and organize it as a class schedule. Each schedule entry should contain:
    - Class name (without spaces),
    - Class days (represented as 1 for Monday to 5 for Friday),
    - Class start and end times (formatted as HH:MM, 24-hour time),
    - Location (classroom or hall).

    Here is the OCR data: {ocr_text}
    
    Please respond in JSON format only, including the fields 'class_name', 'class_days', 'start_time', 'end_time', and 'location'.
    """
    
    retries = 3  # 최대 3번 재시도
    for attempt in range(retries):
        try:
            print("[DEBUG] OpenAI GPT API 호출 준비 완료.")
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
            print("[DEBUG] OpenAI 응답 수신 완료.")

            # JSON 부분만 추출
            json_match = re.search(r'\{.*\}|\[.*\]', response_text, re.DOTALL)
            if json_match:
                schedule_json = json_match.group(0)
                schedule_data = json.loads(schedule_json)

                # 사용자 정보 추가
                final_data = {
                    "_id": student_id,
                    "info": {
                        "name": student_name,
                        "number": student_id
                    },
                    "schedule": schedule_data
                }

                # 최종 JSON 저장
                with open(output_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(final_data, json_file, ensure_ascii=False, indent=4)
                print(f"[DEBUG] Schedule JSON saved: {output_json_path}")
                return final_data
            else:
                print(f"[ERROR] 응답에서 JSON 형식을 찾을 수 없습니다. 재시도 중... ({attempt + 1}/{retries})")
                time.sleep(2)

        except openai.error.APIError as e:
            print(f"[ERROR] OpenAI API 호출 오류: {e}. 재시도 중... ({attempt + 1}/{retries})")
            time.sleep(2)
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON 파싱 오류: {e}. 재시도 중... ({attempt + 1}/{retries})")
            time.sleep(2)
        except Exception as e:
            print(f"[ERROR] 알 수 없는 오류 발생: {e}. 재시도 중... ({attempt + 1}/{retries})")
            time.sleep(2)

    print("[ERROR] OpenAI API 호출에 실패했습니다.")
    return None



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
