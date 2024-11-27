from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import os
import jwt  # PyJWT
from datetime import datetime, timedelta
from PIL import Image, UnidentifiedImageError  # 이미지 열기 용도
import sys

# Flask 설정
app = Flask(__name__)
CORS(app)

# JWT 시크릿 키 설정
SECRET_KEY = "your_secure_secret_key"

# MongoDB 설정
client = MongoClient(
    "mongodb+srv://kylhs0705:smtI18Nl4WqtRUXX@team-click.s8hg5.mongodb.net/?retryWrites=true&w=majority&appName=Team-Click",
    tls=True,
    tlsAllowInvalidCertificates=True
)
db = client.OurTime
collection = db.User

# 이미지 저장 경로
UPLOAD_FOLDER = "/Users/apple/Desktop/Python/Github/Flask/asset/image"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 허용된 확장자
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        user_id = data.get("id")
        user_pw = data.get("pw")

        # 사용자 인증
        user = collection.find_one({"_id": user_id})
        if not user:
            return jsonify({"status": "error", "message": "아이디가 올바르지 않습니다."})
        elif user["info"]["pw"] != user_pw:
            return jsonify({"status": "error", "message": "비밀번호를 확인해주세요."})

        # JWT 생성 (유효시간: 30분)
        expiration_time = datetime.utcnow() + timedelta(minutes=30)
        payload = {"id": user_id, "exp": expiration_time}
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

        return jsonify({"status": "success", "message": "로그인 성공", "token": token})
    except Exception as e:
        return jsonify({"status": "error", "message": f"서버 오류 발생: {str(e)}"})

@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"status": "error", "message": "토큰이 필요합니다."})

        # 토큰 검증
        try:
            jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"status": "error", "message": "토큰이 만료되었습니다."})
        except jwt.InvalidTokenError:
            return jsonify({"status": "error", "message": "유효하지 않은 토큰입니다."})

        # ID 및 파일 처리
        user_id = request.form.get("id")  # HTML에서 전달된 ID
        if not user_id:
            return jsonify({"status": "error", "message": "ID가 필요합니다."})

        if "image" not in request.files:
            return jsonify({"status": "error", "message": "이미지가 첨부되지 않았습니다."})

        file = request.files["image"]

        # 확장자 확인 및 대소문자 처리
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"status": "error", "message": "지원하지 않는 파일 형식입니다."})

        # 파일 저장 (파일 이름은 ID로 설정)
        filename = f"{secure_filename(user_id)}{ext}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # 디버깅: 파일 경로 확인
        print(f"이미지 저장 경로: {file_path}")

        # 이미지 열기
        try:
            img = Image.open(file_path)
            img.show()
        except UnidentifiedImageError:
            return jsonify({"status": "error", "message": "올바른 이미지 파일이 아닙니다."})

        return jsonify({"status": "success", "message": "이미지 업로드 성공", "image_path": file_path})

    except Exception as e:
        return jsonify({"status": "error", "message": f"서버 오류 발생: {str(e)}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
