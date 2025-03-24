from flask import Flask, request, jsonify
from pymongo import MongoClient  # MongoDB 연결을 위해 추가
import os

# Flask 앱 생성
app = Flask(__name__)

# MongoDB 클라이언트 설정
client = MongoClient("mongodb+srv://kylhs0705:smtI18Nl4WqtRUXX@team-click.s8hg5.mongodb.net/?retryWrites=true&w=majority&appName=Team-Click")
db = client.OurTime  # OurTime 데이터베이스
collection = db.User  # User 컬렉션

@app.route('/login', methods=['POST'])
def login():
    """
    로그인 요청을 처리하는 API.
    사용자로부터 받은 ID와 PW를 MongoDB에서 확인하고 상태를 반환합니다.
    """
    try:
        data = request.json  # 클라이언트로부터 JSON 데이터를 받음
        user_id = data.get("id")
        user_pw = data.get("pw")

        # ID 확인
        user = collection.find_one({"_id": user_id})

        if not user:
            return jsonify({"status": "error", "message": "아이디가 올바르지 않습니다."})
        elif user["info"]["pw"] != user_pw:
            return jsonify({"status": "error", "message": "비밀번호를 확인해주세요."})
        else:
            return jsonify({"status": "success", "message": f"로그인 성공. 환영합니다, {user['info']['name']}님!"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"서버 오류 발생: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
