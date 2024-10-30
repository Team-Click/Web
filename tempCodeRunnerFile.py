from flask import Flask, render_template, request, redirect, url_for, send_file
import os

app = Flask(__name__)

# 기본 경로 설정
BASE_DIR = "/Users/apple/Desktop/Python/2nd_Grade/Competition/TEAM-CLICK"
UPLOAD_DIR = os.path.join(BASE_DIR, "asset/images/input")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 허용된 파일 형식 확인 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 루트 페이지 - 사용자 정보 입력 및 중복 검사
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        student_id = request.form['studentId'].strip()
        student_name = request.form['studentName'].strip()

        # 입력 확인용 로그 출력
        print(f"Received student ID: {student_id}")
        print(f"Received student name: {student_name}")

        # 파일명: "학번_이름.확장자" 형식으로 설정하고 중복 검사
        for ext in ALLOWED_EXTENSIONS:
            file_path = os.path.join(UPLOAD_DIR, f"{student_id}_{student_name}.{ext}")
            if os.path.exists(file_path):
                return render_template('index.html', error="중복된 입력입니다. 학번과 이름을 다시 입력해주세요.")

        # 중복이 없으면 업로드 페이지로 이동
        return redirect(url_for('upload_image', student_id=student_id, student_name=student_name))
    
    return render_template('index.html')

# 이미지 업로드 처리
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        student_id = request.form.get('studentId')
        student_name = request.form.get('studentName')

        # 입력 확인용 로그 출력
        print(f"Uploading for student ID: {student_id}")
        print(f"Uploading for student name: {student_name}")

        if 'file' not in request.files:
            return render_template('upload.html', error="파일이 업로드되지 않았습니다.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error="선택된 파일이 없습니다.")
        
        if file and allowed_file(file.filename):
            # 파일명: "학번_이름.확장자" 형식으로 설정
            extension = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{student_id}_{student_name}.{extension}"  # 학번과 이름 사이에 '_' 추가
            file.save(os.path.join(UPLOAD_DIR, filename))

            # 업로드가 완료되면 성공 메시지
            return render_template('upload.html', success=True)
        
        return render_template('upload.html', error="허용되지 않는 파일 형식입니다. PNG 또는 JPG 파일만 가능합니다.")
    
    return render_template('upload.html')

# 파일 다운로드 경로
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(UPLOAD_DIR, filename), as_attachment=True)

if __name__ == "__main__":
    app.run(port=5001, debug=True)
