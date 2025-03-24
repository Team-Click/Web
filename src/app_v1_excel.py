import pandas as pd
import json
from rapidfuzz import process, fuzz

# 1. JSON 파일 불러오기
input_json_path = "Web/asset/json/22011925_권효정.json"  # 불러올 JSON 파일 경로
with open(input_json_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 2. 엑셀 파일 로드
excel_path = "Web/2024-2시간표.xlsx"  # 엑셀 파일 경로
df = pd.read_excel(excel_path)

# 3. 날짜 데이터를 숫자로 변환하는 함수
def convert_days_to_numbers(days):
    """월화수목금을 1, 2, 3, 4, 5로 변환"""
    day_mapping = {"월": 1, "화": 2, "수": 3, "목": 4, "금": 5}
    return [day_mapping[day] for day in days if day in day_mapping]

# 4. 텍스트 및 시간 비교 함수 정의
def find_best_match(ocr_name, ocr_start_time, ocr_end_time, df):
    """OCR 이름, 시작 시간, 종료 시간을 엑셀 데이터와 비교 후 가장 유사한 데이터 반환"""
    best_match = None
    best_score = 0

    for idx, row in df.iterrows():
        # 과목 이름 비교 (가중치 50%)
        name_score = fuzz.token_sort_ratio(ocr_name, row["class_name"]) * 0.5

        # 시작 시간 비교 (가중치 25%)
        start_time_score = (
            fuzz.ratio(str(ocr_start_time), str(row["class_start_time"])) * 0.25
            if not pd.isnull(row["class_start_time"])
            else 0
        )

        # 종료 시간 비교 (가중치 25%)
        end_time_score = (
            fuzz.ratio(str(ocr_end_time), str(row["class_end_time"])) * 0.25
            if not pd.isnull(row["class_end_time"])
            else 0
        )

        # 총 점수 계산
        total_score = name_score + start_time_score + end_time_score

        # 가장 높은 점수를 가진 매칭 데이터 저장
        if total_score > best_score:
            best_match = row.to_dict()
            best_score = total_score

    # 반환: 가장 유사한 데이터와 그 유사도 점수
    return best_match, best_score

# 5. JSON 데이터 업데이트 함수
def update_json_with_excel_data(json_data, df):
    schedule = json_data["schedule"]
    for item in schedule:
        ocr_name = item["class_name"]
        ocr_start_time = item.get("class_start_time", "")  # JSON에서 시작 시간
        ocr_end_time = item.get("class_end_time", "")  # JSON에서 종료 시간

        matched_row, score = find_best_match(ocr_name, ocr_start_time, ocr_end_time, df)
        item["originalOCR"] = ocr_name  # OCR로 추출된 원래 이름 저장
        
        # 유사도가 80% 이상일 경우만 데이터를 업데이트
        if matched_row and score >= 80:
            item["class_name"] = matched_row["class_name"]  # 가장 유사한 과목 이름으로 변경
            # 날짜 변환: 월화수목금을 숫자로
            item["class_days"] = convert_days_to_numbers(str(matched_row["class_days"]))
            item["class_start_time"] = matched_row["class_start_time"]  # 강의 시작 시간 추가
            item["class_end_time"] = matched_row["class_end_time"]  # 강의 종료 시간 추가
            item["matchScore"] = score  # 유사도 저장
        else:
            # 유사도가 80% 미만일 경우 매칭하지 않고 점수만 기록
            item["matchScore"] = score
    return json_data

# 6. JSON 데이터 업데이트 수행
updated_json_data = update_json_with_excel_data(json_data, df)

# 7. 업데이트된 JSON 데이터 저장
output_json_path = "Web/asset/json/updated_schedule.json"
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(updated_json_data, f, ensure_ascii=False, indent=4)

print(f"업데이트된 JSON 파일 저장 완료: {output_json_path}")
