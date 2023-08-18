from flask import Flask, request, jsonify
import requests
import datetime

app = Flask(__name__)

# 사용자 이름 저장 변수
current_username = ""
turtle_count = 22
spine_count = 18
current_date = '2023-08-18'  # 현재 날짜 가져오기 (수정 해야함)

@app.route('/send-username', methods=['POST'])
def send_username():
    global current_username
    data = request.json
    current_username = data.get('username', "")
    print(current_username)
    return 'OK'

# Node.js 서버의 주소
NODE_SERVER_URL = "http://localhost:5500"

# 버튼을 누를 때 실행될 함수
@app.route('/send-data', methods=['POST'])
def send_data():
    global current_username  # 사용자 이름을 함수 내에서 사용하기 위해 global 키워드로 선언
    
    data = {
        "username": current_username,
        "turtleCount": turtle_count,
        "spineCount": spine_count,
        "date": current_date
    }

    try:
        # Node.js 서버로 데이터 전송
        response = requests.post(f"{NODE_SERVER_URL}/save-data", json=data)
        response_data = response.json()
        print('전송 성공')
        return jsonify({"message": "Data sent successfully."})
    except Exception as e:
        print('Error sending data to Node.js server:', e)
        return jsonify({"error": "Error sending data to Node.js server."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
