import web_connect
import Textneck_Model  # 첫 번째 파일을 import
from multiprocessing import Value, Process
from flask import Flask, request, jsonify
import requests
import datetime
from datetime import datetime


app = Flask(__name__)

# 사용자 이름 저장 변수
current_username = ""
turtle_count = 0
spine_count = 18
current_date = datetime.now()  # 현재 날짜 가져오기 (수정 해야함)

# turtle_count


def get_turtle_count(counter):
    while True:
        turtle_count = counter.value
        # turtle_count 값을 사용해 필요한 작업 수행
        # 예를 들어, turtle_count가 특정 값에 도달하면 break
        print("Cnt:", turtle_count)

        if turtle_count >= 15:  # 예시 조건
            break


if __name__ == "__main__":
    counter = Value('i', 0)  # shared counter (주석 추가)

    p1 = Process(target=Textneck_Model.webcam_processing, args=(counter,))
    p2 = Process(target=get_turtle_count, args=(counter,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()


@app.route('/send-username', methods=['POST'])
def send_username():
    global current_username
    data = request.json
    current_username = data.get('username', "")
    print(current_username)
    return 'OK'


# Node.js 서버의 주소
NODE_SERVER_URL = "http://posturetech.ap-northeast-2.elasticbeanstalk.com/"

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
