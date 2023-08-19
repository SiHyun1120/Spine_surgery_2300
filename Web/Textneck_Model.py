# 모듈 import
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 각도 함수


def angle3(a, b, c, df):  # 세 점 사이의 각도를 구한다.
    a_x = np.array(df['x{}'.format(a)])  # First
    a_y = np.array(df['y{}'.format(a)])

    b_x = np.array(df['x{}'.format(b)])  # Mid
    b_y = np.array(df['y{}'.format(b)])

    c_x = np.array(df['x{}'.format(c)])
    c_y = np.array(df['y{}'.format(c)])  # End

    # y from endpoint - y form midpoint, x form end - x from mind
    radians = np.arctan2(c_y-b_y, c_x-b_x) - np.arctan2(a_y-b_y, a_x-b_x)
    angle = np.abs(radians*180.0/np.pi)

    for i in range(0, angle.shape[0]):
        if angle[i] > 180.0:
            angle[i] = 360-angle[i]

    return pd.Series(angle)


# 실시간 판단
def webcam_processing(counter):
    with open('Textneck.pkl', 'rb') as f:
        model = pickle.load(f)

    landmarks = ['class']
    for val in range(1, 13+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]

    # VISUALIZE DEGREE
    # setting video capture device(number은 웹캠을 대표하는 숫자)
    cap = cv2.VideoCapture(0)

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():  # 실시간 영상을 가져올 수 있도록 함.
            ret, frame = cap.read()  # frame은 현재 프레임 이미지가 담긴 것.

            # Detect stuff and render
            # Recolor image to RGB
            # 웹캠으로 읽어온 frame을 BGR에서 RGB로 변환(Mediapipe는 RGB 형식임.)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # 이미지를 불변으로 설정하여 처리 속도를 향상 시킴.

            # Make detection -> 자세 detection을 results라는 변수에
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True  # image 위에 그릴 수 있도록.
            # Mediapipe 처리 결과를 BGR로 변환
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                row = []
                for k in range(12+1):
                    row.append(results.pose_landmarks.landmark[k].x)
                    row.append(results.pose_landmarks.landmark[k].y)
                    row.append(results.pose_landmarks.landmark[k].z)

                row = np.array(row).flatten()

                X = pd.DataFrame([row], columns=landmarks[1:])

                # 각도들도 넣어주기
                # left angle 추가 -> csv에서는 1부터 시작. landmarks에서는 0부터 시작.
                left_angle = angle3(2, 4, 8, X)

                # right angle 추가
                right_angle = angle3(5, 7, 9, X)

                X['left_angle'] = left_angle
                X['right_angle'] = right_angle

                # predict
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                if body_language_class == 0 and body_language_prob[body_language_prob.argmax()] >= .7:
                    current_stage = 'Good'
                elif current_stage == 'Good' and body_language_class == 1 and body_language_prob[body_language_prob.argmax()] >= .7:
                    current_stage = 'Textneck'
                    counter.value += 1
                    cnt = counter.value

                # Setup status bow
                cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

                # 지금 상태
                cv2.putText(image, 'CLASS', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, 'Textneck' if body_language_class == 1 else 'Good',
                            (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Stage data
                cv2.putText(image, 'Count', (180, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                cv2.putText(image, str(cnt), (175, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            except:  # error가 있으면 실행x
                pass

            cv2.imshow('Mediapipe Feed', image)  # 웹캠에서의 실시간 영상 확인 가능

            if cv2.waitKey(10) & 0xFF == ord('q'):  # 웹캠 화면을 종료하는 방법
                break

        cap.release()  # 비디오 객체 해제
        cv2.destroyAllWindows()  # 열린 opencv 창 전부 닫음.


if __name__ == "__main__":
    from multiprocessing import Value, Process
    counter = Value('i', 0)
    webcam_processing(counter)
