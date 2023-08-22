import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import pandas as pd
import pickle
import serial

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

mp_drawing = mp.solutions.drawing_utils 
mp_pose = mp.solutions.pose 

df = pd.read_csv('coords_spine_final.csv')

X = df.drop('spine',axis=1) #shol을 제외한 입력 feature만 남김.
y = df['spine']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=123) #어깨

pipelines = {
    'lr' : make_pipeline(StandardScaler(),LogisticRegression()),
    'rc' : make_pipeline(StandardScaler(),RidgeClassifier()),
    'rf' : make_pipeline(StandardScaler(),RandomForestClassifier()),
    'gb' : make_pipeline(StandardScaler(),GradientBoostingClassifier())
}

fit_models={}
for algo,pipeline in pipelines.items():
    model = pipeline.fit(X_train,y_train)
    fit_models[algo] = model

for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo,accuracy_score(y_test.values,yhat),
    precision_score(y_test.values,yhat,average=None),
    recall_score(y_test.values,yhat,average=None))

yhat = fit_models['rf'].predict(X_test)

with open('Spine.pkl','wb') as f:
    pickle.dump(fit_models['rf'],f)

with open('Spine.pkl','rb') as f:
    model = pickle.load(f)

landmarks=['spine']
for val in range(1,13+1):
    landmarks +=['x{}'.format(val),'y{}'.format(val),'z{}'.format(val),'v{}'.format(val)]


port = "COM6"
brate = 9600 
cmd = 't'

seri = serial.Serial(port, baudrate = brate, timeout = None)
print(seri.name)

seri.write(cmd.encode())

counter =0

cap =cv2.VideoCapture(0) 

with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened(): 
        ret, frame = cap.read() 
        if not ret:
            print("Error: Unable to capture frame from the camera.")


        image =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) 
        image.flags.writeable =False 
        
        results= pose.process(image)
        
        image.flags.writeable=True 
        image =cv2.cvtColor(image,cv2.COLOR_RGB2BGR) 
        

        row =[]
        try:  
            for k in range(12+1):
                row.append(results.pose_landmarks.landmark[k].x)
                row.append(results.pose_landmarks.landmark[k].y)
                row.append(results.pose_landmarks.landmark[k].z)
                row.append(results.pose_landmarks.landmark[k].visibility)

            
            row = np.array(row).flatten()
            
            X = pd.DataFrame([row],columns=landmarks[1:])


            if seri.in_waiting != 0 :
                content1 = seri.readline().decode('ascii')
                X['StnX'] = float(content1[:-2])
                content2 = seri.readline().decode('ascii')
                X['StnY'] = float(content2[:-2])
                content3 = seri.readline().decode('ascii')
                X['X0'] = float(content3[:-2])
                content4 = seri.readline().decode('ascii')
                X['Y0'] = float(content4[:-2])
                content5 = seri.readline().decode('ascii')
                X['X'] = float(content5[:-2])
                content6 = seri.readline().decode('ascii')
                X['Y'] = float(content6[:-2])

            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]

            if body_language_class == 0 and body_language_prob[body_language_prob.argmax()] >= .7:
                current_stage = 'Good'
            elif current_stage == 'Good' and body_language_class ==1 and body_language_prob[body_language_prob.argmax()] >= .7:
                current_stage = 'left'
                counter +=1
            elif current_stage == 'Good' and body_language_class ==2 and body_language_prob[body_language_prob.argmax()] >= .7:
                current_stage = 'right'
                counter +=1


            cv2.rectangle(image,(0,0),(250,60),(245,117,16),-1) 


            cv2.putText(image,'CLASS',(15,12),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image, 'left' if body_language_class == 1 else 'right' if body_language_class == 2 else 'Good',(15,40)
                        ,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            
            cv2.putText(image,'Count',(180,12),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)

            cv2.putText(image,str(counter),(175,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(155,117,166),thickness=2,circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(195,116,230),thickness=2,circle_radius=2))


        except: 
            pass
        cv2.imshow('Mediapipe Feed', image) 

        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break

    cap.release() 
    cv2.destroyAllWindows() 