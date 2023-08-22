import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import serial
import time
import pandas as pd
import pickle
import cv2
import mediapipe as mp

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

mp_drawing = mp.solutions.drawing_utils 
mp_pose = mp.solutions.pose 

df = pd.read_csv('coords.csv')
df = df.filter(regex='^x[0-9]$|^y[0-9]$|^z[0-9]$|^x1[0-3]|^y1[0-3]|^z1[0-3]|^c')

def angle3(a,b,c,df):
    a_x= np.array(df['x{}'.format(a)]) 
    a_y = np.array(df['y{}'.format(a)])

    b_x = np.array(df['x{}'.format(b)]) 
    b_y = np.array(df['y{}'.format(b)])

    c_x = np.array(df['x{}'.format(c)])
    c_y = np.array(df['y{}'.format(c)]) 

    radians=np.arctan2(c_y-b_y,c_x-b_x) - np.arctan2(a_y-b_y,a_x-b_x)
    angle=np.abs(radians*180.0/np.pi)

    for i in range(0,angle.shape[0]):
        if angle[i]>180.0:
            angle[i] = 360-angle[i]

    return pd.Series(angle)

left_angle = angle3(2,4,8,df)
right_angle = angle3(5,7,9,df)

df['left_angle'] = left_angle
df['right_angle'] = right_angle

X = df.drop('class',axis=1) 
y = df['class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=123)

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

yhat = fit_models['rf'].predict(X_test)

with open('Textneck.pkl','wb') as f:
    pickle.dump(fit_models['rf'],f)

py_serial = serial.Serial(
    
    port='COM11',
    
    baudrate=9600,
)

with open('Textneck.pkl','rb') as f:
    model = pickle.load(f)

landmarks=['class']
for val in range(1,13+1):
    landmarks +=['x{}'.format(val),'y{}'.format(val),'z{}'.format(val)]

cap =cv2.VideoCapture(0)

counter = 0

with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened(): 
        ret, frame = cap.read() 
      
        image =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) 
        image.flags.writeable =False 
        

        results= pose.process(image)
        
        image.flags.writeable=True 
        image =cv2.cvtColor(image,cv2.COLOR_RGB2BGR) 
        

        try:  
            row = []
            for k in range(12+1):
                row.append(results.pose_landmarks.landmark[k].x)
                row.append(results.pose_landmarks.landmark[k].y)
                row.append(results.pose_landmarks.landmark[k].z)
            
            row = np.array(row).flatten()
            
            X = pd.DataFrame([row],columns=landmarks[1:])
            
            left_angle = angle3(2,4,8,X)


            right_angle = angle3(5,7,9,X)

            X['left_angle'] = left_angle
            X['right_angle'] = right_angle

            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            
            if body_language_class == 0 and body_language_prob[body_language_prob.argmax()] >= .7:
                current_stage = "Good"
            elif current_stage == 'Good' and body_language_class ==1 and body_language_prob[body_language_prob.argmax()] >= .7 :
                current_stage = "Textneck"
                counter +=1
            
            #통신 코드
            
            py_serial.write(current_stage.encode())
        


            cv2.rectangle(image,(0,0),(250,60),(245,117,16),-1) 



            cv2.putText(image,'CLASS',(15,12),
                       cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image, 'Textneck' if body_language_class == 1 else 'Good',(15,40)
                        ,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)


            cv2.putText(image,'Count',(180,12),
                       cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
    
            cv2.putText(image,str(counter),(175,40),
                       cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                                     mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))

        except: 
            pass
        
        cv2.imshow('Mediapipe Feed', image) 
        

    
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release() 
    cv2.destroyAllWindows() 



