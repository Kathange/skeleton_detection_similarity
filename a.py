"""
    測試Mediapipe是否能夠正常執行
    https://hackmd.io/@am534143/r1pch8Y1p#%E4%BD%BF%E7%94%A8Mediapipe%E5%88%86%E6%9E%90%E5%8B%95%E4%BD%9C
"""


import mediapipe as mp
import numpy as np 
import cv2

mp_drawing = mp.solutions.drawing_utils         # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles # mediapipe 繪圖樣式
mp_holistic = mp.solutions.holistic             # mediapipe 全身偵測方法
#若要偵測姿勢則改成
mp_pose = mp.solutions.pose
#若要偵測手部則改成
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)#開啟攝影機
#需先確定裝備是否有內建攝影機，若有則不須變動，若沒有，請先插上攝影機後再執行

# mediapipe 啟用偵測全身
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()  #若無法開啟攝影機則退出
    while True: #可以打開攝影機
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break #若無法收到畫面則退出
        img = cv2.resize(img,(640,480))               #將攝影機畫面設定大小為(640,480)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = holistic.process(img2)              # 開始偵測全身
        # 身體偵測，繪製身體骨架             
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        cv2.imshow('warmup', img) #視窗名稱
        if cv2.waitKey(5) == ord('q'):
            break       #按下 q 鍵停止
cap.release()           #釋放資源
cv2.destroyAllWindows() #刪除視窗