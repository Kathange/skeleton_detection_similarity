"""
    用 opencv 讀取一個影片
    生成骨架形狀
"""


import numpy as np
import cv2
import mediapipe as mp

# mediapipe 初始設定
mp_drawing = mp.solutions.drawing_utils         # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                     # mediapipe 姿勢偵測方法

cap = cv2.VideoCapture('B1.mp4')

# 啟用姿勢偵測
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    
    # 確認是否成功打開
    if not cap.isOpened():
        print("Cannot open camera1")
        exit()
    
    while(True):
        ret, frame = cap.read()
        after = cv2.resize(frame, (0, 0), fx = 0.2, fy = 0.2)

        resized_frame = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = pose.process(resized_frame)                  # 取得姿勢偵測結果
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            after,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow('frame', after)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()