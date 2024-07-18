"""
    用 opencv 讀取兩個影片
    分別生成骨架形狀
    在分析完之後合併到同一視窗以供查看
"""


import cv2
import numpy as np
import mediapipe as mp

# mediapipe 初始設定
mp_drawing = mp.solutions.drawing_utils         # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                     # mediapipe 姿勢偵測方法

# 打開兩個影片
cap1 = cv2.VideoCapture('B1.mp4')
cap2 = cv2.VideoCapture('B2.mp4')

# 獲取第一個視頻的寬度和高度
width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 獲取第二個視頻的寬度和高度
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 設置調整後的寬度和高度（例如縮小一半）
adjust = 6
resized_width1 = width1 // adjust
resized_height1 = height1 // adjust
resized_width2 = width2 // adjust
resized_height2 = height2 // adjust

# 啟用姿勢偵測
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    # 確認是否成功打開
    if not cap1.isOpened():
        print("Cannot open camera1")
        exit()
    if not cap2.isOpened():
        print("Cannot open camera2")
        exit()

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("Cannot receive frame")
            break
        
        # 調整長寬
        after1 = cv2.resize(frame1, (resized_width1, resized_height1))
        after2 = cv2.resize(frame2, (resized_width2, resized_height2))

        resized_frame1 = cv2.cvtColor(after1, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = pose.process(resized_frame1)                  # 取得姿勢偵測結果
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            after1,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        resized_frame2 = cv2.cvtColor(after2, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = pose.process(resized_frame2)                  # 取得姿勢偵測結果
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            after2,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # 創建分隔線
        separator = np.zeros((resized_height1, 10, 3), dtype=np.uint8)
        separator[:] = (0, 0, 255)  # 將分隔線設置為紅色

        # 合併兩個視頻到一個窗口中
        combined_frame = np.hstack((after1, separator, after2))
        
        # 顯示調整後的視頻
        cv2.imshow('Combined Video', combined_frame)

        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 釋放視頻對象並關閉所有窗口
cap1.release()
cap2.release()
cv2.destroyAllWindows()
