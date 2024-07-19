"""
    用 opencv 讀取兩個影片
    分別生成骨架形狀
    並分析骨架相似度
    在分析完之後合併到同一視窗以供查看
"""


import cv2
import numpy as np
import mediapipe as mp

# mediapipe 初始設定
mp_drawing = mp.solutions.drawing_utils         # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                     # mediapipe 姿勢偵測方法

# 定義需要比較的關鍵點組（如肩、肘、腕等）
key_points = [
    (11, 13, 15),   # 左肩-左肘-左腕
    (12, 14, 16),   # 右肩-右肘-右腕
    (23, 25, 27),   # 左臀-左膝-左踝
    (24, 26, 28),   # 右臀-右膝-右踝
    (11, 23, 25),   # 左肩-左腰-左膝
    (12, 24, 26),   # 右肩-右腰-右膝
    (13, 15, 21),   # 左肘-左腕-左大拇指
    (14, 16, 22),   # 右肘-右腕-右大拇指
    (19, 15, 17),   # 左手-左腕-左小拇指
    (20, 16, 18),   # 右手-右腕-右小拇指
    (15, 17, 19),   # 左腕-左小拇指-左手
    (16, 18, 20),   # 右腕-右小拇指-右手
    (25, 27, 29),   # 左膝-左角踝-左腳跟
    (26, 28, 30),   # 右膝-右角踝-右腳跟
    (27, 29, 31),   # 左腳踝-左腳跟-左腳大拇指
    (28, 30, 32),   # 右腳踝-右腳跟-右腳大拇指
    (27, 31, 29),   # 左腳踝-左腳大拇指-左腳跟
    (28, 32, 30),   # 右腳踝-右腳大拇指-右腳跟
    (7,  3,  2 ),   # 左耳-左外眼角-左眼
    (8,  6,  5 ),   # 右耳-右外眼角-右眼
    (2,  1,  0 ),   # 左眼-左內眼角-鼻子
    (5,  4,  0 ),   # 右眼-右內眼角-鼻子
    (1,  0,  4 )    # 左內眼角-鼻子-右內眼角
]

# 計算兩點之間的角度
def calculate_angle(p1, p2, p3):
    a = np.array(p1) # First
    b = np.array(p2) # Mid
    c = np.array(p3) # End

    # 使用arctan2()函式來進行運算，其得到之值為兩點之弧度，
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    # 而求出兩個弧度後再作相減，再乘以pi，即可得到三點連線之角度。
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# 比較兩個骨架的角度並計算相似度
def calculate_similarity(landmarks1, landmarks2):
    angles1 = []
    angles2 = []
    
    for points in key_points:
        p1_1 = [landmarks1[points[0]].x, landmarks1[points[0]].y]
        p2_1 = [landmarks1[points[1]].x, landmarks1[points[1]].y]
        p3_1 = [landmarks1[points[2]].x, landmarks1[points[2]].y]
        
        p1_2 = [landmarks2[points[0]].x, landmarks2[points[0]].y]
        p2_2 = [landmarks2[points[1]].x, landmarks2[points[1]].y]
        p3_2 = [landmarks2[points[2]].x, landmarks2[points[2]].y]
        
        angle1 = calculate_angle(p1_1, p2_1, p3_1)
        angle2 = calculate_angle(p1_2, p2_2, p3_2)
        
        angles1.append(angle1)
        angles2.append(angle2)
    
    angles1 = np.array(angles1)
    angles2 = np.array(angles2)
    
    cosine_similarity = np.dot(angles1, angles2) / (np.linalg.norm(angles1) * np.linalg.norm(angles2))
    similarity_percentage = cosine_similarity * 100
    
    return similarity_percentage




# 打開兩個影片
cap1 = cv2.VideoCapture('B3.mp4')
cap2 = cv2.VideoCapture('B4.mp4')

# 獲取第一個視頻的寬度和高度
width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 獲取第二個視頻的寬度和高度
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 設置調整後的寬度和高度（例如縮小一半）
adjust = 5
resized_width1 = width1 // adjust
resized_height1 = height1 // adjust
resized_width2 = width2 // adjust
resized_height2 = height2 // adjust

# 將每一幀得相似度放進 list 中
similarity_list = []

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
        results1 = pose.process(resized_frame1)                  # 取得姿勢偵測結果
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            after1,
            results1.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        resized_frame2 = cv2.cvtColor(after2, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results2 = pose.process(resized_frame2)                  # 取得姿勢偵測結果
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            after2,
            results2.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # 如果兩個視頻都有檢測到骨架，計算相似度
        if results1.pose_landmarks and results2.pose_landmarks:
            similarity_percentage = calculate_similarity(results1.pose_landmarks.landmark, results2.pose_landmarks.landmark)
            similarity_list.append(similarity_percentage)


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

print("keypoint len:", len(key_points))
print("list len:", len(similarity_list))
# 印出相似度
print(f"Average Similarity: {np.mean(similarity_list):.2f}%")

# 釋放視頻對象並關閉所有窗口
cap1.release()
cap2.release()
cv2.destroyAllWindows()
