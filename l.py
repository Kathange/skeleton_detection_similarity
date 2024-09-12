"""
    用 opencv 讀取兩個影片
    分別生成骨架形狀
    並分析骨架相似度
    在分析完之後合併到同一視窗以供查看
    相似度部分除了看角度外，也看絕對位置 xyz都看
"""


import cv2
import numpy as np
import mediapipe as mp

class skeleton_detection_similarity:
    def __init__(self, video1, video2, adjust=3):
        # 打開兩個影片
        self.cap1 = cv2.VideoCapture(video1)
        self.cap2 = cv2.VideoCapture(video2)

        # 獲取第一個視頻的寬度和高度
        width1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 獲取第二個視頻的寬度和高度
        width2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 設置調整後的寬度和高度（例如縮小一半）
        self.resized_width1 = width1 // adjust
        self.resized_height1 = height1 // adjust
        self.resized_width2 = width2 // adjust
        self.resized_height2 = height2 // adjust

        # 將每一幀得相似度放進 list 中
        self.angle_similarity_list = []
        self.position_similarity_list = []
        self.average_similarity_list = []

        # mediapipe 初始設定
        self.mp_drawing = mp.solutions.drawing_utils         # mediapipe 繪圖方法
        self.mp_drawing_styles = mp.solutions.drawing_styles # mediapipe 繪圖樣式
        self.mp_pose = mp.solutions.pose                     # mediapipe 姿勢偵測方法

        # 定義需要比較的關鍵點組（如肩、肘、腕等）
        self.key_points = [
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
    def calculate_angle(self, p1, p2, p3):
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
    def calculate_similarity(self, landmarks1, landmarks2):
        angles1 = []
        angles2 = []
        distances = []
        
        for points in self.key_points:
            p1_1 = [landmarks1[points[0]].x, landmarks1[points[0]].y]
            p2_1 = [landmarks1[points[1]].x, landmarks1[points[1]].y]
            p3_1 = [landmarks1[points[2]].x, landmarks1[points[2]].y]
            
            p1_2 = [landmarks2[points[0]].x, landmarks2[points[0]].y]
            p2_2 = [landmarks2[points[1]].x, landmarks2[points[1]].y]
            p3_2 = [landmarks2[points[2]].x, landmarks2[points[2]].y]
            
            # 計算角度相似度
            angle1 = self.calculate_angle(p1_1, p2_1, p3_1)
            angle2 = self.calculate_angle(p1_2, p2_2, p3_2)
            
            angles1.append(angle1)
            angles2.append(angle2)

            # 計算 XYZ 位置相似度
            for idx in points:
                point1 = landmarks1[idx]
                point2 = landmarks2[idx]

                # 歐氏距離計算
                distance = np.sqrt(
                    (point1.x - point2.x) ** 2 + 
                    (point1.y - point2.y) ** 2 + 
                    (point1.z - point2.z) ** 2
                )
                distances.append(distance)
        
        # 計算角度相似度
        angles1 = np.array(angles1)
        angles2 = np.array(angles2)
        
        cosine_similarity = np.dot(angles1, angles2) / (np.linalg.norm(angles1) * np.linalg.norm(angles2))
        angle_similarity_percentage = cosine_similarity * 100

        # 計算位置相似度 (平均距離的倒數)
        avg_distance = np.mean(distances)
        position_similarity_percentage = (1 - avg_distance) * 100

        # 計算兩者的平均相似度
        average_similarity = (angle_similarity_percentage + position_similarity_percentage) / 2
        
        return angle_similarity_percentage, position_similarity_percentage, average_similarity

    def run(self):
        # 啟用姿勢偵測
        with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            # 確認是否成功打開
            if not self.cap1.isOpened():
                print("Cannot open camera1")
                exit()
            if not self.cap2.isOpened():
                print("Cannot open camera2")
                exit()

            while True:
                ret1, frame1 = self.cap1.read()
                ret2, frame2 = self.cap2.read()
                if not ret1 or not ret2:
                    print("Cannot receive frame")
                    break
                
                # 調整長寬
                after1 = cv2.resize(frame1, (self.resized_width1, self.resized_height1))
                after2 = cv2.resize(frame2, (self.resized_width2, self.resized_height2))

                resized_frame1 = cv2.cvtColor(after1, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
                results1 = pose.process(resized_frame1)                  # 取得姿勢偵測結果
                # 根據姿勢偵測結果，標記身體節點和骨架
                self.mp_drawing.draw_landmarks(
                    after1,
                    results1.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                
                resized_frame2 = cv2.cvtColor(after2, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
                results2 = pose.process(resized_frame2)                  # 取得姿勢偵測結果
                # 根據姿勢偵測結果，標記身體節點和骨架
                self.mp_drawing.draw_landmarks(
                    after2,
                    results2.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

                # 如果兩個視頻都有檢測到骨架，計算相似度
                if results1.pose_landmarks and results2.pose_landmarks:
                    angle_similarity, position_similarity, average_similarity = self.calculate_similarity(results1.pose_landmarks.landmark, results2.pose_landmarks.landmark)
                    self.angle_similarity_list.append(angle_similarity)
                    self.position_similarity_list.append(position_similarity)
                    self.average_similarity_list.append(average_similarity)


                # 創建分隔線
                separator = np.zeros((self.resized_height1, 10, 3), dtype=np.uint8)
                separator[:] = (0, 0, 255)  # 將分隔線設置為紅色

                # 合併兩個視頻到一個窗口中
                combined_frame = np.hstack((after1, separator, after2))
                
                # 顯示調整後的視頻
                cv2.imshow('Combined Video', combined_frame)

                # 按 'q' 鍵退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        print("keypoint len:", len(self.key_points))
        print("list len:", len(self.angle_similarity_list))
        # 印出相似度
        print(f"Average Angle Similarity: {np.mean(self.angle_similarity_list):.2f}%")
        print(f"Average Position Similarity: {np.mean(self.position_similarity_list):.2f}%")
        print(f"Overall Average Similarity: {np.mean(self.average_similarity_list):.2f}%")
    
        # 釋放視頻對象並關閉所有窗口
        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    sds = skeleton_detection_similarity('B3.mp4', 'B4.mp4')
    sds.run()
