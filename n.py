"""
    用 opencv 讀取兩個影片
    分別生成骨架形狀
    並分析骨架相似度
    在分析完之後合併到同一視窗以供查看
    相似度部分除了看角度外，也看絕對位置 xyz都看

    使用注意：影片大小如果不一樣，會自動偵測並調整成一樣的大小
    但如果frame rate不一樣, 沒辦法解決
    
    測試不同pixel的股價偵測出來的結果是否有差異
"""



import cv2
import numpy as np
import mediapipe as mp
import csv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# 設定原影片與要比較相似度的影片
input_video = 'B1.mp4'
compare_video = 'B1_trans.mp4'


class skeleton_detection_similarity:
    def __init__(self, video1, video2):
        # 打開兩個影片
        self.cap1 = cv2.VideoCapture(video1)
        self.cap2 = cv2.VideoCapture(video2)

        # 獲取第一個視頻的寬度和高度
        width1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 獲取第二個視頻的寬度和高度
        width2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # # 設置調整後的寬度和高度（例如縮小一半）
        # self.resized_width1 = int(width1 / adjust1)
        # self.resized_height1 = int(height1 / adjust1)
        # self.resized_width2 = int(width2 / adjust2)
        # self.resized_height2 = int(height2 / adjust2)

        # 計算統一的目標寬高，以較小的寬高比例為基準
        target_width = min(width1, width2)
        target_height = min(height1, height2)

        # 計算調整比例
        self.scale1 = target_width / width1
        self.scale2 = target_width / width2
        self.target_width = target_width
        self.target_height = target_height

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
            (13, 11, 23),   # 左肘-左肩-左腰
            (14, 12, 24),   # 右肘-右肩-右腰
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
            (15, 19, 17),   # 左腕-左手-左小拇指
            (16, 20, 18),   # 右腕-右手-右小拇指
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
            (1,  0,  4 ),   # 左內眼角-鼻子-右內眼角
            (10, 0,  9 ),   # 右嘴-鼻子-左嘴
        ]

        # 初始化CSV文件
        self.csv_angle = open(f'skeleton_angle_similarity_{video1}.csv', mode='w', newline='', encoding='utf-8')
        self.csv_angle_writer = csv.writer(self.csv_angle)
        # 寫入標題行
        self.csv_angle_writer.writerow(['point', 'angle1', 'angle2'])

        # 初始化CSV文件
        self.csv_position = open(f'skeleton_position_similarity_{video1}.csv', mode='w', newline='', encoding='utf-8')
        self.csv_position_writer = csv.writer(self.csv_position)
        # 寫入標題行
        self.csv_position_writer.writerow(['point', 'position1.x', 'position1.y', 'position1.z', 'position2.x', 'position2.y', 'position2.z'])

        # 初始化CSV文件
        self.csv_similarity = open(f'skeleton_eachframe_similarity_{video1}.csv', mode='w', newline='', encoding='utf-8')
        self.csv_similarity_writer = csv.writer(self.csv_similarity)
        # 寫入標題行
        self.csv_similarity_writer.writerow(['angle_similarity_percentage', 'position_similarity_percentage', 'average_similarity'])

        # 取相似度圖表的名字
        self.chart = f'skeleton_similarity_{video1}.png'
        

    # 計算三點之間的角度
    def get_angle(self, p1, p2, p3):
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


    def calculate_angle_similarity(self, landmarks1, landmarks2):
        angles1 = []
        angles2 = []

        for points in self.key_points:
            p1_1 = [landmarks1[points[0]].x, landmarks1[points[0]].y]
            p2_1 = [landmarks1[points[1]].x, landmarks1[points[1]].y]
            p3_1 = [landmarks1[points[2]].x, landmarks1[points[2]].y]
            
            p1_2 = [landmarks2[points[0]].x, landmarks2[points[0]].y]
            p2_2 = [landmarks2[points[1]].x, landmarks2[points[1]].y]
            p3_2 = [landmarks2[points[2]].x, landmarks2[points[2]].y]
            
            # 呼叫函式計算角度
            angle1 = self.get_angle(p1_1, p2_1, p3_1)
            angle2 = self.get_angle(p1_2, p2_2, p3_2)
            
            angles1.append(angle1)
            angles2.append(angle2)

            # 寫入 CSV 檔
            self.csv_angle_writer.writerow([points, angle1, angle2])
        
        # 計算角度相似度
        angles1 = np.array(angles1)
        angles2 = np.array(angles2)
        
        cosine_similarity = np.dot(angles1, angles2) / (np.linalg.norm(angles1) * np.linalg.norm(angles2))
        angle_similarity_percentage = cosine_similarity * 100

        # 寫入 CSV 檔
        # self.csv_angle_writer.writerow(["angle_similarity_percentage", angle_similarity_percentage, ""])
 
        return angle_similarity_percentage


    def calculate_position_similarity(self, landmarks1, landmarks2):
        distances = []

        # 提取所有關節點的座標(共33點)
        all_positions1 = {idx: (landmark.x, landmark.y, landmark.z) for idx, landmark in enumerate(landmarks1)}
        all_positions2 = {idx: (landmark.x, landmark.y, landmark.z) for idx, landmark in enumerate(landmarks2)}

        # 對兩部影片的同一個點計算歐幾里得距離
        for idx in all_positions1.keys():
            coord1 = all_positions1[idx]
            coord2 = all_positions2[idx]

            distance = np.sqrt(
                (coord1[0] - coord2[0]) ** 2 +
                (coord1[1] - coord2[1]) ** 2 +
                (coord1[2] - coord2[2]) ** 2
            )
            distances.append(distance)

            # 寫入 CSV 檔
            self.csv_position_writer.writerow([idx, coord1[0], coord1[1], coord1[2], coord2[0], coord2[1], coord2[2]])

        # 計算位置相似度 (平均距離的倒數)
        avg_distance = np.mean(distances)
        position_similarity_percentage = (1 - avg_distance) * 100

        # 寫入 CSV 檔
        # self.csv_position_writer.writerow(["position_similarity_percentage", position_similarity_percentage, '', '', '', '', ''])

        return position_similarity_percentage


    # 比較兩個骨架的角度並計算相似度
    def calculate_similarity(self, landmarks1, landmarks2):
        # 得出角度相似度
        angle_similarity_percentage = self.calculate_angle_similarity(landmarks1, landmarks2)
        # 得出位置相似度
        position_similarity_percentage = self.calculate_position_similarity(landmarks1, landmarks2)
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
                
                # # 調整長寬
                # after1 = cv2.resize(frame1, (self.resized_width1, self.resized_height1))
                # after2 = cv2.resize(frame2, (self.resized_width2, self.resized_height2))
                after1 = cv2.resize(frame1, (self.target_width, self.target_height))
                after2 = cv2.resize(frame2, (self.target_width, self.target_height))

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

                # # 自動調整大小
                # resized_frame1 = cv2.resize(frame1, (self.target_width, self.target_height))
                # resized_frame2 = cv2.resize(frame2, (self.target_width, self.target_height))

                # # 進行骨架偵測與相似度計算
                # frame1_rgb = cv2.cvtColor(resized_frame1, cv2.COLOR_BGR2RGB)
                # frame2_rgb = cv2.cvtColor(resized_frame2, cv2.COLOR_BGR2RGB)
                # results1 = pose.process(frame1_rgb)
                # results2 = pose.process(frame2_rgb)

                # # 畫出骨架
                # if results1.pose_landmarks:
                #     self.mp_drawing.draw_landmarks(
                #         resized_frame1,
                #         results1.pose_landmarks,
                #         self.mp_pose.POSE_CONNECTIONS)

                # if results2.pose_landmarks:
                #     self.mp_drawing.draw_landmarks(
                #         resized_frame2,
                #         results2.pose_landmarks,
                #         self.mp_pose.POSE_CONNECTIONS)

                # 如果兩個視頻都有檢測到骨架，計算相似度
                if results1.pose_landmarks and results2.pose_landmarks:
                    angle_similarity, position_similarity, average_similarity = self.calculate_similarity(results1.pose_landmarks.landmark, results2.pose_landmarks.landmark)
                    self.angle_similarity_list.append(angle_similarity)
                    self.position_similarity_list.append(position_similarity)
                    self.average_similarity_list.append(average_similarity)
                    self.csv_similarity_writer.writerow([angle_similarity, position_similarity, average_similarity])


                # 創建分隔線
                # separator = np.zeros((self.resized_height1, 10, 3), dtype=np.uint8)
                separator = np.zeros((self.target_height, 10, 3), dtype=np.uint8)
                separator[:] = (0, 0, 255)  # 將分隔線設置為紅色

                # 合併兩個視頻到一個窗口中
                combined_frame = np.hstack((after1, separator, after2))
                # combined_frame = np.hstack((resized_frame1, separator, resized_frame2))
                
                # 顯示調整後的視頻
                cv2.imshow('Combined Video', combined_frame)

                # 按 'q' 鍵退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # 釋放視頻對象並關閉所有窗口
        self.cap1.release()
        self.cap2.release()
        self.csv_angle.close()
        self.csv_position.close()
        self.csv_similarity.close()
        cv2.destroyAllWindows()
    

    # 繪製相似度折線圖
    def plot_similarity_graphs(self):
        # 創建圖表和 GridSpec 佈局
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(2, 3, height_ratios=[3, 1], figure=fig)

        # 角度相似度圖
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.angle_similarity_list, label="Angle Similarity", color="blue")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Similarity (%)")
        ax1.set_title("Angle Similarity Percentage")
        ax1.grid(True)
        ax1.legend()

        # 位置相似度圖
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.position_similarity_list, label="Position Similarity", color="green")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Similarity (%)")
        ax2.set_title("Position Similarity Percentage")
        ax2.grid(True)
        ax2.legend()

        # 平均相似度圖
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.average_similarity_list, label="Average Similarity", color="red")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Similarity (%)")
        ax3.set_title("Average Similarity Percentage")
        ax3.grid(True)
        ax3.legend()

        # 文本信息
        ax_text = fig.add_subplot(gs[1, :])
        ax_text.axis('off')  # 隱藏坐標軸

        # 構建信息文本
        info_text = (
            f"Keypoint Length: {len(self.key_points)}\n"
            f"List Length: {len(self.angle_similarity_list)}\n"
            f"Average Angle Similarity: {np.mean(self.angle_similarity_list):.2f}%\n"
            f"Average Position Similarity: {np.mean(self.position_similarity_list):.2f}%\n"
            f"Overall Average Similarity: {np.mean(self.average_similarity_list):.2f}%"
        )

        # 添加文本到圖表
        ax_text.text(0.5, 0.5, info_text, ha="center", va="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":10})

        # 調整佈局
        plt.tight_layout()
        plt.savefig(self.chart, dpi=300)
        plt.close()


if __name__ == '__main__':
    sds = skeleton_detection_similarity(input_video, compare_video)
    sds.run()

    print("keypoint len:", len(sds.key_points))
    print("list len:", len(sds.angle_similarity_list))
    # 印出相似度
    print(f"Average Angle Similarity: {np.mean(sds.angle_similarity_list):.2f}%")
    print(f"Average Position Similarity: {np.mean(sds.position_similarity_list):.2f}%")
    print(f"Overall Average Similarity: {np.mean(sds.average_similarity_list):.2f}%")
    sds.plot_similarity_graphs()

