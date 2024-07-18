"""
    用 opencv 讀取兩個影片
    一個放左邊一個放右邊
"""


import cv2
import numpy as np

# 打開兩個影片
cap1 = cv2.VideoCapture('B1.mp4')
cap2 = cv2.VideoCapture('B2.mp4')

# 獲取第一個視頻的寬度和高度
width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 獲取第二個視頻的寬度和高度
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 設置輸出視頻的寬度和高度
output_width = width1 + width2
output_height = max(height1, height2)
# 設置調整後的寬度和高度（例如縮小一半）
resized_width = output_width // 7
resized_height = output_height // 7

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        print("Cannot receive frame")
        break

    # # 如果兩個視頻的高度不同，調整高度以匹配
    # if height1 != height2:
    #     frame1 = cv2.resize(frame1, (width1, output_height))
    #     frame2 = cv2.resize(frame2, (width2, output_height))
    
    # 創建一個空白畫布
    output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    # 將第一個視頻放在左半邊
    output_frame[:, :width1] = frame1
    # 將第二個視頻放在右半邊
    output_frame[:, width1:] = frame2

    # 在交界處畫一條垂直線
    cv2.line(output_frame, (width1, 0), (width1, output_height), (0, 0, 255), 50)
    
    # 調整合併後的畫布大小
    resized_frame = cv2.resize(output_frame, (resized_width, resized_height))
    # 顯示調整後的視頻
    cv2.imshow('Combined Video', resized_frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放視頻對象並關閉所有窗口
cap1.release()
cap2.release()
cv2.destroyAllWindows()
