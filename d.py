"""
    用 opencv 讀取兩個影片
"""


import cv2
import numpy as np

# 打開兩個影片
cap1 = cv2.VideoCapture('B1.mp4')
cap2 = cv2.VideoCapture('B2.mp4')

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        print("Cannot receive frame")
        break

    after1 = cv2.resize(frame1, (0, 0), fx = 0.2, fy = 0.2)
    after2 = cv2.resize(frame2, (0, 0), fx = 0.2, fy = 0.2)

    cv2.imshow('frame1', after1)
    cv2.imshow('frame2', after2)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放視頻對象並關閉所有窗口
cap1.release()
cap2.release()
cv2.destroyAllWindows()
