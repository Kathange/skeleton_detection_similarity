"""
    用 opencv 讀取影片
"""


import numpy as np
import cv2

cap = cv2.VideoCapture('B1.mp4')

while(True):
    ret, frame = cap.read()

    after = cv2.resize(frame, (0, 0), fx = 0.2, fy = 0.2)

    cv2.imshow('frame', after)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()