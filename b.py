"""
    如何提取骨架做角度運算
    透過計算角度判斷目前姿勢
    https://hackmd.io/@am534143/r1pch8Y1p#%E4%BD%BF%E7%94%A8Mediapipe%E5%88%86%E6%9E%90%E5%8B%95%E4%BD%9C
"""


import cv2
import numpy as np
import mediapipe as mp
import time


mppose = mp.solutions.pose
pose = mppose.Pose()
h = 0                   #int
w = 0                   #int
status = False          #bool
countdown_seconds = 30  #int
start_time = 0          #int


# 在get_knee_angle函式中，先呼叫了get_landmark函式，
# 並傳回一個陣列中包含的該位置的x,y,z座標，
# 並求出要運算之三點座標後，呼叫calc_angles函式來運算角度，
# 在calc_angles函式中，使用arctan2()函式來進行運算，
# 其得到之值為兩點之弧度，
# 而求出兩個弧度後再作相減，再乘以pi，即可得到三點連線之角度。

def calc_angles(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def get_landmark(landmarks, part_name):
    return [
        landmarks[mppose.PoseLandmark[part_name].value].x,
        landmarks[mppose.PoseLandmark[part_name].value].y,
        landmarks[mppose.PoseLandmark[part_name].value].z,
    ]


def get_visibility(landmarks):
    if landmarks[mppose.PoseLandmark["RIGHT_HIP"].value].visibility < 0.8 or \
            landmarks[mppose.PoseLandmark["LEFT_HIP"].value].visibility < 0.8:
        return False
    else:
        return True


def get_body_ratio(landmarks):
    r_body = abs(landmarks[mppose.PoseLandmark["RIGHT_SHOULDER"].value].y
                 - landmarks[mppose.PoseLandmark["RIGHT_HIP"].value].y)
    l_body = abs(landmarks[mppose.PoseLandmark["LEFT_SHOULDER"].value].y
                 - landmarks[mppose.PoseLandmark["LEFT_HIP"].value].y)
    avg_body = (r_body + l_body) / 2
    r_leg = abs(landmarks[mppose.PoseLandmark["RIGHT_HIP"].value].y
                - landmarks[mppose.PoseLandmark["RIGHT_ANKLE"].value].y)
    l_leg = abs(landmarks[mppose.PoseLandmark["LEFT_HIP"].value].y
                - landmarks[mppose.PoseLandmark["LEFT_ANKLE"].value].y)
    if r_leg > l_leg:
        return r_leg / avg_body
    else:
        return l_leg / avg_body


def get_knee_angle(landmarks):
    r_hip = get_landmark(landmarks, "RIGHT_HIP")
    l_hip = get_landmark(landmarks, "LEFT_HIP")

    r_knee = get_landmark(landmarks, "RIGHT_KNEE")
    l_knee = get_landmark(landmarks, "LEFT_KNEE")

    r_ankle = get_landmark(landmarks, "RIGHT_ANKLE")
    l_ankle = get_landmark(landmarks, "LEFT_ANKLE")

    r_angle = calc_angles(r_hip, r_knee, r_ankle)
    l_angle = calc_angles(l_hip, l_knee, l_ankle)


    return [r_angle, l_angle]




video1 = cv2.VideoCapture('output_video.mp4')                    #設定video1開啟物件路徑
video1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'XVID')) #指定讀取格式為XVID
video1.set(cv2.CAP_PROP_FPS, 30)                                 #設定幀率
video1.set(cv2.CAP_PROP_POS_FRAMES, 0)                           #設定物件初始化

video2 = cv2.VideoCapture(0)                                     #設定video2開啟物件路徑

if not video1.isOpened() or not video2.isOpened():
    print("無法打開影片")
    exit()             #若無法打開則退出

frame_width2 = 1280    #設定攝影機螢幕寬
frame_height2 = 960    #設定攝影機螢幕高
print(frame_height2)   #960
print(frame_width2)    #1280

scaled_width = frame_width2 // 6     #骨架影片寬為主螢幕之1/6
scaled_height = frame_height2 // 6   #骨架影片高為主螢幕之1/6

cv2.namedWindow('Combined Video', cv2.WINDOW_NORMAL)             #設定視窗名字
cv2.resizeWindow('Combined Video', frame_width2, frame_height2)  #設定視窗大小

while True:                               #持續偵測
    if not status:                        #若status為False
        start_time = time.time()          #設定開始時間
    ret1, frame1 = video1.read()          #讀取骨架影片
    ret2, frame2 = video2.read()          #讀取攝影機

    if not ret1:                                  #若骨架影片撥放完畢
        video1.set(cv2.CAP_PROP_POS_FRAMES, 0)    #初始化影片
        continue                                  #繼續執行

    if not ret2:                                  #若讀取不到攝影機畫面
        break                                     #跳出迴圈
    results = pose.process(frame2)                #攝影機讀取骨架結果
    if results.pose_landmarks is not None:        #若有讀取到骨架
        mp_drawing = mp.solutions.drawing_utils   #繪圖方式
        annotated_image = frame2.copy()           #複製攝影到的幀
        mp_drawing.draw_landmarks(                #於複製的畫面上作畫
        annotated_image, results.pose_landmarks, mppose.POSE_CONNECTIONS)
        knee_angles = get_knee_angle(results.pose_landmarks.landmark)          #計算角度                     
        if 115<knee_angles[0] < 140 and 160<knee_angles[1]< 180:               #符合標準
            cv2.putText(annotated_image, "Left: {:.1f}".format(knee_angles[0]), (10, 230)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Right: {:.1f}".format(knee_angles[1]), (10, 260)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Good! Keep going!", (10, 290)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            #在複製的畫面上印出文字，分別為左腳右腳之角度，並顯示綠色
            status= True     #令status為True
        elif 115<knee_angles[1] < 140 and  160<knee_angles[0]< 180:            #符合標準
            cv2.putText(annotated_image, "Left: {:.1f}".format(knee_angles[0]), (10, 230)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Right: {:.1f}".format(knee_angles[1]), (10, 260)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA) 
            cv2.putText(annotated_image, "Good! Keep going!", (10, 290)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)               #在複製的畫面上印出文字，分別為左腳右腳之角度，並顯示綠色
            status= True    #令status為True
        else:                                                                  #不符合標準
            cv2.putText(annotated_image, "Left: {:.1f}".format(knee_angles[0]), (10, 230)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Right: {:.1f}".format(knee_angles[1]), (10, 260)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Squat down!", (10, 290)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA) 
            #在複製的畫面上印出文字，分別為左腳右腳之角度，並顯示紅色
            status= False    #令status為False            
        if status:                                                      #若status為True
            current_time = time.time()                                  #取得目前時間
            elapsed_time = int(current_time - start_time)               #目前時間與開始時間之時間差
            remaining_seconds = max(0, countdown_seconds - elapsed_time)#算出剩餘時間

            text = f" {remaining_seconds} second"
            cv2.putText(annotated_image, text, (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                                                      #印出剩餘時間

            if remaining_seconds == 0:                                  #若剩餘時間歸0
                status = False                                          #令status為False

    frame1_resized = cv2.resize(frame1, (scaled_width, scaled_height))  #將骨架影片縮小至指定尺寸
    annotated_image[0:scaled_height, 0:scaled_width] = frame1_resized   #複製幀中放入骨架影片
    cv2.imshow('Combined Video', annotated_image)                       #將複製幀顯示出來
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break            #按q退出
video1.release()         #釋放資源
video2.release()         #釋放資源
cv2.destroyAllWindows()  #釋放畫面



