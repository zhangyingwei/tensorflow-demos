#!/usr/bin/env python  
# encoding: utf-8  

"""
@version: v1.0 
@author: zhangyw
@site: http://blog.zhangyingwei.com
@software: PyCharm 
@file: FaceApplication.py 
@time: 2018/2/26 10:04 
"""

import matplotlib.pyplot as plt
from IPython import display
import face_recognition
import cv2


# 初始化人员(姓名和照片)
user_name = ['who','weixin','gaoshine','suowei']
user_img = []
user_encoding =[]

for i in range(len(user_name)):
    mName = user_name[i]
    mImg = face_recognition.load_image_file("./data/%s.jpg" % mName)
    user_img.append(mImg)
    user_encoding.append(face_recognition.face_encodings(mImg)[0])


# Create arrays of known face encodings and their names
known_face_encodings = user_encoding
known_face_names = user_name

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# 采集摄像头
video_capture = cv2.VideoCapture(0)
i = 0
j = 10000
while True:
    # 捕获单帧图像
    ret, frame = video_capture.read()

    # 将原图缩小到1/4大小用于人脸识别
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # 颜色空间转换BGR转换为RGB,opencv使用的是BGR,而face_recognition使用了RGB
    rgb_small_frame = small_frame[:, :, ::-1]

    # 处理每一帧的图像
    if process_this_frame:
        # 从当前帧中查找和识别人脸
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # 试图从已知的人脸库中匹配, matches返回的独热码来比对那个user_name = ['weixin','gaoshine','suowei']
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # 如果匹配到即采用第一个
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # 显示结果
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # 图像尺寸回复原大小(X4)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 在人脸画方框标注
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 在人脸方框上写标注
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # 显示图像
    cv2.imshow('Video', frame)

    i = i + 1
    if i % 25:
        j = j + 1
        # 将原图缩小到1/2大小保存
        save_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.cv2.imwrite("./output/%d.jpg" % j, save_frame)

    # 按 'q' 退出!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
video_capture.release()
cv2.destroyAllWindows()