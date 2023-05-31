
from scipy.spatial import distance
import math
import numpy as np
import cv2
import sound_play
import image_converter
import landmark_detector
import setting
import EAR_algorithm
import driver_state

yawn_count = 0


idx = 1

# vid_in = cv2.VideoCapture('h_phone_night.mp4')
vid_in = cv2.VideoCapture(0)  # 인자로 0 전달하면 웹캠
# video
# EAR_threshold = setting.setting_mean_EAR_video(vid_in)
# webcam
EAR_threshold = setting.setting_mean_EAR_webcam(vid_in)
# EAR_threshold = 0.29
normal_head_rate = setting.setting_normal_head_rate(vid_in)
print('EAR_threshold : ', EAR_threshold)
print('normal_head_rate : ', normal_head_rate)
while True:
    ret, image_o = vid_in.read()

    # 이미지 크기 조정
    image = cv2.resize(image_o, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    # 영상 좌우 반전
    image = cv2.flip(image, 1)
    img_gray = image_converter.get_gray_image(image)
    img_gray_lab = image_converter.get_graylab_image(image, img_gray)
    ##########################
    face_detector = landmark_detector.detector(img_gray_lab, 1)  # night
    # face_detector = landmark_detector.detector(img_gray, 1)

    cv2.putText(image, "EAR_threshold : ", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    cv2.putText(image, str(EAR_threshold), (230, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    rightEye, leftEye = landmark_detector.eyes_detector(image, face_detector)
    top_lip, bottom_lip = landmark_detector.mouth_detector(
        image, face_detector)
    # 입술 내부 중 가운데
    top_lip_mid = top_lip[8]
    bottom_lip_mid = bottom_lip[9]

    distance_mouth = distance.euclidean(top_lip_mid, bottom_lip_mid)
    cv2.line(image, top_lip_mid, bottom_lip_mid, (255, 255, 0), 1)
    #################################
    nose = landmark_detector.nose_detector(image, face_detector)
    nose_length = landmark_detector.get_face_part_length(nose)
    nose2chin_length = landmark_detector.nose_tip_to_chin_length(
        image, face_detector)
    rate = nose_length / nose2chin_length

    if distance_mouth > 13:
        driver_state.yawn()
        print(f'yawn count: {driver_state.yawn.count}')
        if driver_state.yawn.count >= 5:
            cv2.putText(image, "YAWN~~", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            yawn_count += 1
        if (yawn_count + 1) % 15 == 0:
            sound_play.yawn_warning_sound()
            yawn_count = 0

    right_ear = EAR_algorithm.calculate_EAR(rightEye)
    left_ear = EAR_algorithm.calculate_EAR(leftEye)

    EAR = (right_ear+left_ear)/2
    EAR = round(EAR, 2)
    # print('EAR: ', EAR)
    if EAR < EAR_threshold:
        driver_state.close(image)
        print(f'close count: {driver_state.close.count}')
        if driver_state.close.count == 10:
            driver_state.sleep_state(image)
            sound_play.sleep_warning_sound()
        elif driver_state.close.count == 4 and rate > normal_head_rate * 1.32:
            driver_state.sleep_state(image)
            print('고개 숙임 감지 & 졸음 감지')
            sound_play.sleep_warning_sound()
    cv2.imshow('result', image)
    key = cv2.waitKey(1)

    if key == 27:
        break

vid_in.release()
cv2.destroyAllWindows()
