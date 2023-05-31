# 처음 EAR 평균을 구하는 코드
import time
from time import sleep
import sound_play
import EAR_algorithm
import landmark_detector
import image_converter
import cv2


def setting_normal_head_rate(vid_in):
    print('고개 숙임을 판단하기 위한 얼굴 비율을 측정하겠습니다.')
    sound_play.play_mp3('./alarm/normal_head_rate_start_alarm.mp3')
    sleep(5)
    print('입을 다문 상태로 3초간 정면을 응시해주세요!')
    sound_play.play_mp3('./alarm/normal_head_rate_ing_alarm.mp3')
    sleep(5)
    normal_head_rate = 0
    for i in range(0, 3):
        print(str(i+1) + '초')
        ret, image_o = vid_in.read()
        image = cv2.resize(image_o, dsize=(640, 480),
                           interpolation=cv2.INTER_AREA)
        img_gray = image_converter.get_gray_image(image)
        img_gray_lab = image_converter.get_graylab_image(image, img_gray)
        img_gray = image_converter.get_gray_image(image)
        img_gray_lab = image_converter.get_graylab_image(image, img_gray)
        face_detector = landmark_detector.detector(img_gray_lab, 1)
        nose = landmark_detector.nose_detector(image, face_detector)
        nose_length = landmark_detector.get_face_part_length(nose)
        print('nose_length : ', nose_length)
        nose2chin_length = landmark_detector.nose_tip_to_chin_length(
            image, face_detector)
        print('nose2chin_length : ', nose2chin_length)
        rate = nose_length / nose2chin_length
        rate = round(rate, 2)
        normal_head_rate += rate
    normal_head_rate = normal_head_rate / 3
    normal_head_rate = round(normal_head_rate, 2)
    return normal_head_rate


def setting_mean_EAR_webcam(vid_in):
    sound_play.play_mp3('./alarm/EAR_start_alarm.mp3')
    print('눈 크기 평균을 측정하겠습니다.')
    sleep(3)
    sound_play.play_mp3('./alarm/openEyes_start_alarm.mp3')
    print('6초동안 눈 뜬 상태를 유지해주세요.')
    sleep(3)
    mean_EAR_open = EAR_algorithm.get_mean_EAR(6, vid_in)
    sound_play.play_mp3('./alarm/openEyes_end_alarm.mp3')
    print('눈 뜬 상태의 평균 EAR이 측정되었습니다.')
    print('mean_EAR_open: ', mean_EAR_open)
    sleep(6)

    sound_play.play_mp3('./alarm/closeEyes_start_alarm.mp3')
    print('6초동안 눈 감은 상태를 유지해주세요.')
    sleep(3)
    mean_EAR_close = EAR_algorithm.get_mean_EAR(6, vid_in)
    sound_play.play_mp3('./alarm/closeEyes_end_alarm.mp3')
    print('눈 감은 상태의 평균 EAR이 측정되었습니다.')
    print('mean_EAR_close: ', mean_EAR_close)
    sleep(6)

    EAR_threshold = (mean_EAR_open + mean_EAR_close) / 2
    EAR_threshold = round(EAR_threshold, 2)
    print('EAR threshold : ', EAR_threshold)
    return EAR_threshold


def setting_mean_EAR_video(vid_in):
    R_A = []
    R_B = []
    R_C = []
    L_A = []
    L_B = []
    L_C = []
    for i in range(0, 10):
        ret, image_o = vid_in.read()
        image = cv2.resize(image_o, dsize=(640, 480),
                           interpolation=cv2.INTER_AREA)
        img_gray = image_converter.get_gray_image(image)
        img_gray_lab = image_converter.get_graylab_image(image, img_gray)
        ##########################
        face_detector = landmark_detector.detector(img_gray_lab, 1)
        rightEye, leftEye = landmark_detector.eyes_detector(
            image, face_detector)
        A, B, C = EAR_algorithm.get_threshold_EAR(rightEye)
        R_A.append(A)
        R_B.append(B)
        R_C.append(C)
        A2, B2, C2 = EAR_algorithm.get_threshold_EAR(leftEye)
        L_A.append(A2)
        L_B.append(B2)
        L_C.append(C2)
    EAR_closed_R = (min(R_A) + min(R_B))/(2.0 * max(R_C))
    EAR_open_R = (max(R_A) + max(R_B))/(2.0 * min(R_C))
    EAR_R_threshold = (EAR_closed_R + EAR_open_R) / 2
    print('EAR_R_threshold : ', EAR_R_threshold)
    EAR_closed_L = (min(L_A) + min(L_B))/(2.0 * max(L_C))
    EAR_open_L = (max(L_A) + max(L_B))/(2.0 * min(L_C))
    EAR_L_threshold = (EAR_closed_L + EAR_open_L) / 2
    print('EAR_L_threshold : ', EAR_L_threshold)
    AVG_EAR = (EAR_R_threshold + EAR_L_threshold) / 2
    AVG_EAR = round(AVG_EAR, 2)
    EAR_threshold = AVG_EAR
    return EAR_threshold
