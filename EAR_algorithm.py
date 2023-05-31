from scipy.spatial import distance
import cv2
import image_converter
import landmark_detector

# EAR 알고리즘


def get_threshold_EAR(eyes):
    A = distance.euclidean(eyes[1], eyes[5])
    B = distance.euclidean(eyes[2], eyes[4])
    C = distance.euclidean(eyes[0], eyes[3])
    return A, B, C


def calculate_EAR(eye):
    A, B, C = get_threshold_EAR(eye)
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio


def get_mean_EAR(sec, vid_in):
    mean_right_ear = 0
    mean_left_ear = 0
    for i in range(0, sec):
        print(str(i+1) + '초')
        ret, image_o = vid_in.read()
        image = cv2.resize(image_o, dsize=(640, 480),
                           interpolation=cv2.INTER_AREA)
        img_gray = image_converter.get_gray_image(image)
        img_gray_lab = image_converter.get_graylab_image(image, img_gray)
        ##########################
        face_detector = landmark_detector.detector(img_gray_lab, 1)
        rightEye, leftEye = landmark_detector.eyes_detector(
            image, face_detector)
        #face_detector = landmark_detector.detector(img_gray, 1)
        #rightEye, leftEye = landmark_detector.eyes_detector(image, face_detector)

        mean_right_ear += calculate_EAR(rightEye)
        mean_left_ear += calculate_EAR(leftEye)
    mean_right_ear = mean_right_ear / sec
    mean_left_ear = mean_left_ear / sec
    mean_EAR = (mean_right_ear + mean_left_ear) / 2
    mean_EAR = round(mean_EAR, 2)
    return mean_EAR
