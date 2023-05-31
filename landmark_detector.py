import dlib
import cv2
import image_converter
# create face detector, predictor
detector = dlib.get_frontal_face_detector()  # dlib에 있는 정면 얼굴 검출기 (얼굴 전체)
predictor = dlib.shape_predictor(
    'shape_predictor_68_face_landmarks.dat')  # 얼굴 내 랜드마크

# 개 중요!!!!!!!!!!!!!!!!!!!!!


def get_landmarks(original_image, face_detector):
    gray_img = image_converter.get_gray_image(original_image)
    graylab_img = image_converter.get_graylab_image(original_image, gray_img)
    for face in face_detector:
        # landmarks = predictor(original_image, face)  # 얼굴에서 68개 점 찾기
        landmarks = predictor(original_image, face)  # 얼굴에서 68개 점 찾기
    return landmarks


def get_face_part(landmarks, original_image, index1, index2):
    face_part = []
    for n in range(index1, index2):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        face_part.append((x, y))
        next_point = n+1
        if n == index2-1:
            next_point = index1
        x2 = landmarks.part(next_point).x
        y2 = landmarks.part(next_point).y
        cv2.line(original_image, (x, y), (x2, y2), (255, 255, 0), 1)
        #cv2.circle(original_image,(x2,y2),2,(0x255,0),-1)
    return face_part


def eyes_detector(original_image, face_detector):
    landmarks = get_landmarks(original_image, face_detector)
    rightEye = []
    leftEye = []
    rightEye = get_face_part(landmarks, original_image, 36, 42)
    leftEye = get_face_part(landmarks, original_image, 42, 48)
    return rightEye, leftEye


def mouth_detector(original_image, face_detector):
    landmarks = get_landmarks(original_image, face_detector)
    mouthOutline = []
    mouthInline = []
    mouthOutline = get_face_part(landmarks, original_image, 48, 60)
    mouthInline = get_face_part(landmarks, original_image, 60, 68)
    top_lip = []
    for i in range(0, 7):
        top_lip.append(mouthOutline[i])
    for i in range(3, -1, -1):
        top_lip.append(mouthInline[i])
    bottom_lip = []
    bottom_lip.append(mouthOutline[0])
    for i in range(11, 5, -1):
        bottom_lip.append(mouthOutline[i])
    for i in range(4, 8):
        bottom_lip.append(mouthInline[i])

    return top_lip, bottom_lip


def get_face_part_length(face_part):
    start_face_part = 0
    end_face_part = len(face_part)-1
    length = abs(face_part[start_face_part][1] - face_part[end_face_part][1])
    return length


def nose_detector(original_image, face_detector):
    landmarks = get_landmarks(original_image, face_detector)
    nose = []
    nose = get_face_part(landmarks, original_image, 27, 34)
    return nose


def nose_tip_to_chin_length(original_image, face_detector):
    landmarks = get_landmarks(original_image, face_detector)
    nose_tip_idx = 33
    x1 = landmarks.part(nose_tip_idx).x
    y1 = landmarks.part(nose_tip_idx).y
    chin_idx = 8
    x2 = landmarks.part(chin_idx).x
    y2 = landmarks.part(chin_idx).y
    return abs(y1 - y2)
