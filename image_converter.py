
import cv2


def get_gray_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img_gray


def get_graylab_image(image, img_gray):
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    img_gray_lab = img_gray + img_lab[:, :, 0]  # 밝기(Lightness)
    img_gray_lab = img_gray_lab + img_lab[:, :, 1]  # 적록(red-green)
    img_gray_lab = img_gray_lab + img_lab[:, :, 2]  # 황청(red-green)
    return img_gray_lab
