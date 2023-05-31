
import time
from functools import wraps
import cv2

lastsave = 0  # 전역변수


def counter(func):
    @wraps(func)
    def tmp(*args, **kwargs):   # args : 튜플 형태로 값 저장, kwargs : dictionary 형태로 값 저장하고 파라미터 명을 같이 보낼 수 있음 => 둘다 가변인자를 위한 변수
        tmp.count += 1
        time.sleep(0.01)
        global lastsave
        if time.time() - lastsave > 5:
            lastsave = time.time()
            tmp.count = 0
        return func(*args, **kwargs)
    tmp.count = 0
    return tmp


@counter
def close(image):
    cv2.putText(image, "BLINKING", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    pass


def sleep_state(image):
    cv2.putText(image, "zzz~~", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    pass


@counter
def yawn():
    pass
