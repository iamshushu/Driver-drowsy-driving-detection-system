# Driver-drowsy-driving-detection-system


👩‍👩‍👧 Team Project

💻 실시간 영상에서 얼굴 특징점 기반의 졸음운전 감지 알고리즘 개발

💡 졸음 인식 방법
  1. 눈 졸음 상태 판단

  2. 하품 판단

  3. 고개 숙임 판단

💡 사람마다 다른 얼굴의 특징점을 고려한 적응형 졸음 감지 알고리즘 개발
> **개발 환경**
> 
- Python 3.10.5
- OpenCV 4.6.0
Dlib 19.24

> **시스템 동작 순서**
> 

![Untitled (3)](https://github.com/iamshushu/Driver-drowsy-driving-detection-system/assets/78261259/5c109e9b-8b62-45ed-a723-5742c94a3682)

> **Dlib을 이용한 얼굴의 landmark 검출**
> 
- Dlib: OpenCV 기반의 딥러닝 라이브러리
- Dlib 학습 모델 데이터 사용
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- 68개의 얼굴의 특징점(facial landmarks) 검출

![Untitled (4)](https://github.com/iamshushu/Driver-drowsy-driving-detection-system/assets/78261259/13153f97-5fb0-411e-9757-1b471a2913ee)


> **1. 눈 졸음 상태 판단 - EAR(Eye Aspect Ratio) 이용**
> 

![Untitled (5)](https://github.com/iamshushu/Driver-drowsy-driving-detection-system/assets/78261259/084367b6-628c-477f-9617-cd44d0cbd9e6)

- 눈의 landmark 좌표값을 얻음 → 각 눈은 6개의 (x, y) 좌표로 표현됨
- 눈의 비율을 이용해 눈 감김 확인
    - 오른쪽 눈: 36 - 41, 왼쪽 눈: 42 - 47
    
    ```python
    **[Problem]** : 사람의 눈 크기는 전부 다름
    **고정된 임계값(threshold)**을 사용하면, 
    눈을 뜨고 있음에도 눈을 감고있다고 인식하는 문제 발생
    ```
    
    ```python
    **[Solution]
    적응형 EAR 임계값(threshold)** 사용
    1. 시스템을 시작하면 사용자에게 눈을 편안하게 뜬 상태를 6초 동안 유지하도록 안내
    2. 눈 뜬 상태의 평균 EAR 측정 (6초간)
    3. 같은 방법으로 눈 감은 상태의 평균 EAR 측정 (6초간)
    → **두 EAR 값의 평균 값을 눈의 임계값(threshold)으로 사용**
    ```
    

- **[눈 졸음 상태 판단 알고리즘]**
    
    ![Untitled (6)](https://github.com/iamshushu/Driver-drowsy-driving-detection-system/assets/78261259/1a80d589-a3a4-4412-b6fc-be1183c4fe67)
    
    ```
    - 운전자의 실시간 EAR 값이 구한 임계값보다 작은 경우 눈을 감은 상태로 인식
    - 운전자의 실시간 EAR 값이 구한 임계값보다 큰 경우 눈을 뜬 상태로 인식
    - 눈을 감은 상태가 일정 시간 이상 유지되는 경우 운전자에게 상태 경고
    ```
    

> **2. 하품 상태 판단 - EAR(Eye Aspect Ratio) 알고리즘**
> 
- 입의 landmark 좌표값을 얻음
    - outline: 48 - 59, inline: 60 - 67
- 입술의 inline의 가장 높은 좌표값과 가장 낮은 좌표값을 이용해 입의 높이 측정

- **[하품 상태 판단 알고리즘]**
    
    ```
    - 임계값(threshold)보다 입의 높이가 큰 경우 하품으로 인식
    - 하품 검출 빈도가 일정 횟수를 넘는 경우 운전자에게 상태 경고
    ```
    

> **3. 고개 숙임 판단**
> 
- 눈을 감은 상태로 고개를 숙이면 졸음 상태로 인식
- 눈 감김만 탐지할 때보다 눈 감김 시간이 짧아도 고개 숙임 시 졸음 상태일 확률↑
    - x: 코의 시작 - 코 끝의 거리(27 - 30)
    - y: 코 끝 - 턱의 거리(30 - 8)
- **[고개숙임 판단 알고리즘]**
    
    ![Untitled (7)](https://github.com/iamshushu/Driver-drowsy-driving-detection-system/assets/78261259/735f63f7-90f9-44c0-82c5-7680726b681d)
    
    ```
    평상시 y대한 x의 비를 기준으로 1.15배 이상의 비가 측정되었을 때, 
    고개 숙임으로 인식
    ```
    

> **어두운 환경(야간 주행)의 경우도 고려**
> 

```
아래 그림은 야간 운전 상황을 고려한 어두운 영상으로부터의 눈과 입 인식 결과를 보여준다. 
그림(a) : 카메라로부터 받은 영상 이미지를 gray-scale 이미지만 합성
그림(b) : 카메라로부터 받은 영상 이미지를 gray-scale 이미지 + LAB 이미지 합성

**[gray-scale 이미지만 합성한 경우의 문제점]**
(a)의 경우, 얼굴의 landmark 인식률이 떨어짐을 확인
눈의 EAR 임계값을 측정한 결과가 크게 차이나고, 눈의 감김 상태 역시 인식하지 못함

**[해결방법]
gray-scale 이미지 + LAB 이미지 합성하는 방법을 이용**
(b)의 결과 화면을 보면, 밝기로 인한 졸음 상태 검출의 오차를 줄일 수 있음을 확인
```

(a) Gray Scale Image

![Untitled (8)](https://github.com/iamshushu/Driver-drowsy-driving-detection-system/assets/78261259/a4cbb336-4192-43ad-bc30-4fd1314fe204)

(b) Gray Scale Image + LAB Image

![Untitled (9)](https://github.com/iamshushu/Driver-drowsy-driving-detection-system/assets/78261259/e8e28017-0f6a-4901-b159-0c2752adbe73)


> 개선사항
> 

```
1. 고개를 돌려서 얼굴 인식이 되지 않은 경우, 손으로 얼굴을 가리는 경우
	 landmark를 탐지할 수 없어서 시스템 종료됨
2. 현재 시스템은 졸음 상태로 인식되면 경고 안내음이 재생됨 
	 → 운전자의 졸음 경고 효과에 한계가 있음
	 운전자의 좌석에 진동기 연결 or 자동으로 창문 열리기 or 라디오 틀어주기 등의 기능 추가 
   → 본 프로젝트의 실용성을 높일 수 있음
```
