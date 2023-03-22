import cv2
import mediapipe as mp # pip install mediapipe 치면 그냥 바로 설치된다.
import numpy as np
import os
# import tensorflow as tf

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

'''
mediapipe를 사용해 진행하는 경우 단점으로는 눈의 크기가 작다면 동공을 제대로 포착하지 못한다.
장점은 dlib에 비해 cpu 환경에서 작업할 경우 속도가 비교적 빠르다. 또한, 더욱 세밀하게 얼굴을 구분하기 때문에 다른 기능을 추가하기에도 좋다.
'''


mp_face_mash = mp.solutions.face_mesh

#눈과 동공의 인덱스
LEFT_EYE = [362,382,381,374,373,390,248,263,466,388,387,386,385,384,398]
RIGHT_EYE =[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_IRIS = [474,475,476,477]
RIGHT_IRIS = [469,470,471,472]

cap = cv2.VideoCapture("data/videoplayback.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080);

_, check = cap.read()
img_height, img_width = check.shape[:2]
print(img_height, img_width)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 60.0, (int(img_width), int(img_height)))

with mp_face_mash.FaceMesh(max_num_faces=10, refine_landmarks=True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        frame_copy = np.copy(frame)
        result_image = np.copy(frame)
        
        mask = np.zeros((img_height,img_width),np.uint8)
        
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # mp는 RGB 환경에서 작동하기 떄문에 이미지를 바꿔줘야한다.
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mesh_points = np.array([np.multiply([p.x, p.y], [img_width, img_height]) for p in results.multi_face_landmarks[0].landmark], np.int32)
                
                # 동공의 중심, 동공 인덱스로 만든 사각형에 대한 내접원의 반지름을 구한다
                (l_cx, l_cy), l_r = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_r = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                c_l = np.array([l_cx,l_cy], np.int32)
                c_r = np.array([r_cx,r_cy], np.int32)
                cv2.circle(mask, c_l, int(l_r), (255,255,255), -1, cv2.LINE_4)
                cv2.circle(mask, c_r, int(r_r), (255,255,255), -1, cv2.LINE_4)
                
                # 눈동자 크기에 딱 맞는 하얀 mask로 눈을 인식하고 블러처리를 한다.
                # 이때, mask의 흰 부분은 실제 눈 보다 조금 크게 설정해야 블러처리를 한 뒤 영상을 합칠 때 윤곽선이 나타나지 않는다.
                # 딱 맞춰서 블러처리를 할 경우 경계선도 블러 처리가 되어 동공 주변에 흰색 선이 생긴다.
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) # 타원 모양의 커널을 만든다
                mask = cv2.dilate(mask, kernel)                              # 커널로 convolution하여 마스크의 영역을 조금 확장한다.
                eyes = cv2.bitwise_and(frame_copy, frame_copy,mask= mask)    # 원본 영상과 mask를 통해 and 연산을 해서 눈만 보여준다.
                eyes = cv2.medianBlur(eyes, 5)     
                cv2.circle(frame, c_l, int(l_r), (0,0,0), -1, cv2.LINE_4)    # 원본 영상에서 인식되는 범위를 검은색으로 칠한다.
                cv2.circle(frame, c_r, int(r_r), (0,0,0), -1, cv2.LINE_4)
            
            # frame = cv2.erode(frame, kernel)
            result_image = cv2.max(eyes, frame) # 눈만 보여주는 영상과 나머지 부분을 보여주는 영상을 합친다. 이때, add를 사용하면 눈 주변이 뿌옇게 뜬다. max를 사용하면 둘 중 큰 값만 보여줘서 그런 부분이 사라짐
        # else:
            # print('Cannot detect faces.')
            # continue

        # cv2.imshow('img', result_image)   
        # out.write(result_image)
        cv2.imshow('22', eyes)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
    
# tf.test.gpu_device_name()
cap.release()
cv2.destroyAllWindows()
