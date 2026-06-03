import cv2
import mediapipe as mp
import pickle
import numpy as np
from pygame import mixer 
import os
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

with open('models/drowsiness_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f) 

#sound
mixer.init()
try:
    mixer.music.load('sound/warning5.mp3')
except Exception as e:
    print(f"Lỗi nạp file âm thanh: {e}")

MODEL_PATH = 'face_landmarker.task'
if not os.path.exists(MODEL_PATH):
    if os.path.exists('../' + MODEL_PATH):
        MODEL_PATH = '../' + MODEL_PATH
    else:
        print("📥 Đang tải face_landmarker.task...")
        urllib.request.urlretrieve(
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            MODEL_PATH
        )
        print("✅ Tải thành công model định vị!")

base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_opts,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
# =================================================================

# Cấu hình các chỉ số Landmark
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_INNER = [78, 81, 13, 311, 308, 317, 14, 87]

# Đã sửa lại hàm tính toán (bỏ .landmark)
def calculate_ear(landmarks, indices, w, h):
    coords = [np.array([landmarks[idx].x * w, landmarks[idx].y * h]) for idx in indices]
    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    C = np.linalg.norm(coords[0] - coords[3])
    return (A + B) / (2.0 * C)

def calculate_mar(landmarks, indices, w, h):
    coords = [np.array([landmarks[idx].x * w, landmarks[idx].y * h]) for idx in indices]
    A = np.linalg.norm(coords[1] - coords[7])
    B = np.linalg.norm(coords[2] - coords[6])
    C = np.linalg.norm(coords[3] - coords[5])
    D = np.linalg.norm(coords[0] - coords[4])
    return (A + B + C) / (2.0 * D)

DROWSIER_FRAME_THRES = 20
COUNTER = 0 

cap = cv2.VideoCapture(0)
print("------ ĐANG BẬT WEBCAM HỆ THỐNG CẢNH BÁO REAL-TIME ------")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Ép kiểu dữ liệu chuẩn Tasks API và gọi hàm detect
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_landmarker.detect(mp_image)
    
    # Cấu trúc đọc kết quả mới
    if results.face_landmarks:
        landmarks = results.face_landmarks[0]
        
        # Tính toán EAR và MAR từ luồng camera trực tiếp
        left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
        right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar(landmarks, MOUTH_INNER, w, h)
        
        features = np.array([[ear, mar]])
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        
        if prediction == 1:
            COUNTER += 1
            if COUNTER >= DROWSIER_FRAME_THRES:
                status_text = "NGUY HIEM: BUON NGU!!!"
                color = (0, 0, 255)
                if not mixer.music.get_busy():
                    mixer.music.play(-1)  
            else:
                status_text = "CANH BAO: HOI MET MOI"
                color = (0, 165, 255) 
        else:
            COUNTER = 0 
            status_text = "TRANG THAI: TINH TAO"
            color = (0, 255, 0)
            mixer.music.stop()
            
        cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}  |  MAR: {mar:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frames nham mat: {COUNTER}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Driver Drowsiness Detection System', frame)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
mixer.music.stop()