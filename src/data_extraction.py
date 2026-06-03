import os
import cv2
import csv
import numpy as np
import mediapipe as mp
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


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

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_INNER = [78, 81, 13, 311, 308, 317, 14, 87]

# Đã sửa lại hàm tính toán (Bỏ .landmark)
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

# ----------------------------------------------------------------
ROOT_DIR = "UTA_Data"  
# Đã xóa dấu nháy đơn bị thừa ở cuối chuỗi
OUTPUT_CSV = "data/rldd_geometry_features.csv"

# Đảm bảo thư mục 'data' tồn tại trước khi tạo file CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['EAR', 'MAR', 'Label'])

    print("=== BẮT ĐẦU TIẾN TRÌNH TRÍCH XUẤT ĐẶC TRƯNG HÌNH HỌC ===")
    
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            name_without_ext, ext = os.path.splitext(file)
            ext = ext.lower() 

            if ext in ['.mov', '.mp4']:
                video_path = os.path.join(root, file)
                if name_without_ext == "0":
                    label = 0     
                    frame_step = 30 # Lấy mẫu thưa - tránh trùng lặp dữ liệu mở mắt
                elif name_without_ext == "5" or name_without_ext == "10":
                    label = 1   
                    frame_step = 10 # Lấy mẫu dày hơn để bắt trọn khoảnh khắc nhắm mắt/ngáp ngắn
                else:
                    continue 

                print(f"Đang xử lý: {video_path} -> Gán nhãn: {label}")
                
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                        
                    if frame_count % frame_step == 0:
                        h, w, _ = frame.shape
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Chuyển sang định dạng mp.Image và chạy FaceLandmarker
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        results = face_landmarker.detect(mp_image)
                        
                        # Cấu trúc đọc results của Tasks API
                        if results.face_landmarks:
                            landmarks = results.face_landmarks[0]
                            
                            # tính toán các chỉ số hình học
                            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
                            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
                            ear = (left_ear + right_ear) / 2.0 
                            mar = calculate_mar(landmarks, MOUTH_INNER, w, h)
                            
                            if label == 1:
                                # Trong video buồn ngủ, ta chỉ giữ lại những khung hình tài xế
                                # sụp mí thật sự (EAR thấp) hoặc đang ngáp thật sự (MAR cao)
                                if ear <= 0.23 or mar >= 0.6:
                                    writer.writerow([round(ear, 4), round(mar, 4), label])
                            else:
                                writer.writerow([round(ear, 4), round(mar, 4), label])
                                
                    frame_count += 1
                cap.release()

print(f"\n✅ Quá trình xử lý hoàn tất thành công!")
print(f"📁 File dữ liệu tổng hợp phục vụ huấn luyện đã lưu tại: {OUTPUT_CSV}")