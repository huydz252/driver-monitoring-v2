import os
import cv2
import pickle
import numpy as np
import json
import logging
import urllib.request
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_from_directory, send_file

#chặn spam log terminal
class NoStatusFilter(logging.Filter):
    def filter(self, record):
        return '/api/status' not in record.getMessage()
logging.getLogger("werkzeug").addFilter(NoStatusFilter())

# 1. KHỞI TẠO FLASK VỚI ĐƯỜNG DẪN THƯ MỤC TUYỆT ĐỐI
app = Flask(__name__, 
            template_folder=os.path.abspath('templates'),
            static_folder=os.path.abspath('static'))

# 2. NẠP MÔ HÌNH HỌC MÁY SVM VÀ SCALER CỦA NHÓM
with open('models/drowsiness_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 3. KHỞI TẠO MEDIAPIPE THEO CHUẨN TASKS API (V0.10+)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = 'face_landmarker.task'

# Kiểm tra và tự động tải file face_landmarker.task từ Google nếu chưa có
if not os.path.exists(MODEL_PATH):
    if os.path.exists('../' + MODEL_PATH):
        MODEL_PATH = '../' + MODEL_PATH
    else:
        print("📥 Đang tải face_landmarker.task từ bộ lưu trữ của Google...")
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

# 4. QUY ƯỚC CHỈ SỐ MỐC LANDMARK HÌNH HỌC THEO CODE CỦA NHÓM
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_INNER = [78, 81, 13, 311, 308, 317, 14, 87]

# 5. CÁC HÀM TOÁN HỌC TÍNH TOÁN CHỈ SỐ SINH TRẮC HỌC
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

# 6. BIẾN TRẠNG THÁI TOÀN CỤC ĐỒNG BỘ ĐỮ LIỆU
DROWSIER_FRAME_THRES = 20
COUNTER = 0
CURRENT_STATUS = "tỉnh táo"
CURRENT_EAR = 0.3
CURRENT_MAR = 0.0

# log
last_alert_time = None
current_dir = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(current_dir, '..', 'log')
LOG_FILE_PATH = os.path.join(LOG_DIR, 'alert_log.jsonl')

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

last_alert_time = None

def log_alert(alert_type, level):
    global last_alert_time
    now = datetime.now()
    
    # Chỉ ghi log nếu cách lần cảnh báo trước đó ít nhất 10 giây
    if last_alert_time is None or (now - last_alert_time).total_seconds() >= 10:
        last_alert_time = now
        
        # Tạo bản ghi dữ liệu
        record = {
            "time": now.strftime("%d/%m/%Y %H:%M:%S"),
            "type": alert_type,
            "level": level
        }
        
        # Mở file mode 'a' (append) để ghi nối thêm 1 dòng vào cuối file
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

camera = cv2.VideoCapture(0)

# 7. HÀM XỬ LÝ QUÉT VIDEO VÀ PHÂN LOẠI TRẠNG THÁI BẰNG SVM
def generate_frames():
    global COUNTER, CURRENT_STATUS, CURRENT_EAR, CURRENT_MAR
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1) # Lật gương hình ảnh
        h, w, _ = frame.shape
        
        # Tiền xử lý chuyển đổi kênh màu sang RGB cho MediaPipe Tasks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Thực hiện định vị mốc mặt qua Tasks API
        results = face_landmarker.detect(mp_image)
        
        status_text = "TRANG THAI: TINH TAO"
        color = (0, 255, 0)
        temp_status = "tỉnh táo"
        ear_val, mar_val = 0.3, 0.0

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            
            # Tính toán các chỉ số hình học
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            ear_val = (left_ear + right_ear) / 2.0
            mar_val = calculate_mar(landmarks, MOUTH_INNER, w, h)
            
            # Chuẩn hóa Vector đặc trưng qua bộ Scaler
            features = np.array([[ear_val, mar_val]])
            features_scaled = scaler.transform(features)
            
            # Mô hình SVM tiến hành phân loại nhãn quyết định
            prediction = model.predict(features_scaled)[0]
            
            if prediction == 1:
                COUNTER += 1
                if COUNTER >= DROWSIER_FRAME_THRES:
                    status_text = "NGUY HIEM: BUON NGU!!!"
                    color = (0, 0, 255)
                    temp_status = "buồn ngủ"
                    log_alert("Nhắm mắt quá lâu (Nguy hiểm)", "danger")
                else:
                    status_text = "CANH BAO: HOI MET MOI"
                    color = (0, 165, 255)
                    temp_status = "mệt mỏi"
            else:
                COUNTER = 0
                status_text = "TRANG THAI: TINH TAO"
                color = (0, 255, 0)
                temp_status = "tỉnh táo"
                
            # Vẽ thông điệp trạng thái lên khung hình video
            cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"EAR: {ear_val:.2f} | MAR: {mar_val:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Cập nhật giá trị phục vụ API
        CURRENT_EAR = round(ear_val, 3)
        CURRENT_MAR = round(mar_val, 3)
        CURRENT_STATUS = temp_status

        # Đóng gói frame định dạng JPG chuyển tiếp luồng dữ liệu
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# 8. ĐIỀU PHỐI ĐƯỜNG DẪN (ROUTES) ỨNG DỤNG
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    global COUNTER, CURRENT_STATUS, CURRENT_EAR, CURRENT_MAR
    return jsonify({
        "ear": CURRENT_EAR,
        "mar": CURRENT_MAR,
        "status": CURRENT_STATUS,
        "counter": COUNTER,
        "danger": True if COUNTER >= DROWSIER_FRAME_THRES else False
    })

@app.route('/sound/warning5.mp3')
def serve_sound():
    # 1. Lấy đường dẫn tuyệt đối của thư mục chứa file app.py hiện tại (thư mục src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Nối chuỗi để đi tới thư mục sound (src -> lùi 1 bước -> sound -> warning5.mp3)
    sound_path = os.path.join(current_dir, '..', 'sound', 'warning5.mp3')
    
    # 3. Gửi thẳng file này về cho Front-end bằng send_file
    return send_file(sound_path, mimetype='audio/mpeg')

@app.route('/history.html')
def history_page():
    return render_template('history.html')

@app.route('/api/history')
def get_history():
    history_data = []
    
    if os.path.exists(LOG_FILE_PATH):
        # Mở file đọc từng dòng
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): # Bỏ qua dòng trống
                    history_data.append(json.loads(line.strip()))
                    
    history_data.reverse()
    
    # Trả về tối đa 100 dòng gần nhất
    return jsonify(history_data[:100])

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    global last_alert_time
    last_alert_time = None
    # Mở file mode 'w' để xóa trắng dữ liệu
    with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
        pass
        
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)