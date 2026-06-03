import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle

print("Đang tải tập dữ liệu đặc trưng hình học...")
df = pd.read_csv('data/rldd_geometry_features.csv')

#loại bỏ data lỗi
df = df.dropna()

X = df[['EAR', 'MAR']].values
y = df['Label'].values

print(f" Tổng số mẫu dữ liệu đọc được: {len(df)} dòng.")
print(f"Cơ cấu nhãn: \n- Nhãn 0 (Tỉnh táo): {np.sum(y == 0)} mẫu\n- Nhãn 1 (Buồn ngủ): {np.sum(y == 1)} mẫu")



# Chia dữ liệu: 80% train - 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



print("\n[INFO] Đang tiến hành huấn luyện mô hình SVM với RBF Kernel...")
# Khởi tạo thuật toán SVM, dùng kernel 'rbf' để phân tách ranh giới phi tuyến phức tạp
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_train_scaled, y_train)
print(" Huấn luyện hoàn tất!")




y_pred = model.predict(X_test_scaled)

print("\n[INFO] Đang đóng gói và lưu mô hình xuống ổ cứng...")
# Lưu mô hình toán học SVM
with open('models/drowsiness_svm_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file) 

# Lưu cả bộ chuẩn hóa Scaler để khi chạy camera thực tế cũng scale dữ liệu theo tỷ lệ tương tự
with open('models/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print(" Thành công! Đã sinh ra file 'drowsiness_svm_model.pkl' và 'scaler.pkl' trong thư mục models.")