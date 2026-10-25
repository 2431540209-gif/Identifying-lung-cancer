from ultralytics import YOLO
import cv2

# 1. Load mô hình
model = YOLO('best.pt')

# 2. Thay đường dẫn bên dưới bằng file ảnh nằm trong tập yolo_dataset/images/val/
image_path = "1.png" 

# 3. Chạy dự đoán với ngưỡng conf thấp (0.1)
results = model.predict(source=image_path, conf=0.10, imgsz=512)

# 4. Hiển thị số lượng ô phát hiện được
boxes = results[0].boxes
print(f"Số lượng khối u phát hiện được: {len(boxes)}")

for box in boxes:
    print(f"- Độ tin cậy: {float(box.conf[0])*100:.2f}% | Tọa độ: {box.xyxy[0].tolist()}")

# 5. Mở hình vẽ kết quả
res_img = results[0].plot()
cv2.imshow("Ket qua Test AI", res_img)
cv2.waitKey(0)
cv2.destroyAllWindows()