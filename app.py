import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Hệ Thống Tầm Soát Ung Thư Phổi", page_icon="🫁", layout="wide")

st.title("🫁 Hệ Thống Chẩn Đoán Khối U Phổi Qua Ảnh X-Quang")
st.write("Đồ án chẩn đoán hình ảnh y tế ứng dụng mô hình Deep Learning YOLOv8")


@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:
    model = load_model()
    st.sidebar.success("Đã tải mô hình (best.pt) thành công!")
except Exception as e:
    st.sidebar.error(f"Không tải được best.pt (để cùng thư mục với app.py): {e}")
    st.stop()  # không có model thì dừng, tránh lỗi NameError phía dưới

st.sidebar.header("Cấu hình mô hình")
# Mặc định thấp hơn 0.25: nốt nhỏ/mờ thường chỉ đạt độ tin cậy 0.1–0.2
conf_threshold = st.sidebar.slider("Ngưỡng tin cậy (Confidence Threshold):", 0.05, 1.0, 0.15, 0.05)
st.sidebar.caption("Ngưỡng thấp: bắt được nhiều nốt hơn nhưng dễ báo nhầm. Ngưỡng cao: ngược lại.")

uploaded_file = st.file_uploader("Tải file ảnh X-quang phổi (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ảnh X-quang Đầu Vào")
        st.image(image, use_container_width=True)

    if st.button("🔍 Tiến hành Chẩn đoán"):
        with st.spinner("AI đang phân tích từng vùng phổi..."):
            results = model.predict(source=image, conf=conf_threshold, imgsz=512, verbose=False)
            boxes = results[0].boxes

        with col2:
            st.subheader("Kết Quả Phân Tích Từ AI")
            # plot() trả về BGR nên phải khai báo channels, nếu không màu bị đảo
            st.image(results[0].plot(), channels="BGR", use_container_width=True)

        st.divider()
        if len(boxes) > 0:
            st.error(f"CẢNH BÁO: Phát hiện {len(boxes)} vị trí nghi ngờ Khối u / Nốt sần (Nodule/Mass)!")
            for idx, box in enumerate(boxes):
                st.write(f"- **Vị trí {idx+1}:** Khối u/Nốt sần - Độ tin cậy: **{float(box.conf[0]) * 100:.2f}%**")
        else:
            st.info(
                f"Mô hình không phát hiện vùng nghi ngờ nào ở ngưỡng tin cậy {conf_threshold:.2f}. "
                "Đây KHÔNG phải kết luận phổi bình thường: nốt nhỏ, mờ hoặc bị xương che có thể bị bỏ sót. "
                "Có thể hạ ngưỡng tin cậy để kiểm tra lại."
            )

        st.caption("Công cụ phục vụ học tập, không thay thế chẩn đoán của bác sĩ.")
