# Phân đoạn ảnh giao thông — Nhóm 9

Tiểu luận cuối kỳ môn học Deep Learning: ứng dụng EfficientNet trong kiến trúc Encoder–Decoder.

**Trần Thanh Hùng — Nguyễn Xuân Lâm**

**Mở ứng dụng:** https://nhom9-efficientnet-unet.streamlit.app/

## Demo trên Streamlit Community Cloud

1. Đăng nhập https://share.streamlit.io bằng GitHub.
2. Chọn Create app → repository `Hungpogg/Project_DeepLearning`.
3. Branch: `main`; Main file path: `streamlit_app.py`.
4. Trong Advanced settings chọn Python **3.12**, rồi Deploy.
5. Chia sẻ địa chỉ `.streamlit.app` sau khi app chạy thành công.

Không cần secrets hay API key. Bản cloud dùng cùng giao diện và luồng xử lý với localhost: chọn checkpoint có sẵn, nhập đường dẫn hoặc tải .pth/.pt; chọn encoder B0–B7; tải ảnh hoặc đọc thư mục. Giới hạn 100 ảnh/lượt và 512 MB/file như bản local. Model được giữ riêng theo phiên người dùng. Máy chủ hiện có checkpoint B3 và chạy CPU; các checkpoint B0–B7 khác có thể tải lên. Bộ nhớ thực tế phụ thuộc gói hosting và kích thước model.

## Cách sử dụng

- Tải một hoặc nhiều ảnh lên, chọn ảnh ghép Cityscapes (trái RGB, phải mask) hoặc RGB riêng.
- Nhấn **Chạy segmentation**, xem ảnh dự đoán, overlay và nhãn từng pixel.
- Tab **Phân bố lớp** có IoU (%) từng lớp khi ảnh ghép có ground truth: `100 × TP / (TP + FP + FN)` trên lưới 256×256. Lớp vắng trong cả hai mask hoặc ảnh không có ground truth thì không có IoU.
- Tải mask hoặc ZIP kết quả. Ảnh được gửi tới máy chủ để xử lý; app không chủ động ghi ảnh tải lên ra đĩa.

## Chạy trên máy cá nhân

Xem [README_APP.md](README_APP.md). `streamlit_app.py` chạy trực tiếp cùng `app.py`, không ẩn hoặc thay đổi các tùy chọn của bản local.

Trên cloud, ô đường dẫn checkpoint và thư mục đọc từ **máy chủ**, không đọc ổ đĩa máy người xem. Để dùng file trên máy cá nhân, chọn **Tải file .pth** và **Tải một hoặc nhiều ảnh**.

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

Model B3 là checkpoint được chọn cho demo, không khẳng định là biến thể tốt nhất trong toàn bộ B0–B7.
