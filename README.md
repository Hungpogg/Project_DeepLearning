# Phân đoạn ảnh giao thông — Nhóm 9

Tiểu luận cuối kỳ môn học Deep Learning: ứng dụng EfficientNet trong kiến trúc Encoder–Decoder.

**Trần Thanh Hùng — Nguyễn Xuân Lâm**

## Demo trên Streamlit Community Cloud

1. Đăng nhập https://share.streamlit.io bằng GitHub.
2. Chọn Create app → repository `Hungpogg/Project_DeepLearning`.
3. Branch: `main`; Main file path: `streamlit_app.py`.
4. Trong Advanced settings chọn Python **3.12**, rồi Deploy.
5. Chia sẻ địa chỉ `.streamlit.app` sau khi app chạy thành công.

Không cần secrets hay API key. Máy chủ dùng PyTorch CPU và checkpoint B3 trong repository, không tải pretrained weights. Model được nạp dùng chung giữa các phiên; suy luận được tuần tự hóa để giảm bộ nhớ đỉnh. Giới hạn 10 ảnh/lượt và 20 MB/file. Cần kiểm tra tài nguyên và thời gian xử lý thực tế trên gói hosting.

## Cách sử dụng

- Tải một hoặc nhiều ảnh lên, chọn ảnh ghép Cityscapes (trái RGB, phải mask) hoặc RGB riêng.
- Nhấn **Chạy segmentation**, xem ảnh dự đoán, overlay và nhãn từng pixel.
- Tab **Phân bố lớp** có IoU (%) từng lớp khi ảnh ghép có ground truth: `100 × TP / (TP + FP + FN)` trên lưới 256×256. Lớp vắng trong cả hai mask hoặc ảnh không có ground truth thì không có IoU.
- Tải mask hoặc ZIP kết quả. Ảnh được gửi tới máy chủ để xử lý; app không chủ động ghi ảnh tải lên ra đĩa.

## Chạy trên máy cá nhân

Xem [README_APP.md](README_APP.md). `app.py` giữ chức năng chọn checkpoint B0–B7 và thư mục cục bộ; `streamlit_app.py` chỉ dùng model B3 cố định và nhận ảnh tải lên.

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

Model B3 là checkpoint được chọn cho demo, không khẳng định là biến thể tốt nhất trong toàn bộ B0–B7.
