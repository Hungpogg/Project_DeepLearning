# Cityscapes Segmentation Studio

App Streamlit chạy cục bộ để xem định tính UNet + EfficientNet B0–B7 từ notebook của dự án.

## Chạy app trên Windows

Yêu cầu Python 3.10–3.12, khuyến nghị 3.12. Mở PowerShell tại thư mục dự án:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Sau khi cài đặt, những lần sau chỉ chạy lệnh cuối, hoặc:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_app.ps1
```

Mở http://localhost:8501. App chỉ lắng nghe trên `127.0.0.1`. Cài thư viện cần Internet; inference không tải pretrained weights. Trên macOS/Linux, dùng `python3 -m venv .venv` và `.venv/bin/python` thay đường dẫn Windows.

## Sử dụng

1. Chọn checkpoint có sẵn trong `results_w_EfficientNetB*/best_model.pth`, nhập đường dẫn khác hoặc tải `.pth`/`.pt` lên (tối đa 512 MB/file).
2. Chọn **Tự nhận diện B0–B7**, hoặc chỉ định đúng encoder đã train. Bấm **Nạp model** để kiểm tra; nút chạy segmentation cũng tự nạp nếu cần.
3. Chọn định dạng **Cityscapes ghép ảnh + mask** cho ảnh train/val pix2pix: trái là ảnh RGB, phải là mask màu, tỷ lệ toàn ảnh 2:1. Chọn **Ảnh RGB riêng** cho ảnh không ghép, kể cả ảnh phong cảnh tỷ lệ 2:1; app không tự đoán định dạng.
4. Tải một/nhiều ảnh JPG/PNG hoặc nhập thư mục train/val trên máy chạy app, đọc danh sách và chọn ảnh. Mỗi lần tối đa 100 ảnh, xử lý từng ảnh để hạn chế VRAM.
5. Nhấn **Chạy segmentation**, chọn từng ảnh trong danh sách kết quả. Rê chuột, zoom hoặc nhập tọa độ X/Y để xem class ID, tên lớp và ground truth.
6. Dùng tab **Phân bố lớp** xem bảng màu, số pixel và IoU từng lớp; **Tải kết quả** xuất mask ID riêng hoặc ZIP toàn bộ lượt chạy.

Thay model/ảnh trên giao diện không tự chạy inference. Kết quả luôn ghi rõ checkpoint của lần chạy đã hoàn tất; nhấn nút chạy để tạo lượt mới.

## Tương thích notebook

- Kiến trúc: `smp.Unet`, decoder batch normalization, đầu ra 20 lớp, `activation=None`.
- Checkpoint hỗ trợ dictionary chứa `model_state_dict`, `state_dict`, hoặc state dict trực tiếp; hỗ trợ tiền tố `module.` của DataParallel.
- Nạp bằng `weights_only=True`, `strict=True`. Không tự bỏ qua lớp sai kích thước hoặc tải một phần trọng số. Nếu file cũ chứa đối tượng Python không được hỗ trợ, xuất lại checkpoint inference trong môi trường train như cell `safe_checkpoint` của notebook.
- Tự nhận diện encoder dựa trên kích thước tensor và số block, sau đó kiểm tra toàn bộ state dict.
- Tiền xử lý giống validation: RGB → resize OpenCV bilinear 256×256 → ImageNet mean/std. Mask dùng nearest-neighbor. Không augmentation khi inference.
- Ground truth: ánh xạ RGB sang palette gần nhất theo khoảng cách L1, giống notebook. Đây là nhãn giải mã từ ảnh màu; không thay thế label ID chính thức.
- Class 0 `unlabeled` vẫn là lớp dự đoán; bảng màu 0–19 khớp notebook.
- Chạy FP32 và `eval()`/`inference_mode()`. So với AMP trên Kaggle, pixel gần ranh giới quyết định có thể khác nhẹ.
- Mọi mask và tọa độ xuất ra dùng lưới **256×256**, không phải độ phân giải ảnh gốc. Ảnh RGB khác tỷ lệ sẽ được resize về vuông theo pipeline train.
- IoU từng lớp trên ảnh đang xem: `100 × TP / (TP + FP + FN)` (%), so sánh mask dự đoán với ground truth ở lưới 256×256, gồm cả lớp 0. Không có ground truth hoặc lớp vắng ở cả hai mask thì để trống; lớp chỉ có ở một mask có IoU bằng 0. Đây là IoU từng ảnh, không phải mIoU toàn bộ dataset. Cột IoU cũng được xuất trong `class_pixels.csv`.

## File tải xuống

ZIP có `run.json` (encoder, SHA-256 checkpoint, epoch nếu có), `classes.json` và thư mục riêng cho mỗi ảnh:

- `input.png`: ảnh đầu vào đã resize.
- `prediction_ids.png`: PNG một kênh, giá trị pixel 0–19 (trông gần đen khi mở).
- `prediction_color.png`, `overlay.png`: mask màu và ảnh phủ màu 45%.
- `confidence.npy`: điểm softmax của lớp dự đoán, đọc bằng `numpy.load(..., allow_pickle=False)`.
- `class_pixels.csv`, `image.json`: thống kê lớp và kích thước ảnh nguồn.
- Với ảnh ghép: thêm `ground_truth_ids.png`, `ground_truth_color.png`, `disagreement.png`.

## Thiết bị và lỗi thường gặp

- Sau khi cập nhật mã, nếu giao diện có chú thích mới nhưng bảng vẫn thiếu cột IoU: dừng Streamlit bằng `Ctrl+C` ở terminal chạy app, chạy lại `run_app.ps1`, rồi tải lại trang và chạy segmentation. Chỉ tải lại trình duyệt có thể chưa nạp lại mô-đun Python đang giữ trong bộ nhớ.
- CPU hoạt động với bản cài mặc định. GPU chỉ hiện khi PyTorch nhận CUDA. Muốn dùng GPU NVIDIA, cài bản PyTorch/Torchvision CUDA phù hợp trong cùng `.venv` theo hướng dẫn chính thức của PyTorch.
- Chọn sai encoder hoặc checkpoint có số lớp khác 20: app báo lỗi và không chạy bằng trọng số ngẫu nhiên.
- Thiếu bộ nhớ: chọn CPU hoặc encoder nhỏ hơn đúng với checkpoint, giảm số ảnh trong một lượt; tải kết quả rồi khởi động lại app nếu cần.
- Không thấy ảnh trong folder: đường dẫn phải nằm trên máy chạy app; app đọc JPG/JPEG/PNG ngay trong folder được chọn, không duyệt thư mục con.
- Kết quả chỉ đúng ngữ nghĩa với mô hình đã train cùng thứ tự lớp/palette của notebook.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools\verify_checkpoints.py
```

Lệnh thứ hai nạp và forward checkpoint thật trong 8 folder, dùng ảnh tổng hợp để kiểm tra khả năng chạy; đây không phải phép đo chất lượng segmentation trên dataset.
