"""Local qualitative segmentation viewer: streamlit run app.py."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from threading import Lock
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from segmentation_core import (
    CLASSES, ENCODERS, NAMES, PALETTE, class_statistics, export_zip,
    load_model, overlay, png_bytes, predict,
)


ROOT = Path(__file__).resolve().parent
PUBLIC_MODE = os.environ.get("SEGMENTATION_PUBLIC_MODE") == "1"
MAX_IMAGES = 10 if PUBLIC_MODE else 100
st.set_page_config(page_title="Tiểu luận Deep Learning · Nhóm 9", page_icon="🏙️", layout="wide")
torch.set_num_threads(min(4, torch.get_num_threads()))
st.markdown("""
<style>
.block-container {padding-top: 4rem; max-width: 1500px;}
[data-testid="stMetric"] {background: #eaf0f6; border-radius: 12px; padding: 12px 16px;}
[data-testid="stSidebar"] {border-right: 1px solid #dce5ee;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def public_model():
    path = ROOT / "results_w_EfficientNetB3" / "best_model.pth"
    model, metadata = load_model(path, encoder="efficientnet-b3", device="cpu")
    metadata.update(checkpoint=path.name, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return model, metadata, Lock()


def checkpoint_request():
    """Read current explicit checkpoint choice only when an action is submitted."""
    if PUBLIC_MODE:
        model, metadata, _ = public_model()
        return model, dict(metadata)
    if checkpoint_mode == "Tải file .pth":
        if checkpoint_upload is None:
            raise ValueError("Hãy chọn file checkpoint .pth.")
        data = checkpoint_upload.getvalue()
        name = checkpoint_upload.name
    else:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise ValueError("Không tìm thấy checkpoint. Kiểm tra đường dẫn file .pth.")
        data = path.read_bytes()
        name = path.name
    digest = hashlib.sha256(data).hexdigest()
    key = (digest, encoder, device)
    if st.session_state.get("model_key") != key:
        st.session_state.pop("model", None)
        st.session_state.pop("model_key", None)
        st.session_state.pop("model_metadata", None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, metadata = load_model(data, encoder=encoder, device=device)
        metadata.update(checkpoint=name, checkpoint_sha256=digest)
        st.session_state.model = model
        st.session_state.model_metadata = metadata
        st.session_state.model_key = key
    return st.session_state.model, dict(st.session_state.model_metadata)


def show_error(exc):
    st.error(str(exc))
    with st.expander("Chi tiết lỗi"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.caption("Checkpoint phải khớp UNet 20 lớp của notebook. App dùng weights_only=True khi đọc file.")


with st.sidebar:
    st.markdown("### 01 · Model")
    checkpoint_mode = "Model demo" if PUBLIC_MODE else st.radio(
        "Nguồn checkpoint", ["Có sẵn trong dự án", "Đường dẫn .pth", "Tải file .pth"])
    checkpoint_upload = None
    checkpoint_path = ""
    if PUBLIC_MODE:
        st.info("UNet + EfficientNet-B3 · best model đã huấn luyện")
    elif checkpoint_mode == "Có sẵn trong dự án":
        paths = sorted(ROOT.glob("results_w_EfficientNetB*/best_model.pth"))
        if paths:
            labels = [str(path.relative_to(ROOT)) for path in paths]
            default = next((i for i, path in enumerate(paths) if "EfficientNetB3" in str(path)), 0)
            selected = st.selectbox("Checkpoint", labels, index=default,
                                    format_func=lambda x: Path(x).parent.name.replace("results_w_", ""))
            checkpoint_path = str(ROOT / selected)
        else:
            st.info("Chọn tải file hoặc nhập đường dẫn checkpoint.")
    elif checkpoint_mode == "Đường dẫn .pth":
        checkpoint_path = st.text_input("Đường dẫn checkpoint", placeholder="C:/models/best_model.pth")
    else:
        checkpoint_upload = st.file_uploader("Checkpoint .pth / .pt", type=["pth", "pt"], key="checkpoint_file")
    encoder = "efficientnet-b3" if PUBLIC_MODE else st.selectbox("Encoder", ["auto", *ENCODERS],
                           format_func=lambda x: "Tự nhận diện B0–B7" if x == "auto" else x)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    device = "cpu" if PUBLIC_MODE else st.selectbox("Thiết bị", devices, index=len(devices) - 1,
                          format_func=lambda x: "CPU" if x == "cpu" else "GPU · CUDA")
    if st.button("Nạp model", width="stretch"):
        try:
            with st.spinner("Đang nạp checkpoint…"):
                checkpoint_request()
            st.success("Đã nạp model.")
        except Exception as exc:
            show_error(exc)
    if "model_metadata" in st.session_state:
        loaded = st.session_state.model_metadata
        st.caption(f"Model đã nạp: {loaded['encoder']} · {loaded['device'].upper()}")
        if "epoch" in loaded:
            st.caption(f"Checkpoint epoch {loaded['epoch']}")
    st.divider()
    st.markdown("### Quy ước hiển thị")
    st.caption("Input và mask kết quả: 256×256. Class ID: 0–19, cùng bảng màu với notebook.")
    st.caption("Ảnh ghép: trái là ảnh RGB, phải là nhãn màu. Ground truth được giải mã bằng màu gần nhất như lúc train.")
    st.caption("Ảnh được gửi lên máy chủ để xử lý. Chọn tối đa 10 ảnh mỗi lượt." if PUBLIC_MODE
               else "Ảnh, checkpoint và kết quả được xử lý trên máy chạy app.")


st.title("Tiểu luận cuối kỳ môn học Deep Learning")
st.subheader("Ứng dụng mạng EfficientNet vào bài toán phân đoạn ảnh giao thông sử dụng kiến trúc Encoder-Decoder")
st.markdown("**Nhóm 9 - Trần Thanh Hùng - Nguyễn Xuân Lâm**")
st.write("Model: UNet với EfficientNet encoder")
st.write("Dataset: Cityscapes Semantic Segmentation")

st.markdown("### 02 · Chọn ảnh")
source_mode = "Tải một hoặc nhiều ảnh" if PUBLIC_MODE else st.radio(
    "Nguồn ảnh", ["Tải một hoặc nhiều ảnh", "Thư mục trên máy"], horizontal=True)
input_format = st.radio("Định dạng", ["Cityscapes ghép ảnh + mask", "Ảnh RGB riêng"], horizontal=True)
paired = input_format == "Cityscapes ghép ảnh + mask"
uploads = []
selected_paths = []
if source_mode == "Tải một hoặc nhiều ảnh":
    uploads = st.file_uploader("Chọn ảnh train / validation", type=["png", "jpg", "jpeg"],
                               accept_multiple_files=True, key="input_images")
else:
    folder = st.text_input("Đường dẫn thư mục train hoặc val", placeholder="D:/cityscapes_data/val")
    if st.button("Đọc danh sách ảnh"):
        try:
            directory = Path(folder).expanduser()
            if not directory.is_dir():
                raise ValueError("Thư mục không tồn tại trên máy chạy app.")
            files = sorted(path for path in directory.iterdir()
                           if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
            st.session_state.folder_scan = (folder, [str(path.resolve()) for path in files])
            st.session_state.pop("chosen_paths", None)
        except Exception as exc:
            st.error(str(exc))
    scan_folder, files = st.session_state.get("folder_scan", (None, []))
    if scan_folder == folder:
        st.caption(f"Tìm thấy {len(files)} ảnh. Chọn tối đa 100 ảnh mỗi lần chạy.")
        selected_paths = st.multiselect("Ảnh cần xem", files, default=files[:min(6, len(files))],
                                       format_func=lambda x: Path(x).name, key="chosen_paths",
                                       max_selections=100)
    else:
        st.caption("Nhấn Đọc danh sách ảnh sau khi nhập hoặc thay đổi thư mục.")

count = len(uploads) if source_mode == "Tải một hoặc nhiều ảnh" else len(selected_paths)
if count > MAX_IMAGES:
    st.warning(f"Hãy chọn tối đa {MAX_IMAGES} ảnh mỗi lần để giới hạn bộ nhớ.")
run = st.button(f"Chạy segmentation · {count} ảnh", type="primary", disabled=count == 0 or count > MAX_IMAGES)
if run:
    # Old results must not be mistaken for the requested new run if loading fails.
    for key in ("results", "run_metadata", "export_archive", "result_index", "run_errors"):
        st.session_state.pop(key, None)
    try:
        with st.spinner("Đang chuẩn bị model…"):
            model, metadata = checkpoint_request()
        results, errors = [], []
        started = time.perf_counter()
        progress = st.progress(0.0, text="Bắt đầu dự đoán…")
        items = uploads if source_mode == "Tải một hoặc nhiều ảnh" else selected_paths
        for index, item in enumerate(items):
            name = item.name if source_mode == "Tải một hoặc nhiều ảnh" else Path(item).name
            try:
                data = item.getvalue() if source_mode == "Tải một hoặc nhiều ảnh" else Path(item).read_bytes()
                if PUBLIC_MODE:
                    with public_model()[2]:
                        results.append(predict(model, data, name, paired, device))
                else:
                    results.append(predict(model, data, name, paired, device))
            except Exception as exc:
                errors.append(f"{name}: {exc}")
            progress.progress((index + 1) / count, text=f"Đã xử lý {index + 1}/{count}: {name}")
        metadata.update(input_format=input_format, image_count=len(results),
                        elapsed_seconds=round(time.perf_counter() - started, 3))
        st.session_state.results = results
        st.session_state.run_metadata = metadata
        st.session_state.run_errors = errors
    except Exception as exc:
        show_error(exc)

for error in st.session_state.get("run_errors", []):
    st.warning(error)

results = st.session_state.get("results", [])
if not results:
    st.info("Chọn checkpoint và ảnh, sau đó nhấn Chạy segmentation. Model sẽ được nạp tự động nếu cần.")
    st.stop()

metadata = st.session_state.run_metadata
st.divider()
st.markdown("### 03 · Kết quả định tính")
st.caption(f"Kết quả lần chạy: {metadata['checkpoint']} · {metadata['encoder']} · "
           f"{metadata['device'].upper()} · {metadata['input_format']}. "
           "Đổi lựa chọn phía trên rồi nhấn Chạy segmentation để tạo kết quả mới.")
summary = st.columns(3)
summary[0].metric("Ảnh đã xử lý", len(results))
summary[1].metric("Encoder", metadata["encoder"].replace("efficientnet-", "").upper())
summary[2].metric("Thời gian dự đoán", f"{metadata['elapsed_seconds']:.2f} s")

index = st.selectbox("Ảnh đang xem", range(len(results)),
                     format_func=lambda i: f"{i + 1:02d} · {results[i].name}", key="result_index")
result = results[index]
st.caption(f"Kích thước ảnh RGB nguồn: {result.source_size[0]}×{result.source_size[1]} · "
           "Tọa độ kiểm tra và các mask tải xuống dùng lưới 256×256.")
columns = st.columns(3 if result.ground_truth is not None else 2)
columns[0].image(result.image, caption="Ảnh đầu vào · 256×256", width="stretch")
if result.ground_truth is not None:
    columns[1].image(PALETTE[result.ground_truth], caption="Ground truth", width="stretch")
columns[-1].image(PALETTE[result.labels], caption="Dự đoán · class color", width="stretch")

tab_inspect, tab_stats, tab_export = st.tabs(["Khám phá pixel", "Phân bố lớp", "Tải kết quả"])
with tab_inspect:
    controls, canvas = st.columns([1, 3])
    with controls:
        alpha = st.slider("Độ phủ mask", 0.0, 1.0, 0.45, step=0.05)
        highlighted = st.selectbox("Lớp phủ lên ảnh", [-1, *range(len(CLASSES))],
                                  format_func=lambda i: "Tất cả lớp" if i == -1 else f"{i} · {CLASSES[i][0]}")
        x = int(st.number_input("Pixel X", min_value=0, max_value=255, value=128, step=1))
        y = int(st.number_input("Pixel Y", min_value=0, max_value=255, value=128, step=1))
        label = int(result.labels[y, x])
        st.write(f"**Dự đoán:** {label} · {CLASSES[label][0]}")
        if result.ground_truth is not None:
            truth = int(result.ground_truth[y, x])
            st.write(f"**Ground truth:** {truth} · {CLASSES[truth][0]}")
            st.write("✓ Khớp nhãn" if label == truth else "✕ Khác nhãn")
        st.caption("Rê chuột hoặc phóng to ảnh bên cạnh để kiểm tra thêm.")
    with canvas:
        display = overlay(result, alpha, None if highlighted < 0 else highlighted)
        custom = np.empty((*result.labels.shape, 3), dtype=object)
        custom[..., 0] = result.labels
        custom[..., 1] = NAMES[result.labels]
        custom[..., 2] = NAMES[result.ground_truth] if result.ground_truth is not None else "Không có"
        fig = go.Figure(go.Image(z=display, customdata=custom, hovertemplate=
            "x=%{x}, y=%{y}<br>Dự đoán: %{customdata[0]} · %{customdata[1]}"
            "<br>GT: %{customdata[2]}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", marker=dict(
            symbol="cross", size=13, color="white", line=dict(color="black", width=1)),
            hoverinfo="skip", showlegend=False))
        fig.update_layout(height=570, margin=dict(l=5, r=5, t=5, b=5),
                          dragmode="pan", uirevision=f"{metadata['checkpoint_sha256']}-{index}")
        fig.update_xaxes(range=[-0.5, 255.5], title="X", constrain="domain")
        fig.update_yaxes(range=[255.5, -0.5], title="Y", scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
    if result.ground_truth is not None:
        with st.expander("Vùng dự đoán khác ground truth"):
            error_image = (result.image.astype(np.float32) * 0.35).astype(np.uint8)
            error_image[result.labels != result.ground_truth] = (239, 68, 68)
            st.image(error_image, caption="Đỏ: khác nhãn ground truth đã giải mã; gồm cả lớp unlabeled.", width=512)

with tab_stats:
    legend = " ".join(
        f'<span style="display:inline-block;margin:4px 12px 4px 0">'
        f'<span style="display:inline-block;width:12px;height:12px;background:rgb{color};'
        f'border:1px solid #aaa;margin-right:5px"></span>{i} · {name}</span>'
        for i, (name, color) in enumerate(CLASSES)
    )
    st.markdown(legend, unsafe_allow_html=True)
    st.dataframe(class_statistics(result), hide_index=True, width="stretch")
    st.caption("IoU (%) = 100 × TP / (TP + FP + FN), tính riêng từng lớp trên ảnh đang xem ở lưới 256×256. "
               "Để trống khi không có ground truth hoặc lớp không xuất hiện trong cả dự đoán và ground truth.")

with tab_export:
    st.download_button("Tải mask class ID của ảnh này", data=png_bytes(result.labels),
                       file_name=f"{Path(result.name).stem}_ids.png", mime="image/png")
    st.caption("PNG một kênh chứa giá trị 0–19. Ảnh này trông gần đen khi mở bằng trình xem ảnh thông thường.")
    if st.button("Tạo ZIP cho tất cả ảnh"):
        with st.spinner("Đang đóng gói kết quả…"):
            st.session_state.export_archive = export_zip(results, metadata)
    if "export_archive" in st.session_state:
        st.download_button("Tải toàn bộ kết quả (.zip)", data=st.session_state.export_archive,
                           file_name="segmentation_results.zip", mime="application/zip")
    st.caption("ZIP gồm ảnh đầu vào, mask ID, mask màu, overlay, confidence, phân bố lớp và thông tin checkpoint. "
               "Ảnh ghép có thêm ground truth và mask khác nhãn. Overlay trong ZIP dùng độ phủ 45% cho tất cả lớp.")
