import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from segmentation_core import load_model, png_bytes, predict


ROOT = Path(__file__).resolve().parents[1]


class PublicAppTests(unittest.TestCase):
    def test_public_model_and_results(self):
        with patch.dict(os.environ, {"SEGMENTATION_PUBLIC_MODE": "1"}):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=90).run()
            self.assertEqual(len(app.exception), 0)
            self.assertFalse(app.text_input)
            self.assertEqual(next(r for r in app.radio if r.label == "Nguồn checkpoint").options,
                             ["Có sẵn trong dự án", "Đường dẫn .pth", "Tải file .pth"])
            self.assertEqual(len(next(s for s in app.selectbox if s.label == "Encoder").options), 9)
            self.assertIn("Thư mục trên máy", next(r for r in app.radio if r.label == "Nguồn ảnh").options)
            self.assertEqual(len(app.get("file_uploader")), 1)
            next(button for button in app.button if button.label == "Nạp model").click().run()
            self.assertEqual(len(app.error), 0)
            self.assertEqual(len(app.success), 1)
            # Run B3 inference and render its results through the public entry point.
            model, metadata = load_model(ROOT / "results_w_EfficientNetB3" / "best_model.pth")
            image = np.zeros((256, 512, 3), dtype=np.uint8)
            image[:, 256:] = (128, 64, 128)
            result = predict(model, png_bytes(image), "paired.png", True, "cpu")
            self.assertEqual(metadata["encoder"], "efficientnet-b3")
            app.session_state["results"] = [result]
            app.session_state["run_metadata"] = dict(metadata, checkpoint="best_model.pth", checkpoint_sha256="test",
                                                     input_format="Cityscapes ghép ảnh + mask",
                                                     elapsed_seconds=0.0)
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("IoU (%)", app.dataframe[0].value.columns)

    def test_cloud_upload_and_server_path_controls(self):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=90).run()
        next(r for r in app.radio if r.label == "Nguồn checkpoint").set_value("Tải file .pth").run()
        self.assertEqual(len(app.get("file_uploader")), 2)
        next(b for b in app.button if b.label == "Nạp model").click().run()
        self.assertIn("Hãy chọn file checkpoint", app.error[0].value)
        next(r for r in app.radio if r.label == "Nguồn checkpoint").set_value("Đường dẫn .pth").run()
        next(t for t in app.text_input if t.label == "Đường dẫn checkpoint").input(
            str(ROOT / "results_w_EfficientNetB3" / "best_model.pth")).run()
        next(b for b in app.button if b.label == "Nạp model").click().run()
        self.assertEqual(len(app.error), 0)
        self.assertEqual(app.session_state["model_metadata"]["encoder"], "efficientnet-b3")
        next(r for r in app.radio if r.label == "Nguồn ảnh").set_value("Thư mục trên máy").run()
        self.assertIn("Đường dẫn thư mục train hoặc val", [t.label for t in app.text_input])
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
