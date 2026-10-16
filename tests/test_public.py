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
            self.assertFalse(app.selectbox)
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


if __name__ == "__main__":
    unittest.main()
