"""Public deployment entry point; app.py retains the full local workflow."""

import os
from pathlib import Path
import runpy

os.environ["SEGMENTATION_PUBLIC_MODE"] = "1"
runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
