"""Run the same interface and workflow on cloud and localhost."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
