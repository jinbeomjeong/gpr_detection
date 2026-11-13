"""Run m31_qa8_outward_check.py unchanged except its output path (QA-8 re-check at P* 6.0).

The checked module uses ``make_detection_image(radargram)`` with the code defaults, which are
P* 6.0 / overlap 192 / compress_max 1.35 since spec v0.4. This wrapper asserts those defaults,
redirects ``OUT`` to ``m31_qa8_outward_check_p6.json`` and calls ``main()``; the committed
P* 4.06 result ``m31_qa8_outward_check.json`` is not touched.
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpr_detection.preprocess.to_image import DEFAULT_DETECTION_IMAGE_CONFIG as CFG  # noqa: E402

assert CFG.p_target_px == 6.0 and CFG.overlap_px == 192 and CFG.compress_max == 1.35, CFG

spec = importlib.util.spec_from_file_location("m31_qa8_outward_check", HERE / "m31_qa8_outward_check.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUT = HERE / "m31_qa8_outward_check_p6.json"
module.main()
print("written", module.OUT)
