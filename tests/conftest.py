import os
import sys

# Must be set before any nanotron imports, as config.py reads this at module level
os.environ.setdefault("OUT_DIR", "/tmp/nanotron_test_output")
# train.py prints this env var at import time; default to empty string (CPU-only tests)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Force JAX to use CPU in the test suite to avoid conflicts on shared GPU machines.
# JAX_PLATFORMS (the current name) supersedes the older JAX_PLATFORM_NAME variable.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Add project root to sys.path so tests can import package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
