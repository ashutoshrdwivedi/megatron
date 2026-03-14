import os
import sys

# Must be set before any nanotron imports, as config.py reads this at module level
os.environ.setdefault("OUT_DIR", "/tmp/nanotron_test_output")

# Add project root to sys.path so tests can import package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
