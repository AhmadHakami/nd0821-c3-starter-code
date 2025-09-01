import os
import sys

# Ensure the repository root is on sys.path so imports like
# 'from starter.starter.ml.model' resolve correctly during tests.
# tests/ -> starter/ -> repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
