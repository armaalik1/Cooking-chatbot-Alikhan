
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure() -> str:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    return ROOT
