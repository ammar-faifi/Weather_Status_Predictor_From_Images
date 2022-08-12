"""
    To remove files starting with `.` or `._`
"""

import os
from pathlib import Path

from utils import CLASSES

DATA_DIR = Path("data")

if __name__ == "__main__":
    for dir_ in CLASSES:
        for file_name in os.listdir(DATA_DIR / dir_):
            if file_name.startswith("._"):
                print(f"removing {file_name}")
                os.remove(DATA_DIR / dir_ / file_name)
