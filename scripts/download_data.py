#!/usr/bin/env python
"""Download the HAM10000 dataset from Harvard Dataverse."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.data.download import main as download_main


if __name__ == "__main__":
    sys.exit(download_main())
