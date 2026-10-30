#!/usr/bin/env python3
"""Root entry point for training Plant Disease Classification models.
Delegates to src.train.main().
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import main

if __name__ == "__main__":
    main()
