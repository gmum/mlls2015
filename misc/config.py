# -*- coding: utf-8 -*-
"""
 Configuration constants pulled from environment variables
"""

import os
from os import path

BASE_DIR = os.path.join(path.dirname(path.abspath(__file__)), "..")
DATA_DIR = os.environ.get("MLLS_DATA_DIR", path.join(BASE_DIR, "data"))
CACHE_DIR = os.environ.get("MLLS_CACHE_DIR", path.join(DATA_DIR, "cache"))
RESULTS_DIR = os.environ.get("MLLS_RESULTS_DIR", path.join(BASE_DIR, "results"))
LOG_DIR = os.environ.get("MLLS_LOG_DIR", path.join(BASE_DIR, "log"))
