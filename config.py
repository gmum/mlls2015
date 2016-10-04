# -*- coding: utf-8 -*-

"""
Settings for this project
"""

import logging, os
from dotenv import load_dotenv, find_dotenv
logging.getLogger().setLevel(level=logging.INFO)

REPOSITORY_DIR = os.path.join(os.path.dirname(__file__), ".")

load_dotenv(find_dotenv())

CONFIG = {
    "data_dir": os.environ.get("DATA_DIR"),
    "results_dir": os.environ.get("RESULTS_DIR"),
    "log_dir": os.environ.get("LOG_DIR")
}

PROTEINS = ["5-HT1A", "5-HT2A", "5-HT2C", "5-HT6", "5-HT7", "D2"]
FINGERPRINTS = ["MACCS", "EState", "KR"]

logging.info("Configured env variables: {}".format(str(CONFIG)))
