import os
import re

from misc.config import c
data_dir = c["DATA_DIR"]


def list_all_data():
    """
    Returns list of pairs with compound and fingerprint for all data in DATA_DIR from config
    :return: tuple, (compounds, fingerprints)
    """
    data = []
    for f in os.listdir(data_dir):
        split = re.split('\.|_', f)
        if split[-1] == 'libsvm':
            data.append(split[:-1])

    return data