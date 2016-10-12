
import numpy as np
import logging
from sklearn.metrics import auc

logger = logging.getLogger(__name__)

def generate_time_report(monitor_outputs):
    # Returns dict with percentage/amount of time spent in each section (all keys with "_time" suffix)
    report = {}
    total_time = float(sum(monitor_outputs['iter_time']))
    for k in monitor_outputs:
        if k.endswith("_time"):
            # 1st: time in seconds, 2nd: percent of all time
            report[k] = [sum(monitor_outputs[k]), sum(monitor_outputs[k]) / total_time]
    return report


def calculate_scores(monitor_outputs):
    scores = {}
    for k in monitor_outputs:
        if "score" in k and isinstance(monitor_outputs[k], list) or isinstance(monitor_outputs[k], np.ndarray):
            if len(monitor_outputs[k]) == 0 or isinstance(monitor_outputs[k][0], list) or \
                    isinstance(monitor_outputs[k][0], np.ndarray):
                logger.info("Skipping calculation of scores for " + k + " because is a list of lists or is empty.")
                continue

            try:
                scores[k + "_mean"] = np.mean(monitor_outputs[k])
                scores[k + "_auc"] = auc(np.arange(len(monitor_outputs[k])), monitor_outputs[k])
            except Exception :
                logger.warning("Failed calculating score for " + k + ", might have been expected.")
    return scores