# -*- coding: utf-8 -*-
"""
 Various utils functions for running experiments
"""
from os import path, system
from six import iteritems
import logging
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
import time
from multiprocessing import Pool

logger = logging.getLogger(__name__)

def upload_df_to_drive(df, name="test.csv"):
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")

    drive = GoogleDrive(gauth)
    f = drive.CreateFile({'title':name, 'mimeType':'text/csv', "parents":
        [{"kind": "drive#fileLink", "id": "0B9cVNObhE5w9RzNFOEt1VFFmRUk"}]})
    f.SetContentString(df.to_csv())
    f.Upload(param={'convert': True})


def run_async_with_reporting(f, tasks, n_jobs):
    rs = Pool(n_jobs).map_async(f, tasks, chunksize=1)

    elapsed = 0
    burn_in_time = 9
    started_with = 0

    while True :
        if rs.ready():
            logger.info("Done")
            break
        remaining = rs._number_left
        logger.info(("Waiting for", remaining, "tasks to complete"))

        time.sleep(3)
        elapsed += 3.0
        if elapsed > burn_in_time:
            if started_with == 0:
                started_with = remaining
            completed = started_with - remaining
            if completed > 0:
                logger.info(("Estimated time is: ", (remaining * (elapsed - burn_in_time)) / float(completed)))

    return rs.get()

def dict_hash(my_dict):
    return str(abs(hash(frozenset(my_dict.items()))))

def run_job(job):
    script, kwargs = job
    target = path.join(kwargs['output_dir'], dict_hash(kwargs))
    if not path.exists(target + ".json"):
        kwargs['name'] = target
        cmd = "{} {}".format(script, " ".join("--{}={}".format(k, v) for k, v in iteritems(kwargs)))
        logger.info("Running " + cmd)
        res = system(cmd)
        if res != 0:
            logger.error("Failed job {}".format(cmd))
            with open("failed_jobs.err", "a") as f:
                f.write("{}\n".format(cmd))

def wac_score(Y_true, Y_pred):
    cm = confusion_matrix(Y_true, Y_pred)
    if cm.shape != (2,2):
        return accuracy_score(Y_true, Y_pred)
    tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
    if tp == 0 and fn == 0:
        return 0.5*tn/float(tn+fp)
    elif tn == 0 and fp == 0:
        return 0.5*tp/float(tp+fn)
    return 0.5*tp/float(tp+fn) + 0.5*tn/float(tn+fp)
