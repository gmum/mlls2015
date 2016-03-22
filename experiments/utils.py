# -*- coding: utf-8 -*-
"""
 Various utils functions for running experiments
"""
from os import path, system
import os
from six import iteritems
import logging
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
import time
from multiprocessing import Pool
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import json
from misc.config import RESULTS_DIR
import base64
import numpy as np

logger = logging.getLogger(__name__)

def upload_df_to_drive(df, name="test.csv"):
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



def _check_duplicates(tasks):
    """ Checks that name is unique """
    tasks_dict = {}
    for t in tasks:
        if t[1]['name'] in tasks:
            logger.error(tasks_dict[1]['name'])
            logger.error(t[1])
            raise RuntimeError("Duplicated name in tasks")
        tasks_dict[t[1]['name']] = t

    already_calculated = 0
    for name in tasks_dict:
        kwargs = tasks_dict[name][1]
        target = path.join(kwargs['output_dir'], name) + ".json"
        if path.exists(target):
            already_calculated += 1
            done_job = json.load(open(target))
            shared_items = set(kwargs.items()) & set(done_job['opts'].items())
            if not (len(shared_items) == len(kwargs) == len(done_job['opts'])):
                print set(kwargs.items()) - set(done_job['opts'].items())
                raise RuntimeError("Found calculated job with same name but different parameters in %s" % target)

    if already_calculated:
        logger.warning("Skipping calculation of " + str(already_calculated) + " jobs (already calculated)")



def run_async_with_reporting(f, tasks, output_dir, n_jobs):


    # Naming should be unique and dir shouldn't have duplicated jobs already calculated
    _check_duplicates(tasks)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(path.join(output_dir, "failed_jobs.err"), "w") as _:
        pass

    with open(path.join(output_dir, "duplicated_jobs.err"), "w") as _:
        pass

    elapsed = 0
    burn_in_time = 9
    started_with = 0

    rs = Pool(n_jobs).map_async(f, tasks, chunksize=1)

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

    if os.stat(path.join(output_dir, "duplicated_jobs.err")).st_size != 0:
        raise RuntimeError("Some jobs were duplicated")

    if os.stat(path.join(output_dir, "failed_jobs.err")).st_size != 0:
        raise RuntimeError("Some jobs failed")

    return rs.get()

def dict_hash(my_dict):
    return str(abs(hash(frozenset(my_dict.items()))))

def run_job(job):
    script, kwargs = job
    target = path.join(kwargs['output_dir'], kwargs['name'])+ ".json"

    if not path.exists(target):

        # Escape \ with \\ for bash escaping
        for key, value in iteritems(kwargs):
            if isinstance(value, str):
                kwargs[key] = value.replace(r'"', r'\"')

        cmd = "{} {}".format(script, " ".join("--{}={}".format(k, v) for k, v in iteritems(kwargs)))
        logger.info("Running " + cmd)
        res = system(cmd)
        if res != 0:
            logger.error("Failed job {}".format(cmd))
            with open(path.join(kwargs['output_dir'], "failed_jobs.err"), "a") as f:
                f.write("{}\n".format(cmd))
    else:
        done_job = json.load(open(target))
        shared_items = set(kwargs.items()) & set(done_job['opts'].items())
        if not (len(shared_items) == len(kwargs) == len(done_job['opts'])):
            logger.error("Wrote down job has non matching json")
            with open(path.join(kwargs['output_dir'], "duplicated_jobs.err"), "a") as f:
                f.write("{}\n".format(target + ".json"))

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

def wac_scoring(estimator, X, y):
    return wac_score(y, estimator.predict(X))


def get_output_dir(model, compound, fingerprint, strategy, param=None, special=None):

    if strategy in ['PassiveStrategy', 'UncertaintySampling']:
        dir_name = 'unc'
    elif strategy == 'CSJSampling':
        dir_name = 'csj'
    elif strategy == 'QuasiGreedyBatch':
        dir_name = 'qgb'
    elif strategy == 'QueryByBagging':
        dir_name = 'qbb'

    if param is not None:
        assert isinstance(param, int) or isinstance(param, float)
        dir_name += "-" + str(param)

    if special is None:
        return path.join(RESULTS_DIR, model, compound, fingerprint, dir_name)
    else:
        assert isinstance(special, str)
        return path.join(RESULTS_DIR, special, model, compound, fingerprint, dir_name)


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict
        holding dtype, shape and the data, base64 encoded.
        """
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)

def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct