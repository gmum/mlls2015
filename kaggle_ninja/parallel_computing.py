#TODO: empty result
#TODO: uuid!

import multiprocessing
from multiprocessing import Pool
from kaggle_ninja import ninja_globals, register, find_obj
from multiprocessing import TimeoutError
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
from utils import find_obj
import uuid
from multiprocessing import Lock
from functools import partial

parallel_computing_lock = Lock()



def ready_jobs():
    global ninja_globals, parallel_computing_lock
    parallel_computing_lock.acquire()
    try:
        val = all(t.ready() if hasattr(t, "ready") else True for t in ninja_globals["current_tasks"])
    except Exception, e:
        val = str(e)
    finally:
        parallel_computing_lock.release()
        return val

def clear_all():
    global ninja_globals, parallel_computing_lock
    try:
        if ninja_globals["slave_pool"]:
            ninja_globals["slave_pool"].terminate()
            ninja_globals["slave_pool"].join()
            ninja_globals["slave_pool"] = ThreadPool(1)
        ninja_globals["current_tasks"] = []
    except Exception, e:
        pass
    return None

def get_results(timeout=0):
    global ninja_globals
    results = []
    for t in ninja_globals["current_tasks"]:
        try:
            results.append(t.get(timeout) if hasattr(t, "get") else t)
        except TimeoutError:
            return "TimeoutError"
    ninja_globals["current_tasks"]= []
    return results

def get_engines_memory(client):
    """Gather the memory allocated by each engine in MB"""
    def memory_mb():
        import os
        import psutil
        return psutil.Process(os.getpid()).get_memory_info().rss / 1e6

    return client[:].apply(memory_mb).get_dict()

import os
def restart(n=2):
    os.system("shell_scripts/restart_ipengines.sh "+str(n)+" &")

def tester(sleep=1):
    import time
    time.sleep(sleep)
    return "Test successful"

register("tester", tester)

def run_job(fnc, *args):
    global ninja_globals
    if not ninja_globals["slave_pool"]:
        ninja_globals["slave_pool"] = ThreadPool(1)

    if isinstance(fnc, str):
        # try:
        fnc = find_obj(fnc)
    else:
        ninja_globals["current_tasks"].append("Not defined function, remember to define function in caller not callee :"+fnc+"|")

    ninja_globals["current_tasks"].append(ninja_globals["slave_pool"].apply_async(fnc, args=args))


def abortable_worker(func, func_kwargs={}, **kwargs):
    timeout = kwargs.get('timeout', 0)
    id = kwargs.get('id', -1)


    if isinstance(func, str):
        # Remember to register it!
        func = find_obj(func)

    if timeout > 0:
        p = ThreadPool(1)
        res = p.apply_async(partial(func, **func_kwargs))
        try:
            out = res.get(timeout)  # Wait timeout seconds for func to complete.
            return out
        except multiprocessing.TimeoutError:
            print("Aborting due to timeout "+str(id))
            return "Timedout for id="+str(id)
    else:
        print func_kwargs
        return func(**func_kwargs)

def get_hostnames(client):
    def hostname():
        import socket
        return socket.gethostname()

    return client[:].apply(hostname).get_dict()

def representative_of_hosts(c):
    hosts = get_hostnames(c)
    hosts_rev = dict({v: k for k,v in hosts.iteritems()})
    return hosts_rev

def get_host_free_memory(client):
    """Free memory on each host of the cluster in MB."""
    all_engines = client[:]
    def hostname():
        import socket
        return socket.gethostname()

    hostnames = all_engines.apply(hostname).get_dict()
    one_engine_per_host = dict((hostname, engine_id)
                               for engine_id, hostname
                               in hostnames.items())

    def host_free_memory():
        import psutil
        return psutil.virtual_memory().free / 1e6


    one_engine_per_host_ids = list(one_engine_per_host.values())
    host_mem = client[one_engine_per_host_ids].apply(
        host_free_memory).get_dict()

    return dict((hostnames[eid], m) for eid, m in host_mem.items())


