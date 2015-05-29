#TODO: empty result

from multiprocessing import Pool
from kaggle_ninja import ninja_globals
from multiprocessing import TimeoutError
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing

def representative_of_hosts(c):
    hosts = get_hostnames(c)
    hosts_rev = dict({v: k for k,v in hosts.iteritems()})
    return hosts_rev

def ready_jobs():
    global ninja_globals
    return all(t.ready() if hasattr(t, "ready") else True for t in ninja_globals["current_tasks"])


def cluster_ready_jobs(view):
    return view.apply(ready_jobs).get()

from functools import partial

def cluster_run_jobs(view, fnc,  *args, **kwargs):
    """
    :param view: Might be both balanced and direct
    :param fnc: Note: must be defined on caller NOT caee
    :param timeout: after timeout returns just note that timeouted
    :return:
    """
    view.apply(run_job, partial(abortable_worker, timeout=kwargs.get("timeout", 0), id=kwargs.get("id", -1)),
                       fnc, *args)

def cluster_get_results(view):
    return view.apply(get_results)



def clear_all():
    global ninja_globals
    ninja_globals["slave_pool"].terminate()
    ninja_globals["slave_pool"].join()
    ninja_globals["slave_pool"] = Pool(1)
    ninja_globals["current_tasks"] = []
    return None

def get_results():
    global ninja_globals
    results = [(t.get(0) if hasattr(t, "get") else t) for t in ninja_globals["current_tasks"]]
    ninja_globals["current_tasks"] = []
    return results

def get_engines_memory(client):
    """Gather the memory allocated by each engine in MB"""
    def memory_mb():
        import os
        import psutil
        return psutil.Process(os.getpid()).get_memory_info().rss / 1e6

    return client[:].apply(memory_mb).get_dict()

import os
def restart():
    os.system("shell_scripts/restart_ipengines.sh &")

def tester(sleep=1):
    import time
    time.sleep(sleep)
    return "Test successful"

def run_job(fnc, *args):
    global ninja_globals
    # ninja_globals["current_tasks"].append(ninja_globals["slave_pool"].apply_async(abortable_worker, fnc, \
    # timeout=timeout, *args))
    if isinstance(fnc, str):
        if not (fnc in globals()) and not (fnc in locals()):
            ninja_globals["current_tasks"].append("Not defined function, remember to define function in caller not callee :"+fnc+"|")
            return
        fnc = globals()[fnc] if fnc in globals() else locals()[fnc]

    ninja_globals["current_tasks"].append(ninja_globals["slave_pool"].apply_async(fnc, args=args))


def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', 0)
    id = kwargs.get('id', -1)

    if isinstance(func, str):
        if not (func in globals()) and not (func in locals()):
            ninja_globals["current_tasks"].append("Not defined function, remember to define function in caller not callee :"+fnc+"|")
            return
        func = globals()[func] if func in globals() else locals()[func]


    if timeout > 0:
        p = ThreadPool(1)
        res = p.apply_async(func, args=args)
        try:
            out = res.get(timeout)  # Wait timeout seconds for func to complete.
            return out
        except multiprocessing.TimeoutError:
            print("Aborting due to timeout "+str(id))
            return "Timedout for id="+str(id)
    else:
        return func(*args)

def get_hostnames(client):
    def hostname():
        import socket
        return socket.gethostname()

    return client[:].apply(hostname).get_dict()



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


