#TODO: empty result
#TODO: uuid!

import multiprocessing
from multiprocessing import Pool
from kaggle_ninja import ninja_globals, register, find_obj
from multiprocessing import TimeoutError
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
from multiprocessing import TimeoutError
import threading
from utils import find_obj
import uuid
import time
from multiprocessing import Lock
from functools import partial

parallel_computing_lock = Lock()

class ParallelComputingError(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super(ParallelComputingError, self).__init__(message)

from multiprocessing import Queue, Pool, current_process
import sys

from StringIO import StringIO

def initializer():
    global ninja_globals
    ninja_globals["slave_pool_out"] = StringIO() # Delete all content
    sys.stderr = sys.stdout = ninja_globals["slave_pool_out"]


class IPClusterTask(object):
    def __init__(self, job_name, job_args=[]):
        """
        @param direct_view Where is run
        """
        self.direct_view = None
        self.job_name = job_name
        self.job_args = job_args
        self.finished = False
        self.results = None
        self.uuid = None

    def execute(self, direct_view):
        self.direct_view = direct_view
        self.uuid = self.direct_view.apply(run_job, self.job_name, *self.job_args).get()


    def ready(self):
        if not self.direct_view or not self.uuid:
            return False

        if self.finished:
            return self.finished
        else:
            self.finished = self.direct_view.apply(ready_job, self.uuid).get()
            return self.finished

    def get(self, timeout=0):
        if not self.direct_view or not self.uuid:
            raise TimeoutError()

        if self.results:
            return self.results

        self.results = self.direct_view.apply(get_result, self.uuid, timeout).get()
        self.finished = True
        return self.results

class IPClusterPool(object):
    def __init__(self, workers):
        self.queue_lock = Lock()
        self.queue = []
        self.workers = {w: None for w in workers}
        self.terminated = False
        self.closed = False
        threading.Thread(target=self._manager).start()


    def terminate(self):
        self.terminated = True

    def join(self):
        while self.terminated != True:
            pass

    def close(self):
        self.closed = True

    def apply_async(self, job, *job_args):
        assert isinstance(job, str), "Job has to be string"
        try:
            self.queue_lock.acquire()
            self.queue.append(IPClusterTask(job, job_args))
            return self.queue[-1]
        except Exception, e:
            raise e
        finally:
            self.queue_lock.release()

    def _manager(self):
        while not self.terminated:

            if self.closed:
                if len(self.queue) == 0:
                    break

            try:
                self.queue_lock.acquire()

                # Finished tasks?
                for w in self.workers:
                    if self.workers[w]:
                        try:
                            self.workers[w].get(0)
                            self.workers[w] = None
                        except TimeoutError:
                            pass


                for w in self.workers:
                    if self.workers[w] is None and len(self.queue):
                        task = self.queue.pop()
                        self.workers[w] = task
                        self.workers[w].execute(w)

            except Exception, e:
                raise e
            finally:
                self.queue_lock.release()

            time.sleep(1)

        self.terminated = True
        self.closed = True

def list_jobs():
    global ninja_globals, parallel_computing_lock
    parallel_computing_lock.acquire()
    try:
        val = []
        for t in ninja_globals["current_tasks"]:
            try:
                _ = ninja_globals["current_tasks"][t].get(0)
                val.append((t, True))
            except:
                val.append((t, False))
    except Exception, e:
        val = e
    finally:
        parallel_computing_lock.release()
        return val

def run_job(fnc, *args):
    global ninja_globals, parallel_computing_lock
    parallel_computing_lock.acquire()
    try:
        if isinstance(fnc, str):
            uid = uuid.uuid1()

            if not ninja_globals["slave_pool"]:
                ninja_globals["slave_pool"] = ThreadPool(1, initializer)

            try:
                fnc = find_obj(fnc)
            except:
                raise ParallelComputingError("Not defined function, remember to define function in caller not callee")

            ninja_globals["current_tasks"][uid] = ninja_globals["slave_pool"].apply_async(fnc, args=args)
            val = uid
        else:
            raise ParallelComputingError("Fnc has to be registered function in kaggle_ninja and passed as str")
    except Exception, e:
        val = e
    finally:
        parallel_computing_lock.release()
        return val


def ready_job(uid):
    global ninja_globals, parallel_computing_lock
    parallel_computing_lock.acquire()
    try:
        val = ninja_globals["current_tasks"][uid].ready()
    except Exception, e:
        val = e
    finally:
        parallel_computing_lock.release()
        return val

def wazzup(n_lines):
    global ninja_globals, parallel_computing_lock
    try:
        parallel_computing_lock.acquire()
        val = ninja_globals["slave_pool_out"].buflist[-n_lines:]
    except Exception, e:
        val = e
    finally:
        parallel_computing_lock.release()
        return val

def get_result(uid, timeout=0):
    global ninja_globals, parallel_computing_lock
    try:
        parallel_computing_lock.acquire()
        val = ninja_globals["current_tasks"][uid].get(timeout)
        del ninja_globals["current_tasks"][uid]
    except Exception, e:
        val = e
    finally:
        parallel_computing_lock.release()
        return val

def clear_all():
    global ninja_globals, parallel_computing_lock
    try:
        parallel_computing_lock.acquire()
        if ninja_globals["slave_pool"]:
            ninja_globals["slave_pool"].terminate()
            ninja_globals["slave_pool"].join()
            ninja_globals["slave_pool"] = ThreadPool(1, initializer)
        ninja_globals["current_tasks"] = {}
        val = True
    except Exception, e:
        val = ParallelComputingError(str(e))
    finally:
        parallel_computing_lock.release()
        return val
    






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

def wazzup_slaves(direct_view, n_lines=3):
    while True:
        from IPython.core.display import clear_output
        clear_output()

        wazzups = direct_view.apply(wazzup, n_lines).get()

        for hostname, wazzup_val in zip(get_hostnames(direct_view).iteritems(), wazzups):
            print "\x1b[31m"+str(hostname)+"\x1b[0m"
            print "".join(wazzup_val)
            print ""
            sys.stdout.flush()

        time.sleep(1)

def get_hostnames(direct_view):
    def hostname():
        import socket
        return socket.gethostname()

    return direct_view.apply(hostname).get_dict()

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


