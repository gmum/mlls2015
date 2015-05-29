from nose import tools
from nose.case import TestBase
import kaggle_ninja
from kaggle_ninja import cached, get_last_cached, get_all_by_search_arg

class BasicTests(TestBase):
    def test(self):
        arg2 = {"x": {"w":20}}
        arg1 = 1
        @cached(skip_args=["arg2.x.w"])
        def f1(arg1, arg2):
            return 12

        # skip_args works
        arg2["x"]["w"] = 32
        f1(arg1=arg1, arg2=arg2, force_reload=True)

        # should not reload!
        arg2["x"]["w"] = 30
        f1(arg1=arg1, arg2=arg2)

        # cached_ram works
        arg2 = {"x":{"w":20}}
        arg1 = 1
        @cached(skip_args=["arg2.x.w"], cached_ram=True)
        def f2(arg1, arg2):
            return {"x":13}

        d = f2(arg1=arg1, arg2=arg2)
        d['x'] = 14
        d_modified = f2(arg1=arg1, arg2=arg2)
        tools.assert_true(d_modified["x"] == 14)

    def test_search_args(self):
        @cached(search_args=["experiment_name"])
        def h(a=1, b=2, experiment_name=""):
            return a+b
        h(force_reload=True, a=2,b=2, experiment_name="ww")
        assert get_last_cached("h")[0][1]['experiment_name'] == 'ww'


        @cached(search_args=["experiment_name"])
        def h(a=1, b=2, experiment_name=""):
            return a+b
        h(a=2,b=2, experiment_name="ww")
        h(a=3,b=4, experiment_name="ww")
        h(a=10,b=1, experiment_name="ww")
        h(a=10,b=1, experiment_name="wwx")
        assert get_last_cached("h")[0][1]['experiment_name'] == 'ww'
        assert len(get_all_by_search_arg("h", query_search_args={"experiment_name":"w.*"})) == 3
        assert len(get_all_by_search_arg("h", query_args={"a": lambda a: a>4}, query_search_args={"experiment_name":"w.*"})) == 1