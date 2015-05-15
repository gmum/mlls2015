from nose import tools
import kaggle_ninja
from kaggle_ninja import cached

def test():
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
