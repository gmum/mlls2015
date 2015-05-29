def run_experiment(name, **kwargs):

    if not hasattr(globals(), name):
        raise ValueError("Not found module")

    if not hasattr(globals()[name], "ex"):
        raise ValueError("Found module is not an experiment?")

    ex = getattr(globals()[name], "ex")