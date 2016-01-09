# -*- coding: utf-8 -*-
"""
 Simple adaptive enabled GridSearch
"""
from sklearn.grid_search import GridSearchCV
from six import iteritems

# NOTE: Class is violating convention that fit is refitting whole model
# and forgetting state. This makes it less likely to be extracted later
class AdaptiveGridSearchCV(GridSearchCV):

    def __init__(self, d, estimator, param_grid, scoring=None, fit_params=None, n_jobs=1,
                 iid=True, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score='raise'):
        self.d = d
        # Not using kwargs to be correctly cloneable
        super(AdaptiveGridSearchCV, self).__init__(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            fit_params=fit_params,
            n_jobs=n_jobs,
            iid=iid,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score)

    def fit(self, X, y):
        # Adaptive
        if getattr(self, "best_params_", None):
            # Construct smaller param grid
            original_param_grid = dict(self.param_grid)
            self.param_grid = {}
            for key, best_param in iteritems(self.best_params_):
                i = original_param_grid[key].index(best_param)
                if i != 0 and i != len(original_param_grid[key]) - 1:
                    self.param_grid[key] = [original_param_grid[key][j] for j in range(i-self.d, i+self.d+1)]
                elif i == 0:
                    self.param_grid[key] = [original_param_grid[key][j] for j in range(i, i+1+2*self.d)]
                elif i == len(self.param_grid[key]) - 1:
                    self.param_grid[key] = [original_param_grid[key][j] for j in range(i-2*self.d, i+1)]
                else:
                    assert False, "Not handled param_grid case"

            super(AdaptiveGridSearchCV, self).fit(X, y)
            # Trick with swapping to behave like a subclass of GridSearchCV
            self.param_grid = original_param_grid
        else:
            super(AdaptiveGridSearchCV, self).fit(X, y)

        return self.best_estimator_
