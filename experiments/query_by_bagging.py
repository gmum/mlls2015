from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from get_data import get_data, get_splitted_data
from models.active_model import ActiveModel
from models.strategy import *
from models.utils import ObstructedY

from misc.config import c
data_dir = c["DATA_DIR"]

seed = 777
comp = [['5ht7', 'ExtFP']]
loader = ["get_splitted_data",
          {"n_folds": 2,
           "seed": seed,
           "test_size": 0.0}]
preprocess_fncs = []

folds, test_data, data_desc = get_data(comp, loader, preprocess_fncs).values()[0]
print data_desc

X = folds[0]['X_train']
y = ObstructedY(folds[0]['Y_train'])

X_test = folds[0]['X_valid']
y_test = folds[0]['Y_valid']

# ===

strategy_args = {'batch_size': 10, 'seed': seed, 'method': 'KL', 'n_bags': 5}
svm = SVC(C=0.01, kernel='linear', probability=True)

model = ActiveModel(strategy=query_by_bagging, base_model=svm)
model.fit(X, y, strategy_args=strategy_args)
p = model.predict(X_test)
print accuracy_score(p, y_test)

# ===

strategy_args = {'batch_size': 10, 'seed': seed, 'method': 'entropy', 'n_bags': 5}
svm = SVC(C=0.01, kernel='linear')

model = ActiveModel(strategy=query_by_bagging, base_model=svm)
model.fit(X, y, strategy_args=strategy_args)
p = model.predict(X_test)
print accuracy_score(p, y_test)