from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from get_data import get_data, get_splitted_data
from models.active_model import ActiveLearningExperiment
from models.strategy import *
from models.utils import ObstructedY

from misc.config import c
data_dir = c["DATA_DIR"]

seed = 666
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

sgd = SGDClassifier(random_state=seed, alpha=1)
model = ActiveLearningExperiment(strategy=uncertainty_sampling, base_model=sgd, batch_size=10)

model.fit(X, y)
p = model.predict(X_test)
print accuracy_score(p, y_test)

# ===

from sklearn.svm import SVC
svm = SVC(C=0.01, kernel='linear', probability=True)

model = ActiveLearningExperiment(strategy=uncertainty_sampling, base_model=svm, batch_size=10)
model.fit(X, y)
p = model.predict(X_test)
print accuracy_score(p, y_test)