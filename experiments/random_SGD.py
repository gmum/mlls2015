from get_data import get_data
from models.active_model import ActiveModel
from models.strategy import random_query
from models.utils import ObstructedY

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


seed = 666
strategy_args = {'batch_size': 10, 'seed': seed}
comp = [['5ht7', 'ExtFP']]
loader = ["get_splitted_data",
          {"n_folds": 3,
           "seed":666,
           "test_size":0.1}]
preprocess_fncs = []


sgd = SGDClassifier(random_state=seed)
model = ActiveModel(strategy=random_query, base_model=sgd)

folds, test_data, data_desc = get_data(comp, loader, preprocess_fncs).values()[0]
print data_desc

X = folds[0]['X_train']
y = ObstructedY(folds[0]['Y_train'])

X_test = folds[0]['X_valid']
y_test = folds[0]['Y_valid']

model.fit(X, y, strategy_args=strategy_args, verbose=True)
p = model.predict(X_test)
print "acc:", accuracy_score(p, y_test)