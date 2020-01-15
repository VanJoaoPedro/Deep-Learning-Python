import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('/home/van/Deep-Learning-Python/Iris/iris.csv')
predictors = base.iloc[:, 0:4].values
group = base.iloc[:, 4].values

labelencoder = LabelEncoder()
group = labelencoder.fit_transform(group)
group_dummy = np_utils.to_categorical(group)


def create_network(optimizer, activation, neurons):
    classifier = Sequential()
    classifier.add(Dense(units=neurons, activation=activation, input_dim=4))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=neurons, activation=activation))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=3, activation='softmax'))
    classifier.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=[
                       'accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=create_network)
parameter = {'batch_size': [5, 15],
             'epochs': [1000, 1500],
             'optimizer': ['adam', 'adadelta'],
             'activation': ['relu', 'selu'],
             'neurons': [4, 8]}
grid_search = GridSearchCV(
    estimator=classifier, param_grid=parameter, cv=5)
grid_search = grid_search.fit(predictors, group)
best_parameter = grid_search.best_params_
best_prevision = grid_search.best_score_
