import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

predictors = pd.read_csv(
    '/home/van/Deep-Learning/Breast-Cancer/CSV/entradas.csv')
group = pd.read_csv(
    '/home/van/Deep-Learning/Breast-Cancer/CSV/saidas.csv')


def Create_network(optimizer, loos, kernel_initializer, activation, neurons):
    classifier = Sequential()

    classifier.add(Dense(units=neurons, activation=activation,
                         kernel_initializer=kernel_initializer, input_dim=30))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(units=neurons, activation=activation,
                         kernel_initializer=kernel_initializer))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss=loos,
                       metrics=['binary_accuracy'])

    return classifier


classifier = KerasClassifier(build_fn=Create_network)
parameter = {'batch_size': [5, 15],
             'epochs': [5, 10],
             'optimizer': ['adam', 'sgd'],
             'loos': ['binary_crossentropy', 'hinge'],
             'kernel_initializer': ['random_uniform', 'normal'],
             'activation': ['relu', 'tanh'],
             'neurons': [8, 2]}

grid_search = GridSearchCV(
    estimator=classifier, param_grid=parameter, scoring='accuracy', cv=5)
grid_search = grid_search.fit(predictors, group)
best_parameter = grid_search.best_params_
best_prevision = grid_search.best_score_
