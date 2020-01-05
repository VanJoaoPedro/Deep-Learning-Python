import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

predictors = pd.read_csv(
    '/home/van/Deep-Learning/Breast-Cancer/CSV/entradas.csv')
group = pd.read_csv(
    '/home/van/Deep-Learning/Breast-Cancer/CSV/saidas.csv')


def Create_network():
    classifier = Sequential()
    classifier.add(Dense(units=32, activation='relu',
                         kernel_initializer='TruncatedNormal', input_dim=30))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(units=32, activation='relu',
                         kernel_initializer='TruncatedNormal'))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(units=16, activation='relu',
                         kernel_initializer='TruncatedNormal'))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(units=8, activation='relu',
                         kernel_initializer='TruncatedNormal'))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(units=1, activation='sigmoid'))
 
    optimizer = keras.optimizers.Adam(lr=0.002, decay=0.0002, clipvalue=0.5)
    classifier.compile(optimizer=optimizer,
                       loss='binary_crossentropy', metrics=['binary_accuracy'])

    return classifier


classifier = KerasClassifier(build_fn=Create_network, epochs=100, batch_size=10)
result = cross_val_score(estimator=classifier, X=predictors,
                         y=group, cv=10, scoring='accuracy')

mean = result.mean()
deviation = result.std()

print(mean)
print(deviation)
