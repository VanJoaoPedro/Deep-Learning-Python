from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 5
np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
predictors = X_train.reshape(X_train.shape[0], 28, 28, 1)
predictors = predictors.astype('float32')
predictors /= 255
group = np_utils.to_categorical(y_train, 10)

kflod = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=seed)
results = []

for id_train, id_test in kflod.split(predictors, np.zeros(shape=(group.shape[0], 1))):
    classifier = Sequential()
    classifier.add(
        Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=10, activation='softmax'))
    classifier.compile(loss='categorical_crossentropy',
                       optimizer='adam', metrics=['accuracy'])
    classifier.fit(predictors[id_train], group[id_train],
                   batch_size=128, epochs=5)
    precision = classifier.evaluate(predictors[id_test], group[id_test])
    results.append(precision[1])
