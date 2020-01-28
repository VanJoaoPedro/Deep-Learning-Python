import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
plt.imshow(X_train[0], cmap='gray')
plt.title('Class' + str(y_train[0]))

predictors_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
predictors_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

predictors_train = predictors_train.astype('float32')
predictors_test = predictors_test.astype('float32')

predictors_train /= 255
predictors_test /= 255

group_train = np_utils.to_categorical(y_train, 10)
group_test = np_utils.to_categorical(y_test, 10)

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3),
                      input_shape=(28, 28, 1),
                      activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))


classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=10, activation='softmax'))
classifier.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])
classifier.fit(predictors_train, group_train,
               batch_size=128, epochs=5,
               validation_data=(predictors_test, group_test))

results = classifier.evaluate(predictors_test, group_test)

plt.imshow(X_test[0], cmap='gray')
test_image = X_test[0].reshape(1, 28, 28, 1)
test_image = test_image.astype('float32')
test_image /= 255
prevision = classifier.predict(test_image)
result = np.argmax(prevision)
