from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

predictor_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
predictor_train = predictor_train.astype('float32')
predictor_train /= 255

predictor_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
predictor_test = predictor_test.astype('float32')
predictor_test /= 255


group_train = np_utils.to_categorical(y_train, 10)
group_test = np_utils.to_categorical(y_test, 10)

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())

classifier.add(Dense(units=256, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=10, activation='softmax'))

classifier.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])
classifier.fit(predictor_train, group_train, batch_size=128, epochs=10,
               validation_data=(predictor_test, group_test), verbose=2)
