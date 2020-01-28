from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

(X_train, y_train), (X_test, y_test) = mnist.load_data()
predictors_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
predictors_train = predictors_train.astype('float32')
predictors_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
predictors_test = predictors_test.astype('float32')
predictors_train /= 255
predictors_test /= 255
group_train = np_utils.to_categorical(y_train, 10)
group_test = np_utils.to_categorical(y_test, 10)

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='softmax'))
classifier.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

generator_train = ImageDataGenerator(rotation_range=7, horizontal_flip=True,
                                     shear_range=0.2, height_shift_range=0.07, zoom_range=0.2)
generator_test = ImageDataGenerator()

base_train = generator_train.flow(predictors_train, group_train, batch_size=128)
base_test = generator_test.flow(predictors_test, group_test, batch_size=128)

classifier.fit_generator(base_train, steps_per_epoch=60000/128, epochs=5, 
                         validation_data=base_test, validation_steps=1000/2)
