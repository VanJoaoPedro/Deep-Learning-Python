from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing import image
import numpy as np

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))


classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

generator_train = ImageDataGenerator(rescale=1./255, rotation_range=7,
                                     horizontal_flip=True, shear_range=0.2,
                                     height_shift_range=0.07, zoom_range=0.2)
generator_test = ImageDataGenerator(rescale=1./255)

base_train = generator_train.flow_from_directory(
    '/home/van/Deep-Learning-Python/Redes-neurais-convolucionais/Cats_Dogs/dataset/training_set',
    target_size=(64, 64), batch_size=32, class_mode='binary')

base_test = generator_test.flow_from_directory(
    '/home/van/Deep-Learning-Python/Redes-neurais-convolucionais/Cats_Dogs/dataset/test_set',
    target_size=(64, 64), batch_size=32, class_mode='binary')


classifier.fit_generator(base_train, steps_per_epoch=4000/16,
                         epochs=10, validation_data=base_test, 
                         validation_steps=1000/16)

image_test = image.load_img(
    '/home/van/Deep-Learning-Python/Redes-neurais-convolucionais/Cats_Dogs/morg.png',
    target_size=(64, 64))
image_test = image.img_to_array(image_test)
image_test /=255
image_test = np.expand_dims(image_test, axis=0)

prevision = classifier.predict(image_test)
base_train.class_indices

image_test2 = image.load_img( 
    '/home/van/Deep-Learning-Python/Redes-neurais-convolucionais/Cats_Dogs/dog.jpg',
    target_size=(64, 64))

image_test2 = image.img_to_array(image_test2)
image_test2 /= 255 
image_test2 = np.expand_dims(image_test2, axis=0)

prevision2 = classifier.predict(image_test2)        