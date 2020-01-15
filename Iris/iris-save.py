import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('/home/van/Deep-Learning-Python/Iris/iris.csv')
predictors = base.iloc[:, 0:4].values
group = base.iloc[:, 4].values

labelencoder = LabelEncoder()
group = labelencoder.fit_transform(group)
group_dummy = np_utils.to_categorical(group)

# Iris Setosa        1 | 0 | 0
# Iris Virginica     0 | 1 | 0
# Iris Versicolor    0 | 0 | 1

classifier = Sequential()
classifier.add(Dense(units=8, activation='relu', input_dim=4))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=8, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=3, activation='softmax'))
classifier.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=[
                   'categorical_accuracy'])

classifier.fit(predictors, group_dummy, batch_size=5, epochs=1000)

classifier_json = classifier.to_json()
with open('classifier_iris.json', 'w') as json_file:
    json_file.write(classifier_json)
classifier.save_weights('classifier_iris.h5')

new = np.array([[0.122, 1.233, 1, 0.23]])

prevision = classifier.predict(new)
prevision = (prevision > 0.5)

