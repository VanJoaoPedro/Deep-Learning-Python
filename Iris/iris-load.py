import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

file = open('/home/van/Deep-Learning-Python/Iris/classifier_iris.json')
structure_network = file.read()
file.close()

classifier = model_from_json(structure_network)
classifier.load_weights(
    '/home/van/Deep-Learning-Python/Iris/classifier_iris.h5')

new = np.array([[0.122, 1.233, 1, 0.23]])

prevision = classifier.predict(new)
prevision = (prevision > 0.5)

base = pd.read_csv('/home/van/Deep-Learning-Python/Iris/iris.csv')
predictors = base.iloc[:, 0:4].values
group = base.iloc[:, 4].values

labelencoder = LabelEncoder()
group = labelencoder.fit_transform(group)
group_dummy = np_utils.to_categorical(group)

classifier.compile(loss='categorical_crossentropy',
                   optimizer='adadelta', metrics=['categorical_accuracy'])
result = classifier.evaluate(predictors, group_dummy)
