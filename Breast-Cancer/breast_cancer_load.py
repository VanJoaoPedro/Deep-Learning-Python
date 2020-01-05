import numpy as np
import pandas as pd
from keras.models import model_from_json

file = open('classifier_breast.json', 'r')
structure_network = file.read()
file.close()

classifier = model_from_json(structure_network)
classifier.load_weights('classifier_breast.h5') 

new = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20,
                 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                 0.03, 0.007, 23.15, 16.14, 178, 2018, 0.14, 0.185, 0.84, 158, 0.363]])

prevision = classifier.predict(new)
prevision = (prevision > 0.5)

predictors = pd.read_csv(
    '/home/van/Deep-Learning/Breast-Cancer/CSV/entradas.csv')
group = pd.read_csv(
    '/home/van/Deep-Learning/Breast-Cancer/CSV/saidas.csv')

classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
result = classifier.evaluate(predictors, group)