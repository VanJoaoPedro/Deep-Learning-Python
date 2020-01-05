import pandas as pd
import keras 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

predictors = pd.read_csv('/home/van/Deep-Learning/Breast-Cancer/CSV/entradas.csv')
group = pd.read_csv('/home/van/Deep-Learning/Breast-Cancer/CSV/saidas.csv')
predictors_train, predictors_test, group_train, group_test = train_test_split(predictors, group, test_size=0.25)

classifier = Sequential()
classifier.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30 ))
classifier.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

optimizer = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
classifier.fit(predictors_train, group_train, batch_size = 10, epochs = 50)

weigths0 = classifier.layers[0]
weigths1 = classifier.layers[1]
weigths2 = classifier.layers[2]

predictors = classifier.predict(predictors_test)
predictors = (predictors > 0.5)

accuracy = accuracy_score(group_test, predictors)
matrix = confusion_matrix(group_test, predictors)

result = classifier.evaluate(predictors_test, group_test)

