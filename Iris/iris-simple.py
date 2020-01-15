import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

base = pd.read_csv('/home/van/Deep-Learning-Python/Iris/iris.csv')
predictors = base.iloc[:, 0:4].values
group = base.iloc[:, 4].values

labelencoder = LabelEncoder()
group = labelencoder.fit_transform(group)
group_dummy = np_utils.to_categorical(group)
# Iris Setosa        1 | 0 | 0
# Iris Virginica     0 | 1 | 0
# Iris Versicolor    0 | 0 | 1

predictores_train, predictores_test, group_train, group_test = train_test_split(
    predictors, group_dummy, test_size=0.25)

classifier = Sequential()
classifier.add(Dense(units=4, activation='relu', input_dim=4))
classifier.add(Dense(units=4, activation='relu'))
classifier.add(Dense(units=3, activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[
                   'categorical_accuracy'])
classifier.fit(predictores_train, group_train, batch_size=10, epochs=1000)
