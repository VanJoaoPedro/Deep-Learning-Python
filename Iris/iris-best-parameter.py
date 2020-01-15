import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
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


def create_network():
    classifier = Sequential()
    classifier.add(Dense(units=8, activation='relu', input_dim=4))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=8, activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=3, activation='softmax'))
    classifier.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=[
                       'categorical_accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=create_network, epochs=1000, batch_size=5)

results = cross_val_score(estimator=classifier,
                          X=predictors, y=group, cv=10, scoring='accuracy')
mean = results.mean()
deviation = results.std
