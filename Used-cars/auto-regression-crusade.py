import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

base = pd.read_csv(
    '/home/van/Deep-Learning-Python/Used-cars/autos.csv', encoding='ISO-8859-1')
base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)
base = base.drop('name', axis=1)
base = base.drop('seller', axis=1)
base = base.drop('offerType', axis=1)


i1 = base.loc[base.price <= 10]
i2 = base.loc[base.price > 350000]
base = base.loc[base.price < 350000]
base = base[base.price > 10]


values = {'vehicleType': 'limousine',
          'gearbox': 'manuell',
          'model': 'golf',
          'fuelType': 'benzin',
          'notRepairedDamage': 'nein'}

base = base.fillna(value=values)

predictors = base.iloc[:, 1:13].values
price_real = base.iloc[:, 0].values
labelencoder_predictors = LabelEncoder()
predictors[:, 0] = labelencoder_predictors.fit_transform(predictors[:, 0])
predictors[:, 1] = labelencoder_predictors.fit_transform(predictors[:, 1])
predictors[:, 3] = labelencoder_predictors.fit_transform(predictors[:, 3])
predictors[:, 5] = labelencoder_predictors.fit_transform(predictors[:, 5])
predictors[:, 8] = labelencoder_predictors.fit_transform(predictors[:, 8])
predictors[:, 9] = labelencoder_predictors.fit_transform(predictors[:, 9])
predictors[:, 10] = labelencoder_predictors.fit_transform(predictors[:, 10])


onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [
                                  0, 1, 3, 5, 8, 9, 10])], remainder='passthrough')
predictors = onehotencoder.fit_transform(predictors).toarray()


def create_network():
    regressor = Sequential()
    regressor.add(Dense(units=158, activation='relu', input_dim=316))
    regressor.add(Dense(units=158, activation='relu'))
    regressor.add(Dense(units=1, activation='linear'))
    regressor.compile(loss='mean_absolute_error', optimizer='adam',
                      metrics=['mean_absolute_error'])

    return regressor


regressor = KerasRegressor(build_fn=create_network, epochs=100, batch_size=300)

results = cross_val_score(estimator=regressor, X=predictors,
                          y=price_real, cv=10, scoring='neg_mean_absolute_error')

mean = results.mean()
deviation = results.std()
