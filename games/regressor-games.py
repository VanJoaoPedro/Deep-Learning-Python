import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

base = pd.read_csv('/home/van/Deep-Learning-Python/games/games.csv')
base = base.drop('Other_Sales', axis=1)
base = base.drop('Global_Sales', axis=1)
base = base.drop('Developer', axis=1)
base = base.drop('Name', axis=1)

base = base.dropna(axis=0)
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

name_games = base.Name

predictors = base.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11]].values
sell_na = base.iloc[:, 4].values
sell_eu = base.iloc[:, 5].values
sell_jp = base.iloc[:, 6].values

labelencoder = LabelEncoder()
predictors[:, 0] = labelencoder.fit_transform(predictors[:, 0])
predictors[:, 2] = labelencoder.fit_transform(predictors[:, 2])
predictors[:, 3] = labelencoder.fit_transform(predictors[:, 3])
predictors[:, 8] = labelencoder.fit_transform(predictors[:, 8])

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [
                                  0, 2, 3, 8])], remainder='passthrough')
predictors = onehotencoder.fit_transform(predictors).toarray()

input_layer = Input(shape=(61,))
hidden_layer1 = Dense(units=32, activation='sigmoid')(input_layer)
hidden_layer2 = Dense(units=32, activation='sigmoid')(hidden_layer1)
output_layer1 = Dense(units=1, activation='linear')(hidden_layer2)
output_layer2 = Dense(units=1, activation='linear')(hidden_layer2)
output_layer3 = Dense(units=1, activation='linear')(hidden_layer2)

regressor = Model(inputs=input_layer,
                  outputs=[output_layer1, output_layer2, output_layer3])

regressor.compile(optimizer='adam', loss='mse')
regressor.fit(predictors, [sell_na, sell_eu, sell_jp],
              epochs=5000, batch_size=100)

prevision_na, prevision_eu, prevision_jp = regressor.predict(predictors)
