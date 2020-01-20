import pandas as pd
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

base = pd.read_csv('/home/van/Deep-Learning-Python/games/games.csv')
base = base.drop('Other_Sales', axis=1)
base = base.drop('Developer', axis=1)

base = base.drop('NA_Sales', axis=1)
base = base.drop('EU_Sales', axis=1)
base = base.drop('JP_Sales', axis=1)

base = base.dropna(axis=0)

base = base.loc[base['Global_Sales'] > 1]

base['Name'].value_counts()
name_games = base.Name
base = base.drop('Name', axis=1)

predictors = base.iloc[:, [0, 1, 2, 3, 5, 6, 7, 8, 9]].values
price_sell = base.iloc[:, 4].values

labelencoder = LabelEncoder()
predictors[:, 0] = labelencoder.fit_transform(predictors[:, 0])
predictors[:, 2] = labelencoder.fit_transform(predictors[:, 2])
predictors[:, 3] = labelencoder.fit_transform(predictors[:, 3])
predictors[:, 8] = labelencoder.fit_transform(predictors[:, 8])

onehotencorder = ColumnTransformer(transformers=[(
    "OneHot", OneHotEncoder(), [0, 2, 3, 8])], remainder='passthrough')
predictors = onehotencorder.fit_transform(predictors).toarray()

input_layer = Input(shape=(99,))
activation = Activation(activation='sigmoid')
hidden_layer1 = Dense(units=50, activation=activation)(input_layer)
hidden_layer2 = Dense(units=50, activation=activation)(hidden_layer1)
output_layer = Dense(units=1, activation='linear')(hidden_layer2)

regressor = Model(inputs=input_layer, outputs=[output_layer])
regressor.compile(optimizer='adam', loss='msle')
regressor.fit(predictors, price_sell, epochs=5000, batch_size=100)
prevision = regressor.predict(predictors)
