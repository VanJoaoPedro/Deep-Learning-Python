from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

base = pd.read_csv(
    '/home/van/Deep-Learning-Python/Redes-neurais-recorrentes/Stock-exchange/CSV/petr4-treinamento.csv')
base = base.dropna()
base_train = base.iloc[:, 1:2].values
normalizer = MinMaxScaler(feature_range=(0, 1))
base_train_normalizer = normalizer.fit_transform(base_train)

predictors = []
price_real = []
for i in range(90, 1242):
    predictors.append(base_train_normalizer[i-90:i, 0])
    price_real.append(base_train_normalizer[i, 0])
predictors, price_real = np.array(predictors), np.array(price_real)

predictors = np.reshape(
    predictors, (predictors.shape[0], predictors.shape[1], 1))

regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True,
                   input_shape=(predictors.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=100, return_sequences=True,))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True,))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=25))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation='linear'))
regressor.compile(optimizer='rmsprop', loss='mse',
                  metrics=['mean_absolute_error'])
regressor.fit(predictors, price_real, epochs=100, batch_size=32)

base_test = pd.read_csv(
    '/home/van/Deep-Learning-Python/Redes-neurais-recorrentes/Stock-exchange/CSV/Base-Petrobras-greve/PETR4.SA.csv')

price_real_test = base_test.iloc[:, 1:2].values
base_complete = pd.concat((base['Open'], base_test['Open']), axis=0)
inputs = base_complete[len(base_complete) - len(base_test) - 90:].values

inputs = inputs.reshape(-1, 1)
inputs = normalizer.transform(inputs)

X_test = []
for i in range(90, 112):
    X_test.append(inputs[i-90:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
prevision = regressor.predict(X_test)
prevision = normalizer.inverse_transform(prevision)

plt.plot(price_real_test, color='orange', label='Price Real')
plt.plot(prevision, color='green', label='Prevision')
plt.title('Prevision Price Stock Exchange')
plt.xlabel('Time')
plt.ylabel('Vall Yahoo')
plt.legend()
plt.show()
