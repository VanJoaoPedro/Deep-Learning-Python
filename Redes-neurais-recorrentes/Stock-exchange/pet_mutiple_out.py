from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

base = pd.read_csv(
    '/home/van/Deep-Learning-Python/Redes-neurais-recorrentes/Stock-exchange/CSV/petr4-treinamento.csv')
base = base.dropna()
base_train = base.iloc[:, 1:2].values
base_val_max = base.iloc[:, 2:3].values

normalizer = MinMaxScaler(feature_range=(0, 1))
base_train_normalizer = normalizer.fit_transform(base_train)
base_val_max_normalizer = normalizer.fit_transform(base_val_max)

predictors = []
price_real1 = []
price_real2 = []
for i in range(90, 1242):
    predictors.append(base_train_normalizer[i-90:i, 0])
    price_real1.append(base_train_normalizer[i, 0])
    price_real2.append(base_train_normalizer[i, 0])
predictors, price_real1, price_real2 = np.array(
    predictors), np.array(price_real1), np.array(price_real2)
predictors = np.reshape(predictors, (predictors.shape[0], predictors.shape[1], 1))

price_real = np.column_stack((price_real1, price_real2))

regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True,
                   input_shape=(predictors.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=2, activation='linear'))

regressor.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
regressor.fit(predictors, price_real, epochs=50, batch_size=32)

base_test = pd.read_csv('/home/van/Deep-Learning-Python/Redes-neurais-recorrentes/Stock-exchange/CSV/petr4-teste.csv')
price_real_open = base_test.iloc[:, 1:2].values
preco_real_high = base_test.iloc[:, 2:3].values

base_complete = pd.concat((base['Open'], base_test['Open']), axis = 0)
inputs = base_complete[len(base_complete) - len(base_test) - 90:].values
inputs = inputs.reshape(-1, 1)
inputs = normalizer.transform(inputs)

X_test = []
for i in range(90, 112):
    X_test.append(input[i-90:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

prevision = regressor.predict(X_test)
prevision = normalizer.inverse_transform(previsoes)

   
plt.plot(price_real_open, color = 'red', label = 'Preço abertura real')
plt.plot(price_real_high, color = 'black', label = 'Preço alta real')

plt.plot(previsoes[:, 0], color = 'blue', label = 'Previsões abertura')
plt.plot(previsoes[:, 1], color = 'orange', label = 'Previsões alta')

plt.title('Previsão preço das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()


