from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base = pd.read_csv(
    '/home/van/Deep-Learning-Python/Redes-neurais-recorrentes/Stock-exchange/CSV/petr4-treinamento.csv')
base = base.dropna()
base_train = base.iloc[:, 1:7].values

normalizer = MinMaxScaler(feature_range=(0, 1))
base_train_normalizer = normalizer.fit_transform(base_train)

predictors = []
price_real = []
for i in range(90, 1242):
    predictors.append(base_train_normalizer[i-90:i, 0:6])
    price_real.append(base_train_normalizer[i, 0])
predictors, price_real = np.array(predictors), np.array(price_real)

regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True,
                   input_shape=(predictors.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation='sigmoid'))
regressor.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

earlystopping = EarlyStopping(
    monitor='loss', min_delta=1e-10, patience=10, verbose=1)
reduceLRO = ReduceLROnPlateau(
    monitor='loss', factor=0.2, patience=5, verbose=1)
modelcheckpoint = ModelCheckpoint(
    filepath='weigths.h5', monitor='loss', save_best_only=True, verbose=1)

regressor.fit(predictors, price_real, epochs=100, batch_size=32,
              callbacks=[earlystopping, reduceLRO, modelcheckpoint])

base_test = pd.read_csv(
    '/home/van/Deep-Learning-Python/Redes-neurais-recorrentes/Stock-exchange/CSV/petr4-teste.csv')
price_real_test = base_test.iloc[:, 1:2].values
frames = [base, base_test]
base_complete = pd.concat(frames)
base_complete = base_complete.drop('Date', axis=1)

inputs = base_complete[len(base_complete) - len(base_test) - 90:].values
inputs = normalizer.transform(inputs)

X_test = []
for i in range(90, 112):
    X_test.append(inputs[i-90:i, 0:6])
X_test = np.array(X_test)

prevision = regressor.predict(X_test)

normalizer_prevision = MinMaxScaler(feature_range=(0, 1))
normalizer_prevision.fit_transform(base_train[:, 0:1])
prevision = normalizer_prevision.inverse_transform(prevision)

plt.plot(price_real_test, color='orange', label='Price Real')
plt.plot(prevision, color='green', label='Prevision')
plt.title('Prevision Price Stock Exchange')
plt.xlabel('Time')
plt.ylabel('Vall Yahoo')
plt.legend()
plt.show()
