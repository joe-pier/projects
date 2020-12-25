# insieme al notebook lascio anche il file python. Non è un grande modello e forse ci
# sono anche errorima almeno mi sono divertito a farlo e applicare un po' di quello che ho studiato da autodidatta!

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers


import time

ds = pd.read_csv('BSDS_November.csv', sep=',')

y = ds.iloc[:, -1].values
X = ds.iloc[:, 0:9].values

y = np.where(y == 'Yes', 1, 0)
gender = pd.get_dummies(X[:, 0])
partner = pd.get_dummies(X[:, 1])
PhoneService = pd.get_dummies(X[:, 2])
internet_service = pd.get_dummies(X[:, 3])
online_security = pd.get_dummies(X[:, 4])
contract = pd.get_dummies(X[:, 5])
PaymentMethod = pd.get_dummies(X[:, 6])
month_charge = pd.DataFrame(X[:, 7])
total_charge = pd.DataFrame(X[:, 8])
churn = pd.DataFrame(y)

df = {}

df['gender'] = gender
df['partner'] = partner
df['Phoneservice'] = PhoneService
df['internet service'] = internet_service
df['online security'] = online_security
df['contract'] = contract
df['payment method'] = PaymentMethod
df['monthlycharges'] = month_charge
df['totalcharges'] = total_charge
df['churn'] = churn

for i in range(0, len(df['totalcharges'][0][:])):
    try:
        df['totalcharges'][0][i] = float(df['totalcharges'][0][i])
        df['monthlycharges'][0][i] = float(df['monthlycharges'][0][i])
    except:
        df['totalcharges'][0][i] = np.nan
        df['monthlycharges'][0][i] = np.nan

df = pd.concat(df.values(), axis=1, ignore_index=False)
df = pd.DataFrame(df)
labels = ['Female', 'Male', 'partner_No', 'partner_Yes', 'phone_service_No', 'phone_service_Yes', 'internet_DSL',
          'internet_Fiber_optic', 'internet_No', 'online_sec_No', 'No_internet_service', 'Yes', 'Month-to-month',
          'One year', 'Two year', 'Bank transfer(automatic)', 'Credit card(automatic)', 'Electronic_check',
          'Mailed_check', 'month_charge', 'total_charge', 'churn']

df.set_axis(labels, axis=1, inplace=True)
df.fillna(df.median(), inplace=True)


# print(df['churn'].value_counts())


df.iloc[:, -2] = scale(df.iloc[:, -2])
df.iloc[:, -3] = scale(df.iloc[:, -3])

# SALVA LE MODIFICHE
df.to_csv('output.csv', sep=',', mode='w', header=True, index=False)





##################################################################################################################

y_ = df.iloc[:, -1].values
X_ = df.iloc[:, 0:21].values

X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=0.95, random_state=True)

Inputs = keras.Input(shape=21)
x = layers.Dense(45, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0001),activation="relu", kernel_initializer= initializers.RandomNormal())(Inputs)
x = layers.Dense(35, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0001),activation="relu", kernel_initializer= initializers.RandomNormal())(x)
x = layers.Dense(30, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0001),activation="relu", kernel_initializer= initializers.RandomNormal())(x)
x = layers.Dense(15, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0001),activation="relu", kernel_initializer= initializers.RandomNormal())(x)
outputs = layers.Dense(1, activation="relu",kernel_initializer= initializers.RandomNormal())(x)

model = keras.Model(inputs=Inputs, outputs=outputs)

#model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0003), loss=keras.losses.BinaryCrossentropy(), metrics=keras.metrics.binary_accuracy)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00002,beta_1=0.95,beta_2=0.999,epsilon=1e-07,amsgrad=False), loss=keras.losses.BinaryCrossentropy(), metrics=keras.metrics.binary_accuracy)



training = model.fit(X_train, y_train, batch_size=100, epochs=150, verbose=2, validation_data=(X_test, y_test))
#print(training.history.keys())

# plottiamo l'evoluzione della LOSS e della ACCURACY
plt.plot(training.history['binary_accuracy'])
plt.plot(training.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



print('\n----------------------------------------------------------\n')
print('\nMETRICS\n')

#TESTING THE MODEL with ACCURACY
print('\n--TEST--')
test_results = model.evaluate(X_test, y_test, batch_size=1000)
print('\n--TRAIN--')
train_results = model.evaluate(X_train, y_train, batch_size=1000)





#TESTING THE MODEL
p_test = model.predict(X_test)
p_train = model.predict(X_train)
p_test_ = np.where(p_test >= 0.5, 1, 0)
y_test_ = np.where(y_test >= 0.5, 1, 0)

confusion_matrix = confusion_matrix(y_test_, p_test_)
print('\n--CONFUSION MATRIX--')
print(confusion_matrix)
sns.heatmap(confusion_matrix, annot=False)
plt.show()






print('\n')
#print(y_test[0:100])
#print(p_test_[0:100])

data_frame_predictions= zip(y_test[0:1000],p_test_[0:1000])
data_frame_predictions = pd.DataFrame(data_frame_predictions)
labels = ['target', 'previsione']

data_frame_predictions.set_axis(labels, axis=1, inplace=True)
#print(data_frame_predictions)
data_frame_predictions.to_csv('previsioni.csv')

model.summary()

print('\n il cmd si chiuderà tra 60 secondi')


time.sleep(60)
