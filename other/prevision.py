# Import all the useful libraries

import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from pandas.plotting import register_matplotlib_converters
import argparse
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import scipy.odr as odr 
from keras.activations import relu
from keras.activations import tanh
import matplotlib.pyplot as plt 
import matplotlib as mp
import itertools
import random
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dropout
from array import array
from sklearn import metrics
from keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn import svm, datasets
from scipy import interp
from keras.callbacks import History 
history = History()
from keras.callbacks import *
from sklearn.externals import joblib
from scipy.stats import norm
import scipy.stats
from keras.models import Sequential, load_model
from keras.activations import relu
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam, SGD
from keras import metrics
from keras import losses



#Start the code
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--output', type=str, required=True, help="Inserire la directory corrente")
args = parser.parse_args()
# Plots the loss function
def loss_plotter(history):
    fig = plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label = "train_loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.legend(['loss_train', 'loss_validation'], loc='upper right', fontsize = 20)
    plt.xticks( fontsize = 20)
    plt.yticks( fontsize = 20)
    plt.xlabel('epochs', size = 35)
    plt.ylabel('loss', size = 35)
    plt.show()
    fig.savefig(args.output+ '/loss_plot.pdf', bbox_inches='tight')
# Definition of interpolation function
def esponenziale(x, a, b, c):
    return a * np.exp(b * x) + c

def retta(x, a, b):
    return a * x + b

def logistica(x, a, b):
    return 1/(1 + np.exp(a * x)) + b

def time_plot(time, data): 
    fig, ax = plt.subplots()
    plt.plot(time, data, 'ko', label="Original Data")
    plt.plot(time, esponenziale(x, *popt), 'r-', label="Esponenziale")
    plt.plot(time, retta(x, *popt2), '+-', color = "blue", label="Retta")
    plt.plot(time, logistica(x, *popt3), '-', color = "green", label="Logistica")
    #plt.plot(time, predictions, 'r', color= 'blue', label="Fitted Line")
    plt.xticks(rotation = 45, fontsize=8)
    plt.legend()
    plt.show()
    fig.savefig(args.output + '/time_plot.png', bbox_inches='tight', dpi = 300)

# Import the dataset.

pd.options.mode.chained_assignment = None 
# Ricordiamo che gli attributi del nostro dataset sono i seguenti:
# data,stato,codice_regione,denominazione_regione,lat,long,ricoverati_con_sintomi,terapia_intensiva,totale_ospedalizzati
# ,isolamento_domiciliare,totale_attualmente_positivi,nuovi_attualmente_positivi,dimessi_guariti,deceduti,totale_casi,tamponi
dati = pd.read_csv("dati.csv")
# Rimuovo l'ora nel timestamp che tanto non è necessaria

regione = dati.loc[dati['regione'] == 'Lombardia']
regione['data'] = pd.to_datetime(regione['data'])
pd.plotting.register_matplotlib_converters()

# Simple exponential regression
x = np.linspace(0, len(regione['data']), len(regione['data']))


y = esponenziale(x, 2.5, 1.3, 0.5)
y2 = retta(x,  0, 2)
y3 = logistica(x, 10 ,2)

popt, pcov = curve_fit(esponenziale, x, regione['totale_casi'])
popt2, pcov2 = curve_fit(retta, x, regione['totale_casi'])
popt3, pcov3 = curve_fit(logistica, x, regione['totale_casi'])
print('--------------------------------------------------------------------------\n')
print('I parametri stimati dai minimi quadrati per la funzione esponenziale sono: ',popt, '\n')
print('--------------------------------------------------------------------------\n')
print('I parametri stimati dai minimi quadrati per la retta sono: ',popt2, '\n')
print('--------------------------------------------------------------------------\n')
print('I parametri stimati dai minimi quadrati per la rlogistica sono: ',popt3, '\n')
print('--------------------------------------------------------------------------\n')
print("La matrice delle covarianze dei parametri stimati dall'esponenziale sono: ",pcov, '\n')
print('--------------------------------------------------------------------------\n')
print('La matrice delle covarianze dei parametri stimati dalla retta sono: ',pcov2, '\n')
print('--------------------------------------------------------------------------\n')
print('La matrice delle covarianze dei parametri stimati dalla retta sono: ',pcov3, '\n')
print('--------------------------------------------------------------------------\n')

time_plot(regione['data'], regione['totale_casi'])

# MACHINE LEARNING
'''
# Initializing parameters
decay_rate = 0
learning_rate = 1e-3
epoch = 100
patience = "0.001:5"
initial_epochs = 0
batch_size = 5
print("starting the analysis with the following parameters: \n learning rate:", f'{learning_rate}' "\n",
"epoch:", f'{epoch}', "\n", "patience:", f'{patience}' "\n", "batch size:", f'{batch_size}' "\n",)
print('--------------------------------------------------------------------------\n')

# Preparing data for the analysis:
print('Preparing the data for the analysis')

x_train = x[0:len(regione['data'])-10].reshape(-1, 1)
y_train = regione['totale_casi'][0: len(regione['totale_casi']) -10 ]

x_validation = x[len(regione['data'])-10:len(regione['data'])-6].reshape(-1, 1)
y_validation = regione['totale_casi'][len(regione['totale_casi']) -10: len(regione['totale_casi']) - 6]

x_test = x[len(regione['data'])-6:len(regione['data'])].reshape(-1, 1)
y_test = regione['totale_casi'][len(regione['totale_casi']) - 6: len(regione['totale_casi'])]
print('done')
print('--------------------------------------------------------------------------\n')

print("training of the model")

model = Sequential()

model.add(Dense(units=10000, input_dim=x_train.shape[1], activation="relu"))
#model.add(Dropout(0.2))
model.add(Dense(units=10000,activation="relu"))
#model.add(Dropout(0.2))

model.add(Dense(1))
        
model.compile(loss=losses.mean_squared_error,
              optimizer='adam',
              metrics=[metrics.mean_absolute_error])


auto_save = ModelCheckpoint(args.output + "/current_model", monitor='val_loss',
                    verbose=1, save_best_only=True, save_weights_only=False,
                    mode='auto', period=2)

min_delta = float(patience.split(":")[0])
p_epochs = int(patience.split(":")[1])
early_stop = EarlyStopping(monitor='val_loss', min_delta=min_delta,
                               patience=p_epochs, verbose=1)

def reduceLR (epoch):
    return learning_rate * (1 / (1 + epoch*decay_rate))

lr_sched = LearningRateScheduler(reduceLR, verbose=1)
csv_logger = CSVLogger(args.output + '/training.log')


print(">>> Training...")
W_val = np.ones(x_validation.shape[0])
history = model.fit(x_train, y_train,
                        validation_data = (x_validation, y_validation),
                        epochs=epoch, initial_epoch=initial_epochs,
                        batch_size=batch_size, shuffle=True,
                        callbacks=[auto_save, early_stop, lr_sched, csv_logger])

print('--------------------------------------------------------------------------\n')
print("Saving the prediction")
predictions = model.predict(x_test,batch_size=2048)
predictions = np.concatenate(predictions)
#predictions = scaler_truth.inverse_transform(predictions)

# Some basics plot
predictions = np.concatenate((y_train, y_validation, predictions), axis=None)

loss_plotter(history)
'''