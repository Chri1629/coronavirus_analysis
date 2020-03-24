import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from pandas.plotting import register_matplotlib_converters
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--output', type=str, required=True, help="Inserire la directory corrente")
args = parser.parse_args()

# Definition of interpolation function
def func(x, a, b, c):
    return a * np.exp(b * x) + c

def func2(x, a, b):
    return a * x + b

def time_plot(time, data): 
    fig, ax = plt.subplots()
    plt.plot(time, data, 'ko', label="Original Data")
    plt.plot(time, func(x, *popt), 'r-', label="Fitted Exponential")
    #plt.plot(time, func2(x, *popt2), 'r', color= 'blue', label="Fitted Line")
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
x = np.linspace(0, 1, len(regione['data']))
y = func(x, 2.5, 1.3, 0.5)
y2 = func2(x,  0,2)

popt, pcov = curve_fit(func, x, regione['totale_casi'])
popt2, pcov2 = curve_fit(func2, x, regione['totale_casi'])
print('--------------------------------------------------------------------------\n')
print('I parametri stimati dai minimi quadrati per la funzione esponenziale sono: ',popt, '\n')
print('--------------------------------------------------------------------------\n')
print('I parametri stimati dai minimi quadrati per la retta sono: ',popt2, '\n')
print('--------------------------------------------------------------------------\n')
print("La matrice delle covarianze dei parametri stimati dall'esponenziale sono: ",pcov, '\n')
print('--------------------------------------------------------------------------\n')
print('La matrice delle covarianze dei parametri stimati dalla retta sono: ',pcov2, '\n')
print('--------------------------------------------------------------------------\n')

# Some basics plot
time_plot(regione['data'], regione['totale_casi'])

