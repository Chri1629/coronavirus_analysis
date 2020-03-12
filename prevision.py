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
    return a * np.exp(-b * x) + c

def time_plot(time, data): 
    fig, ax = plt.subplots()
    plt.plot(time, data, 'ko', label="Original Data")
    plt.plot(time, func(x, *popt), 'r-', label="Fitted Curve")
    hfmt = mdates.DateFormatter('%d-%m')
    ax.xaxis.set_major_formatter(hfmt)
    plt.xticks(rotation = 90, fontsize=8)
    plt.legend()
    plt.show()
    fig.savefig(args.output + '/time_plot.png', bbox_inches='tight', dpi = 600)

# Import the dataset.
dati = pd.read_csv("dati.csv")

regione = dati.loc[dati['regione'] == 'Lombardia']
print(regione)

regione['data'] = pd.to_datetime(regione['data'])

pd.plotting.register_matplotlib_converters()
# Simple exponential regression
x = np.linspace(0,1,len(regione['data']))
y = func(x, 2.5, 1.3, 0.5)

popt, pcov = curve_fit(func, x, regione['totale_casi'])
print('--------------------------------------------------------------------------')
print('I parametri stimati sono: ',popt, pcov)

print('--------------------------------------------------------------------------')
# Some basics plot

time_plot(regione['data'], regione['totale_casi'] )

