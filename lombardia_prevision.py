import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import matplotlib.dates as mdates
import plotly.express as px
from scipy.optimize import curve_fit

# Definition of interpolation function
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# import the dataset.
dati = pd.read_csv("dati.csv")
dati = dati.drop('denominazione_regione', axis = 1)

milano = dati.loc[dati['denominazione_provincia'] == 'Milano']
print(milano)

milano['data'] = pd.to_datetime(milano['data'])


# simple exponential regression
x = np.linspace(0,1,16)
y = func(x, 2.5, 1.3, 0.5)

popt, pcov = curve_fit(func, x, milano['totale_casi'])
print(popt, pcov)
# Some basics plot

fig, ax = plt.subplots()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
plt.plot(milano['data'], milano['totale_casi'], 'ko', label="Original Data")
plt.plot(milano['data'], func(x, *popt), 'r-', label="Fitted Curve")
plt.legend()
plt.show()
'''
fig = px.line(milano, x='data', y='totale_casi', var_name = 'Reale')
fig.add_scatter(x=milano['data'], y=func(x, *popt), var_name = 'Fittato')
fig.show()
'''