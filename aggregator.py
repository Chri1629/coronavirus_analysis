import pandas as pd

dati = pd.read_csv("dati.csv")
regioni = pd.read_csv("regioni.csv")
#print(dati)
#print(regioni)
dati_giusti = pd.merge(dati, regioni, left_on = 'denominazione_regione', right_on = 'regione')
dati_giusti.to_csv('dati.csv')