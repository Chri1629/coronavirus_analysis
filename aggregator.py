import pandas as pd


dati = pd.read_csv("dpc-covid19-ita-regioni.csv")
regioni = pd.read_csv("regioni.csv")
dati_giusti = pd.merge(dati, regioni, left_on = 'denominazione_regione', right_on = 'regione')
dati_giusti = dati_giusti.drop('denominazione_regione', axis = 1)
dati_giusti.to_csv('dati.csv')