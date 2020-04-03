import pandas as pd


dati = pd.read_csv("dpc-covid19-ita-regioni.csv")
regioni = pd.read_csv("regioni.csv")
# devo unire i dati del trentino
df_r = dati.loc[(dati['denominazione_regione'] == "P.A. Bolzano") | (dati['denominazione_regione'] == "P.A. Trento")]
df_trentino = df_r.groupby("data").sum()
df_trentino['denominazione_regione'] = "Trentino Alto Adige" 
df_trentino['lat'] = 46.068935
df_trentino['long'] = 11.121231
df_trentino = df_trentino.reset_index()
dati = dati.loc[(dati['denominazione_regione'] != "P.A. Trento") & (dati['denominazione_regione'] != "P.A. Bolzano")]
dati_fix = pd.concat([dati, df_trentino], sort=False)
dati_fix['stato'] = "ITA"
dati = dati_fix.drop(dati_fix[["note_en", "note_it"]], axis=1)

dati_giusti = pd.merge(dati, regioni, left_on = 'denominazione_regione', right_on = 'regione')
dati_giusti = dati_giusti.drop('denominazione_regione', axis = 1)
dati_giusti.to_csv('dati.csv')
print("FATTO yeeeeeeeeeee")