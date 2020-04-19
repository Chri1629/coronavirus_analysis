import pandas as pd


dati = pd.read_csv("dpc-covid19-ita-regioni.csv")
dati_p = pd.read_csv("dpc-covid19-ita-province.csv")
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
print("Dati regioni SISTEMATO")

#sistemo le provincie
dati_p = dati_p[dati_p['denominazione_provincia'] != "In fase di definizione/aggiornamento"] # drop in fase di aggiornamento
df_tb = dati_p.loc[(dati_p['denominazione_regione'] == "P.A. Bolzano") | (dati_p['denominazione_regione'] == "P.A. Trento")]
df_tb['denominazione_regione'] = "Trentino Alto Adige"
dati_p["denominazione_provincia"][dati_p['denominazione_regione'] == "Valle d'Aosta"] = "Valle d'Aosta"
#tolgo i dati del Trentino
dati_p = dati_p.loc[(dati_p['denominazione_regione'] != "P.A. Trento") & (dati_p['denominazione_regione'] != "P.A. Bolzano")]
dati_p_giusti = pd.concat([dati_p, df_tb], sort=False)
dati_p_giusti = dati_p_giusti.drop(dati_p_giusti[["note_en", "note_it"]], axis=1)
dati_p_giusti.to_csv('dati_p.csv')

print("Dati province SISTEMATO")