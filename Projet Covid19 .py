#!/usr/bin/env python
# coding: utf-8

# # Projet Covid19
# 

# Le 30 janvier 2020, l’OMS (Organisation Mondiale de la Santé) prononce l’état d’urgence de santé publique de portée internationale pour une maladie infectieuse émergente appelé la maladie de coronavirus 2019 ou Covid-19. À l’origine d’une pandémie mondiale, celle-ci apparait pour la première fois à Wuhan, une province de Hubei en Chine. Tous les pays du monde vont prendre des mesures plus ou moins fortes et efficaces pour essayer de réduire la propagation de ce virus. Des décisions fortes vont être prises comme le masque obligatoire en extérieur, des mises en places de quarantaine ou de confinement. Deux ans après, le virus évolue et les cas et décès liés a la Covid-19 continuent encore à augmenter. Toutes ces mesures ont des conséquences économiques, sociales ou environnementales. 

# Nous allons dans ce projet, nous intéresser de plus près au cas confirmés et décès liés a la Covid-19. Nous visualiserons à l’échelle mondiale puis plus précisément dans trois différents pays pour pouvoir comparer différentes gestions de cette crise sanitaire et/ou comment ces pays ont été plus ou moins touchés. Le but de ce projet sera dans un second temps d’essayer de prédire le nombre de nouveaux cas quotidiens dans ces pays à l’aide de modèle de machine learning pour les séries temporelles. 

# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>

# In[1]:


get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")


# ## Importation des jeux de données

# In[2]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


#  > Les bases de données suivantes représentent respectivement le nombre décès cumulés, le nombre de cas positifs confirmés, le nombre de décès par jour et le nombre de cas positifs confirmés par jour liés a la **Covid-19**. Ces valeurs sont prises entre le 22 janvier 2020 et le 23 juin 2021, dans chaque pays du monde. Ces bases de données ont aussi les variables *latitude* et *longitude* pour chaque pays/régions.
# >
# > On utilise un *groupby* pour regrouper en une seule valeur les pays découper en états/régions/territoires.

# In[3]:


#Nombre de décès cumulés chaque jour.
df_raw_deaths = pd.read_csv('RAW_global_deaths.csv')
df_raw_deaths = df_raw_deaths.groupby(['Country/Region'], as_index=False).sum() 
df_raw_deaths.head(5)


# In[4]:


df_raw_deaths.info()


# In[5]:


#Nombre de cas positifs confirmés cumulés chaque jour.
df_raw_cases = pd.read_csv('RAW_global_confirmed_cases.csv') 
df_raw_cases = df_raw_cases.groupby(['Country/Region'], as_index=False).sum()
df_raw_cases.head()


# In[6]:


df_raw_cases.info()


# In[7]:


#Nombre de décès par jour lié au Covid 19.
df_conv_deaths = pd.read_excel('CONVENIENT_global_deaths.xlsx') 
df_conv_deaths = df_conv_deaths.groupby(['Country/Region'], as_index=False).sum()
df_conv_deaths.head()


# In[8]:


df_conv_deaths.info()


# In[9]:


#Nombre de cas positifs confirmés par jour.
df_conv_cases = pd.read_excel('CONVENIENT_global_confirmed_cases.xlsx') 
df_conv_cases = df_conv_cases.groupby(['Country/Region'], as_index=False).sum()
df_conv_cases.head()


# In[10]:


df_conv_cases.info()


# > Cette base de données contient la population de chaque pays et régions du monde. 

# In[11]:


#Population par pays
df_population = pd.read_excel('popu_country.xls')
df_population.rename(columns={'Country Name':'Country/Region','2020':'Population'}, inplace=True)
df_population.head()


# In[12]:


df_population.info()


# ## Visualisation à l'échelle mondiale

# In[13]:


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 


# >La pandemie est dite mondiale car tous les pays du monde ont été touchés par le coronavirus. On peut voir sur ces maps le nombre de cas confirmés cumulés (jusqu’au 22 juin 2021). On constate que l’Europe à été très touché par le virus contrairement à l’Afrique par exemple (il faut aussi prendre en compte que les données du nombres de cas et de décès dans certains pays notamment en afrique sont moins surs/vérifiés que dans des pays plus développés au niveau de la santé et de l’administration). On remarque également un nombre très important de cas aux Etats Unis, dans certains pays d'Amérique du sud et en Inde.

# In[14]:


#Bubble map des cas positifs confirmés totaux liés au covid 19. (22/06/2021)

#Import du dataframe des cas cumulés.
#On utilise *melt* pour placer les dates en une seule variable et on garde seulemnet la date qui nous interesse. 
df_c = df_raw_cases.melt(id_vars =['Country/Region'],value_vars=df_raw_cases.columns.tolist()[4:522]).rename(columns={'variable':'date','value':'cases'})
df_c = df_c.loc[df_c['date'] == '6/22/21']

#Import d'un dataframe contenant le code iso-aplha_3 de chaque pays pour pouvoir les placer sur la map 
df_cont = pd.read_csv('continents2.csv')
df_cont.rename(columns={'name':'Country/Region'}, inplace=True)
df_cont.drop(['alpha-2','intermediate-region-code','country-code','iso_3166-2','region','sub-region','intermediate-region','region-code','sub-region-code'], axis=1, inplace=True)

#Création d'un dataframe contenant le nom du pays et son code alpha-3 à l'aide d'un *merge*
df_cases = df_cont.merge(df_c,on="Country/Region")
#On ajoute une ligne pour les US
df_cases = df_cases.append({'Country/Region' : 'United States', 'alpha-3' : 'USA', 'date' : '6/22/21' , 'cases' : 33551974}, ignore_index=True)

#On crée la bubble map 
fig = px.scatter_geo(df_cases, locations="alpha-3", color_discrete_sequence=['#636EFA'],
                                hover_name="Country/Region", size="cases",
                                projection="natural earth2")
fig.update_layout(
    title_text="Bubble Map : Nombre de cas totals du Covid-19 dans chaque pays du monde")
fig.update_geos(showcountries=True)
fig.show()





# > Une seconde map nous montre le nombre de décès cumulés dans chaque pays. On en déduit la forte corrélation entre le nombre de cas et le nombre de décès liés au covid 19. On note une forte mortalié en Amérique du Nord et du Sud, en Europe et Moyent Orient. L'Inde aussi à été très touchée.

# In[15]:


#Bubble map des décès totaux liés au covid 19. (22/06/2021)

#Import du dataframe des décès cumulés.
#On utilise *melt* pour placer les dates en une seule variable et on garde seulemnet la date qui nous interesse. 
df_d = df_raw_deaths.melt(id_vars =['Country/Region'],value_vars=df_raw_deaths.columns.tolist()[4:522]).rename(columns={'variable':'date','value':'deaths'})
df_d = df_d.loc[df_d['date'] == '6/22/21']

#Import d'un dataframe contenant le code iso-aplha_3 de chaque pays pour pouvoir les placer sur la map. (déjà importer dans le premier graphe)
#df_cont = pd.read_csv('continents2.csv')
#df_cont.rename(columns={'name':'Country/Region'}, inplace=True)
#df_cont.drop(['alpha-2','intermediate-region-code','country-code','iso_3166-2','region','sub-region','intermediate-region','region-code','sub-region-code'], axis=1, inplace=True)

#Création d'un dataframe contenant le nom du pays et son code alpha-3 à l'aide d'un *merge*.
df_deaths = df_cont.merge(df_d,on="Country/Region")
#On ajoute une ligne pour les US
df_deaths = df_deaths.append({'Country/Region' : 'United States', 'alpha-3' : 'USA', 'date' : '6/22/21' , 'deaths' : 602150}, ignore_index=True)

#On crée la bubble map.
fig = px.scatter_geo(df_deaths, locations="alpha-3", color_discrete_sequence=["#EF553B"],
                                hover_name="Country/Region", size="deaths",
                                projection="natural earth2")
fig.update_layout(
    title_text="Bubble Map : Nombre de décès totals du covid19 dans chaque pays du monde")
fig.update_geos(showcountries=True)
fig.show()




# >Nous rappelons qu’à cette date du 22 juin 2021, la maladie à touchés environ 179 107 034 millions de personnes (il est possible que certaines personnes l’ont eu plusieurs fois..) et à causer 3 880 876 décès. 

# In[16]:


print("Le nombre de cas confirmés totals dans le monde est de : "  , df_raw_cases['6/22/21'].sum() )  
print("Le nombre de décès confirmés totals dans le monde est de : " , df_raw_deaths['6/22/21'].sum() )  


# > Voici deux pie-charts nous montrant le top 10 des des pays ayant le plus de cas et de décès liés a la Covid-19. Les USA, l'Inde et le Brésil ont eu le plus de cas confirmés. Mais pour le nombre de décès, la classemennt change légèrement, on a toujours les même 3 pays dans le top 3, mais on note la présence du Mexique et du Pérou (4 et 5èeme) qui n'étaient pas présents dans le premier graphe. Cela nous montre que tous les pays n’ont pas le même rapport cas confirmés/décès, leur taux de mortalités est donc plus élevé. 

# In[17]:


#Import des dataframes du nombre total de cas et décès ranger dans l'ordre croissant avec *sort_values*.
#On garde avec *tail(10)* les 10 valeurs de deaths et cases les plus grandes.
df_d = df_deaths.sort_values(by='deaths').tail(10)
df_c = df_cases.sort_values(by='cases').tail(10)

#On crée les deux pie charts.
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=df_d['Country/Region'].tolist(), values=df_d['deaths'].tolist(),textinfo='label+value', name="Deaths"),
              1, 2)
fig.add_trace(go.Pie(labels=df_c['Country/Region'].tolist(), values=df_c['cases'].tolist(),textinfo='label+value' , name="Cases"),
              1, 1)

fig.update_traces(hole=.3, hoverinfo="label+percent+name")

fig.update_layout(
    title_text="Top 10 pays cas/décès les plus élevés",
    showlegend=False,
    annotations=[dict(text='Cases', x=0.18, y=0.5, font_size=20, showarrow=False),
                 dict(text='Deaths', x=0.82, y=0.5, font_size=20, showarrow=False)])
fig.show()


# ## Création de la base de données par pays

# > Ici est  crée une fonction qui permet, après avoir rentré le nom du pays souhaité en paramètre, de créer un dataframe
# contenant les variables suivantes: décès liées a la Covid 19 et cas confirmés cumulées, décès et cas confirmés par jour, décès et cas confirmés par la moyenne mobile* lissés sur 7 jours, décès et cas confirmés lissés sur 7 jours pour 1 million d'habitants. Ces valeurs seront comprises entre le 22/01/2020 et le 22/06/21. 
# >
# > On note la présence de valeurs manquantes, ou négatives dans les données de certains pays. Ceci est sûrement du au fait que les données ne sont pas récupérées tous les jours ainsi qu'à des réajustements des données. On les remplacera par 0, et c'est pour cela qu'on crée des lissages par la moyenne mobile pour certaines variables. 
# >
# > *La moyenne mobile est un type de moyenne statistique utilisée pour analyser des séries ordonnées de données, le plus souvent des séries temporelles. Celle ci est calculée tour à tour sur chaque sous-ensemble de N valeurs consécutives (ici nous avons choisi 7 valeurs consécutives). 

# In[18]:


def dataframe_covid(pays):
    df_rd = df_raw_deaths[df_raw_deaths['Country/Region'] == pays] #on crée un dataframe contenant le nombre de décès cumulés dans le pays choisi.
    df_rd = df_rd.iloc[:,3:521] #on enlève les variables latitude et longitude. 
    df_rd = df_rd.melt( value_vars=df_rd.columns.tolist()) #on place les date dans une seule varable
    df_rd = df_rd.rename(columns={"variable":"date","value":"deaths"}) #on renome les nouvelles variables
    df_rd.drop(0,0,inplace=True) #on enlève la premiere ligne qui n'a aucune information.
    
    df_rc = df_raw_cases[df_raw_cases['Country/Region'] == pays]
    df_rc = df_rc.iloc[:,3:521]
    df_rc = df_rc.melt( value_vars=df_rc.columns.tolist())
    df_rc.drop(0,0,inplace=True)
    df_rc = df_rc.rename(columns={"variable":"date","value":"cases"})
    df_rc.drop(['date'], axis=1,inplace=True) #on enlève la viable date qui est déjà présente dans df_rd
    
    df_cc = df_conv_cases[df_conv_cases['Country/Region'] == pays]
    df_cc = df_cc.melt(value_vars=df_cc.columns.tolist())
    df_cc.drop(0,0,inplace=True)
    df_cc = df_cc.rename(columns={"variable":"date","value":"cases_per_day"})
    df_cc.drop(['date'], axis=1,inplace=True)

    df_cd = df_conv_deaths[df_conv_deaths['Country/Region'] == pays]
    df_cd = df_cd.melt(value_vars=df_cd.columns.tolist())
    df_cd.drop(0,0,inplace=True)
    df_cd = df_cd.rename(columns={"variable":"date","value":"deaths_per_day"})
    df_cd.drop(['date'], axis=1,inplace=True)

    df_covid = pd.concat([df_rd,df_rc,df_cd,df_cc], axis=1) #on regroupe en un seul dataframe les 4 crées précedemment à l'aide de *concat*.

    df_covid.set_index('date', inplace = True) #on met la variable date en indice.
    df_covid.index = pd.to_datetime(df_covid.index) #on met l'indice en format datetime.

    df_covid.loc[(df_covid['cases_per_day']<0), 'cases_per_day'] = np.nan #certaines valeurs sont négatives (problèmes de données), on les transformes en NaN.
    df_covid.loc[(df_covid['deaths_per_day']<0), 'deaths_per_day'] = np.nan
    df_covid = df_covid.fillna(0) #on remplace toutes les valeurs manquantes par 0.
    
    #création des variables par la moyenne mobile sur 7 jours avec *rolling*, puis en pourcentages de population et pour 1 million d'habitants.
    df_covid['ma_cases']=df_covid['cases_per_day'].rolling(window = 7, center = True).mean().round()
    df_covid['ma_deaths']=df_covid['deaths_per_day'].rolling(window = 7, center = True).mean().round()
    df_covid['ma_casesM']=((df_covid['ma_cases'])*1000000/df_population[df_population['Country/Region'] == pays].values.tolist()[0][1]).round()
    df_covid['ma_deathsM']=((df_covid['ma_deaths'])*1000000/df_population[df_population['Country/Region'] == pays].values.tolist()[0][1])
    
    return df_covid 


# Nous avons fait le choix de prendre 3 pays, le Brésil, le Japon et la France pour les étudier plus spécifiquement. Nous allons donc créer leurs dataframes. Ces pays ont été choisis pour la difference notable de gestion de la crise sanitaire, comme nous le verrons plus tard sur les graphiques. 
# 

# ### Le Brésil

# > Le brésil fait parti des 3 pays les plus durement touchés par le Covid 19 que ce soit en nombre de cas ou de décès. On estime même que les données sont en deça de la réalité selon plusieurs organisations. La gestion de la crise du coronavirus 2019 par le gouvernement brésilien a été jugé très mauvaise par certaines autorités exterieures au pays, peu de mesures fortes ont étées prises ou alors très en retard par rapport à beaucoup de pays, comme les fermetures partielles des frontières, beaucoup de régions pauvres et de favelas ont été desertées de médecins et d'infrastructures de santé. Le Brésil ayant seulement le 63ème système de soins au monde, selon l'édition 2021 du magazine CEOWORLD Health Care Index, qui classe 89 pays selon les facteurs qui contribuent à la santé globale.
# 

# In[19]:


#On appelle la fonction pour crée notre dataframe. (ici pour le Brésil)
df_covid_bra = dataframe_covid('Brazil') 
df_covid_bra.head()


# In[20]:


df_covid_bra.info() #On retourve bien l'indice en datetime, et les valeurs sont en int et float pour les moyennes.


# In[21]:


# Courbe des cas confirmés de Covid19 au Bresil.
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_covid_bra.index, y=df_covid_bra['cases_per_day'],
                         mode='lines',
                         name='Cas/jour Bresil'))
fig.add_trace(go.Scatter(x=df_covid_bra.index, y=df_covid_bra['ma_cases'],
                         mode='lines',
                         name='Cas/jour Brésil (moyenne 7 jours)'))

fig.add_trace(go.Scatter(x=df_covid_bra.index, y=df_covid_bra['deaths_per_day'],
                         mode='lines',
                         name='Décès/jour Bresil'))
fig.add_trace(go.Scatter(x=df_covid_bra.index, y=df_covid_bra['ma_deaths'],
                         mode='lines',
                         name='Décès/jour Brésil (moyenne 7 jours)'))

fig.update_layout(
    title_text="Cas et décès confirmés de Covid19 au Bresil.")

fig.show()


# In[22]:


fig = go.Figure()
fig.add_trace(go.Bar(x=df_covid_bra.index, y=df_covid_bra['cases'],marker_color='blue', name= 'cas'))
fig.add_trace(go.Bar(x=df_covid_bra.index, y=df_covid_bra['deaths'],marker_color='red', name= 'décès'))
fig.add_trace(go.Scatter(x=df_covid_bra.index, y=df_covid_bra['cases'],marker_color='blue',mode='lines',name= 'cas'))
fig.add_trace(go.Scatter(x=df_covid_bra.index, y=df_covid_bra['deaths'],marker_color='red', mode='lines',name= 'décès'))


fig.update_layout(
    title_text="Courbe du nombre de cas et décès cumulés au Brésil")
fig.update_layout(barmode='overlay')
fig.show()


# ### Le Japon

# > Le Japon, contrairement au Brésil, a gérer la crise sanitaire d'une facon beaucoup plus rigide. D'un point de vue propre au pays, il pratique très peu de tests, le nombre de lits de réanimation n’a pas été drastiquement augmenté, et aucun confinement sévère n’a été mis en place. Mais celui-ci a un système de santé et d’assurance sociale très performant (5ème) et a eu des politiques sur la fermeture des frontières très fortes. En effet il est très dur de rentrer dans le pays et une quarantaine obligatoire a été mis en place en arrivant à certaines périodes. Le Japon bénéficie également de bonnes conditions sans doute liées aux modes de vie qui semblent jouer un rôle protecteur : les contacts physiques en société et même en famille sont très limités, le niveau d’hygiène est traditionnellement très élevé (habitude de se laver les mains par exemple). Le masque est porté très régulièrement au printemps et en hiver pour se protéger des allergies, de pollution, éviter de se contaminer en cas d’infection ou d’être contaminé.

# In[23]:


df_covid_jap = dataframe_covid('Japan') 
df_covid_jap.head()


# In[24]:


df_covid_jap.info()


# In[25]:


# Courbe des cas confirmés de Covid19 au Japon.
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_covid_jap.index, y=df_covid_jap['cases_per_day'],
                         mode='lines',
                         name='Cas/jour Japon'))
fig.add_trace(go.Scatter(x=df_covid_jap.index, y=df_covid_jap['ma_cases'],
                         mode='lines',
                         name='Cas/jour Japon (moyenne 7 jours)'))

fig.add_trace(go.Scatter(x=df_covid_jap.index, y=df_covid_jap['deaths_per_day'],
                         mode='lines',
                         name='Décès/jour Japon'))
fig.add_trace(go.Scatter(x=df_covid_jap.index, y=df_covid_jap['ma_deaths'],
                         mode='lines',
                         name='Décès/jour Japon (moyenne 7 jours)'))

fig.update_layout(
    title_text="Cas et décès confirmés de Covid19 en Japon.")

fig.show()


# In[26]:


fig = go.Figure()
fig.add_trace(go.Bar(x=df_covid_jap.index, y=df_covid_jap['cases'],marker_color='blue', name= 'cas'))
fig.add_trace(go.Bar(x=df_covid_jap.index, y=df_covid_jap['deaths'],marker_color='red', name= 'décès'))

fig.update_layout(
    title_text="Nombre de cas et décès cumulés en Japon")
fig.update_layout(barmode='overlay')
fig.show()


# ### La France

# > Le dernier pays chosie est la France, ça gestion de la crise et le nombre de cas et décès quotidien ressemble plus ou moins aux autres pays d'Europe de l'Ouest comme l'Italie, l'Allemagne ou le Royaume Uni. Le systeme de santé est performant (7ème) et les mesures internes sont présentes, comme le port du masque obligatoire, la mise en place de confiments, la fermeture des lieux de rencontres (stades, cinemas, boites de nuits etc..). Mais comme la plupart de ces pays voisins, la France a une grande difficulté à gérer la fermeture de ces frontiéres. La situation gégraphique et politique de l'Europe et de l'Union européene fait qu'il est diffcile de se refermer complétement comme le Japon par exemple. En géneral, la France fait partie des pays les plus touchés d'Europe et même du monde. 

# In[27]:


df_covid_fr = dataframe_covid('France') 
df_covid_fr.head()


# In[28]:


df_covid_fr.info()


# In[29]:


# Courbe des cas confirmés de Covid19 en France.
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_covid_fr.index, y=df_covid_fr['cases_per_day'],
                         mode='lines',
                         name='Cas/jour France'))
fig.add_trace(go.Scatter(x=df_covid_fr.index, y=df_covid_fr['ma_cases'],
                         mode='lines',
                         name='Cas/jour France (moyenne 7 jours)'))

fig.add_trace(go.Scatter(x=df_covid_fr.index, y=df_covid_fr['deaths_per_day'],
                         mode='lines',
                         name='Décès/jour France'))
fig.add_trace(go.Scatter(x=df_covid_fr.index, y=df_covid_fr['ma_deaths'],
                         mode='lines',
                         name='Décès/jour France (moyenne 7 jours)'))

fig.update_layout(
    title_text="Cas et décès confirmés de Covid19 au France.")

fig.show()


# In[30]:


fig = go.Figure()
fig.add_trace(go.Bar(x=df_covid_fr.index, y=df_covid_fr['cases'],marker_color='blue', name= 'cas'))
fig.add_trace(go.Bar(x=df_covid_fr.index, y=df_covid_fr['deaths'],marker_color='red', name= 'décès'))

fig.update_layout(
    title_text="Nombre de cas et décès cumulés en France")
fig.update_layout(barmode='overlay')
fig.show()


# ## Visualisation des cas Brésil, Japon et France

# Nous allons à présent essayer de comparer visuellement ces trois pays que sont le Brésil, le Japon et la France. 
# 
# > Au niveau des cas et décès cumulés, on voit clairement la difference entre ces pays, le Brésil ayant eu plus de 18 millions de cas et 500 000 décès alors que le Japon n'a pas dépassé a ce stade la barre du million de cas positifs. 
# La France quant à elle, a eu près de 6 millions de cas pour plus de 100 000 décès.

# In[31]:


#Nombre de cas total pour chaque pays. (22/06/2021)
print("Le nombre de cas confirmés totals au Brésil est de : "  , df_covid_bra['cases']['6/22/21'] )  
print("Le nombre de cas confirmés totals en Japon est de : "  , df_covid_jap['cases']['6/22/21'] ) 
print("Le nombre de cas confirmés totals en France est de : "  , df_covid_fr['cases']['6/22/21'] ) 


# In[32]:


#Nombre de décès total pour chaque pays. (22/06/2021)
print("Le nombre de décès totals au Brésil est de : "  , df_covid_bra['deaths']['6/22/21'] )  
print("Le nombre de décès totals en Japon est de : "  , df_covid_jap['deaths']['6/22/21'] ) 
print("Le nombre de décès totals en France est de : "  , df_covid_fr['deaths']['6/22/21'] ) 


# > Le taux de mortalité liée au covid (nombre de décès sur le nombre de cas positifs) est très elevé au Brésil, 2,8%, contrairement au Japon et à la France qui ne dépasse pas les 2%. Le taux de mortalité entre le Japon et la France est presque le même, pourtant on à vu précédemment que la France a plus de 7 fois le nombre de cas du Japon. Le système de santé et la gestion de la crise sanitaire peut être mis en cause pour le fort taux de mortalité du Brésil.

# In[33]:


#Taux de mortalité du Covid pour chaque pays.
print("Le taux de mortalité liés au Covid19 au Brésil est de : "  , ((df_covid_bra['deaths']['6/22/21']/df_covid_bra['cases']['6/22/21'])*100).round(2) , "%" )  
print("Le taux de mortalité liés au Covid19 en Japon est de : "  , ((df_covid_jap['deaths']['6/22/21']/df_covid_jap['cases']['6/22/21'])*100).round(2) , "%" )  
print("Le taux de mortalité liés au Covid19 en France est de : "  , ((df_covid_fr['deaths']['6/22/21']/df_covid_fr['cases']['6/22/21'])*100).round(2) , "%" )  


# > Sur les 2 courbes suivantes on peut voir l'évolution des cas confirmés et décès par jour au Brésil, au Japon et en France. Les pics de cas positifs sont differents selon les pays, on peut voir que la France a eu des périodes très dur notament en novembre 2020. Le Brésil n'a jamis eu de période ou le nombre de cas quotidiens était de 0. Les pics de cas suivent ceux des décès quotidiens notament pour le Japon et surtout en début de pandemie pour la France qui a peut etre su s'adapter à la situation. On note 2 vagues peu distinctes pour le Brésil, 3 pour la France (mars-avril 2020, octobre novembre 2020, mars mais 2021) et 4 pour le Japon (mars - avril 2020, juillet -septembre 2020, novembre -fevrier 2021 et mars-juin 2021). On constate que les périodes de confinement en France ont entrainé la baisse du nombre de cas. 

# In[34]:


# Courbe de la moyenne des cas confirmés de Covid19 au Bresil, Japon et France
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_covid_bra.index, y=df_covid_bra['ma_cases'],
                         mode='lines',
                         name='Cas/jour Bresil (moyenne 7 jours)'))
fig.add_trace(go.Scatter(x=df_covid_jap.index, y=df_covid_jap['ma_cases'],
                         mode='lines',
                         name='Cas/jour Japon (moyenne 7 jours)'))
fig.add_trace(go.Scatter(x=df_covid_fr.index, y=df_covid_fr['ma_cases'],
                         mode='lines',
                         name='Cas/jour France (moyenne 7 jours)'))


fig.update_layout(
    title_text="Courbe de la moyenne sur 7 jours des cas confirmés de Covid-19 au Bresil, Japon et France")

fig.show()


# In[35]:


# Courbe de la moyenne des décès confirmés de Covid19 au Bresil, Japon et France
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_covid_bra.index, y=df_covid_bra['ma_deaths'],
                         mode='lines',
                         name='Décès/jour Bresil (moyenne 7 jours)'))
fig.add_trace(go.Scatter(x=df_covid_jap.index, y=df_covid_jap['ma_deaths'],
                         mode='lines',
                         name='Décès/jour Japon (moyenne 7 jours)'))
fig.add_trace(go.Scatter(x=df_covid_fr.index, y=df_covid_fr['ma_deaths'],
                         mode='lines',
                         name='Décès/jour France (moyenne 7 jours)'))


fig.update_layout(
    title_text="Courbe de la moyenne sur 7 jours des décès confirmés de Covid-19 au Bresil, Japon et France")

fig.show()


# > On a ici les même graphiques mais le nombre de cas et de dècès est rapporté pour 1 million d'habitants. On voit ainsi que la France et très durement touchés par le Covid 19, plus que le Brésil si l'on regarde sur la même échelle. (Il ne faut pas oublié qu'il est possible que les données brésilienne soit plus fortes que celles que nous avons ici). La courbe du Japon reste très faible par rapport au deux autres pays. La France et le Japon on des vagues de cas très notables ce qui n'est pas le cas du Brésil.

# In[36]:


# Courbe de la moyenne des cas confirmés de Covid-19 pour 1 million d'habitans par jour au Bresil, Japon et France
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_covid_bra.index, y=df_covid_bra['ma_casesM'],
                         mode='lines',
                         name='Cas/jour pour 1M hab Bresil'))
fig.add_trace(go.Scatter(x=df_covid_jap.index, y=df_covid_jap['ma_casesM'],
                         mode='lines',
                         name='Cas/jour pour 1M hab Japon'))
fig.add_trace(go.Scatter(x=df_covid_fr.index, y=df_covid_fr['ma_casesM'],
                         mode='lines',
                         name='Cas/jour pour 1M hab France'))
#fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(
    title_text="Courbe de la moyenne des cas confirmés de Covid-19 pour 1 million d'habitans par jour au Bresil, Japon et France")
fig.show()


# In[37]:


# Courbe de la moyenne des cas confirmés de Covid-19 pour 1 million d'habitans par jour au Bresil, Japon et France
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_covid_bra.index, y=df_covid_bra['ma_deathsM'],
                         mode='lines',
                         name='Décès/jour pour 1M hab Bresil'))
fig.add_trace(go.Scatter(x=df_covid_jap.index, y=df_covid_jap['ma_deathsM'],
                         mode='lines',
                         name='Décès/jour pour 1M hab Japon'))
fig.add_trace(go.Scatter(x=df_covid_fr.index, y=df_covid_fr['ma_deathsM'],
                         mode='lines',
                         name='Décès/jour pour 1M hab France'))

fig.update_layout(
    title_text="Courbe de la moyenne des décès liés au Covid-19 pour 1 million d'habitans par jour au Bresil, Japon et France")
fig.show()


# > Au niveaux des cas et décès totaux, on voit clairement sur ces courbes que le Bresil à été le plus touché des 3 pays. Le nombre de cas cumulés augmente assez tard pour la France et le Japon, mais on retourve que le nombre de décès augmente plus vite en début de pandemie pour la France, plus que le Brésil jusqu'a debut juin 2020.

# In[38]:


#Courbe des cas confirmés cumulées de Covid19 au Bresil et Japon et France
fig = go.Figure()


fig.add_trace(go.Bar(x=df_covid_bra.index, y=df_covid_bra['cases'],
                         marker_color='#00CC96',
                         name='Cas cumulés Bresil'))
fig.add_trace(go.Bar(x=df_covid_fr.index, y=df_covid_fr['cases'],
                         marker_color='#636EFA',
                         name='Cas cumulés France'))

fig.add_trace(go.Bar(x=df_covid_jap.index, y=df_covid_jap['cases'],
                         marker_color='#EF553B',
                         name='Cas cumulés Japon'))



fig.add_trace(go.Scatter(x=df_covid_bra.index, y=df_covid_bra['cases'],
                         mode='lines',
                         marker_color='#00CC96',
                         name='Cas Bresil'))
fig.add_trace(go.Scatter(x=df_covid_jap.index, y=df_covid_jap['cases'],
                         mode='lines',
                         marker_color='#EF553B',
                         name='Cas Japon'))
fig.add_trace(go.Scatter(x=df_covid_fr.index, y=df_covid_fr['cases'],
                         mode='lines',
                         marker_color='#636EFA',
                         name='Cas France'))

fig.update_layout(barmode='overlay')
fig.update_layout(
    title_text="Cas confirmés cumulées de Covid19 au Bresil et Australie et France")

fig.show()


# In[39]:


# Courbe des décès cumulées de Covid19 au Bresil et Japon et France.
fig = go.Figure()

fig.add_trace(go.Bar(x=df_covid_bra.index, y=df_covid_bra['deaths'],
                         marker_color='#00CC96',
                         name='Décès cumulés Bresil'))
fig.add_trace(go.Bar(x=df_covid_fr.index, y=df_covid_fr['deaths'],
                         marker_color='#636EFA',
                         name='Décès cumulés France'))
fig.add_trace(go.Bar(x=df_covid_jap.index, y=df_covid_jap['deaths'],
                         marker_color='#EF553B',
                         name='Décès cumulés Japon'))

fig.add_trace(go.Scatter(x=df_covid_bra.index, y=df_covid_bra['deaths'],
                         mode='lines',
                         marker_color='#00CC96',
                         name='Décès Bresil'))
fig.add_trace(go.Scatter(x=df_covid_jap.index, y=df_covid_jap['deaths'],
                         mode='lines',
                         marker_color='#EF553B',
                         name='Décès Japon'))
fig.add_trace(go.Scatter(x=df_covid_fr.index, y=df_covid_fr['deaths'],
                         mode='lines',
                         marker_color='#636EFA',
                         name='Décès France'))

fig.update_layout(barmode='overlay')
fig.update_layout(
    title_text="Courbe des cas confirmés cumulées de Covid19 au Bresil et Japon et France")

fig.show()


# ## Modèle de prédiction

# Dans la suite de ce projet, nous allons faire des prédictions du nombre de cas quotidiens dans chacun des trois pays analisés précedement, le Brésil, le Japon et la France. Nous allons utiliser pour cela des modèles de machine learning adaptés aux séries temporelles. Nous verrons des prédictions avec le modèle SARIMA, Prophet et Holt-Winters. Nous appliquerons nos modèles sur les 3 dernières semaines et feront des prédictions à court termes sur les 7 jours suivants. On n'essayera pas de prévoir plus que les 7 prochains jours car beaucoup trop de facteurs (évolutions du virus, prisent de nouvelles mesures, vaccins etc..) vont jouer sur une évolution à plus long termes de la Covid-19. Nous les comparerons à l'aide de differente métriques. Dans un premier temps nous allons importer toute les bibliothèques et certaines fonctions nécessaires pour nos modèles. 

# In[40]:


#Import 

#SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import pmdarima as pm #à installer à l'aide de *pip*
import statsmodels.api as sm

#PROPHET
from prophet import Prophet #à installer à l'aide de *pip* (compatble avec python <= 3.9)

#HOLT-WINTER
from statsmodels.tsa.api import ExponentialSmoothing

#METRIQUE
from tabulate import tabulate
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

def MAPE(y_true, y_prediction): #Fonction pour calculer l'erreur absolue moyenne en pourcentage.
    y_true, y_prediction = np.array(y_true), np.array(y_prediction)
    return np.mean(np.abs((y_true - y_prediction) / y_true)) * 100


# In[41]:


#La fonction df_model() permet de creer un dataframe contenant seulement la variable date et le nombre de cas par jour.
def df_model(df):
    df = pd.DataFrame(df['cases_per_day']['2020-01-22':'2021-06-15']) #.rolling(window = 7, center = True).mean().round())
    df = df.reset_index()
    df = df.rename(columns={"date":"ds","cases_per_day":"y"})
    return df


# In[42]:


#Fenetres de prévision et de prédiction qu'on utilisera pour les modèles

prediction_w = 21 #prises en comptes des 3 dernieres semaines 
forecast_w = 7 #prévision sur les 7 jours suivant
window = prediction_w + forecast_w #fenetre total 


# #### Explication des modèles 

# > SARIMA: est une extension du modèle ARIMA (Seasonal ARIMA). Le modèle ARIMA (Autegressive Integrated Moving Average) est désigné par 3 paramètres: 
# - p : ordre de la partie autorégressive (AR), il est déterminé par la Fonction d’Autocorrélation ( ACF).
# - d : ordre de la différence ou dérivation du processus. Cela permet de rendre stationnaire la série au cas où elle ne l’était pas. Ainsi, d=0 si la série est au préalable stationnaire sinon d>=1, on le trouve grace au test Augmented Dickey Fuller.
# - q : ordre de la moyenne mobile (MA), ce paramètre est déterminé par la  Fonction d’Autocorrélation Partielle (PACF).
# >Le modèle SARIMA permet modéliser les séries temporelles comportant une composante saisonnière en plus et contient 4 autres paramétres.

# >PROPHET: est une méthode de prédiction développée par Facebook dans le but de démocratiser les prévisions des séries temporelles et de les simplifier. Ce modèle convient particulièrement à des séries temporelles  affectées par des événements ou des saisonnalités liées à l’activité humaine (exemple : fêtes de fin d’année, soldes, saisons, vacances, etc.). Le modèle de Facebook Prophet est un modèle additionnant 3 éléments : la tendance, la saisonnalité et l’effet des vacances / événements, plus du bruit : Y(t) = g(t) + s(t) + h(t) + e(t). Nous ne prendrons pas en compte la variable h(t), pour simplifier le modèle et nous supposerons que cela n'est pas un facteur qui a un role important dans le nombre de cas quotidien de la Covid 19.

# > HOLT-WINTERS: Le lissage exponentiel est une méthode empirique de lissage et de prévision de données chronologiques. Comme dans la méthode des moyennes mobiles, chaque donnée est lissée successivement en partant de la valeur initiale. Le lissage exponentiel donne aux observations passées un poids décroissant exponentiellement avec leur ancienneté. 
# Ce modèle est une extension du lissage exponentiel qui capture la saisonnalité. Cette méthode génère des valeurs lissées de façon exponentielle pour le niveau, la tendance et l'ajustement saisonnier de la prévision.
# 

# #### Explication des mesures de qualité
# 

# > Nous allons après chaque prédictons de modèle effectuer des mesures de qualité du modèle à l'aide de certaines métriques. Nous avons:
# >
# > MAPE (Mean Absolute Percentage Error) : c'est l'erreur absolue moyenne en pourcentage. On l'a calcul avec la formule suivante: np.mean(np.abs(forecast - actual)/np.abs(actual))
# >
# > MAE (Root Mean Squared Error) : c'est l'erreur absolue moyenne. On l'a calcul avec la commande mean_absolute_error().
# >
# > R2 : c'est le coefficient de determination. On l'a calcul avec la commande r2_score().
# >
# > RMSE (Root Mean Squared Error) : c'est l'erreur quadratique moyenne. On l'a calcul avec la commande mean_squared_error().

# ### Prédiction du nombre de cas quotidien au Brésil
# 

# > Nous allons tout d'abord crée notre dataframe contenant la date et le nombre de cas par jour au Brésil.

# In[43]:


df_bresil = df_model(df_covid_bra).dropna()
df_bresil.tail()


# In[44]:


df_bresil_dt = df_bresil.set_index('ds')
df_bresil_dt.tail()


# #### Prédiction avec SARIMA (Brésil)

# > Tout d'abord nos allons vérifer si la série est stationnaire à l'aide du test Augmented Dickey Fuller.
# L'hypothèse nulle du test ADF est que la série chronologique n'est pas stationnaire. Ainsi, si la p-value du test est inférieure à 5%, on rejete l'hypothèse nulle et on en déduit que la série chronologique est effectivement stationnaire. 
# Si aucune différenciation n'est nécessaire la valeur du d est 0.

# > On a notre p-value du test ADF supérieur à 0,05. On estime donc que la série n'est pas stationnaire. On recommence le test avec une série différenciée d'ordre 1.

# In[45]:


result=adfuller(df_bresil_dt['y'].dropna())
print(f'ADF Statistics:{result[0]}')
print(f'p-value:{result[1]}')


# > Après avoir différencier la série, la p-value passe en dessous des 5%. On estime alors la valeur de d à 1.
# 

# In[46]:


result=adfuller(df_bresil_dt['y'].diff().dropna())
print(f'ADF Statistics:{result[0].round(4)}')
print(f'p-value:{result[1].round(4)}')


# > On regarde ainsi le graphe d'autocorrélation et d'autocorellation partielle pour esssayer de trouver les valeurs de p et q. On remarque dans un premier temps une forme de saisonnalité de 7 sur le 1er graphique. Les deux graphes ne sont pas très précis et on retoruve des *lags* non significativement differents de 0 qu'a partir du 7eme. Il est donc assez difficile de donner les valeurs de p et q.

# In[47]:


fig, (ax1, ax2)=plt.subplots(2,1,figsize=(8,8))

plot_acf(df_bresil_dt['y'].diff().dropna(),lags=30, zero=False, ax=ax1)
plot_pacf(df_bresil_dt['y'].diff().dropna(),lags=30, zero=False, ax=ax2)
plt.show()


# > Heuresement pour nous, il existe une commande auto_arima de la bibliotheque *pmdarima* qui cherche les meilleurs valeurs des paramètres SARIMA en comparant la mesure de qualité AIC (Akaike information criterion): il propose une estimation de la perte d'information lorsqu'on utilise le modèle considéré pour représenter le processus qui génère les données. On choisit alors le modèle avec le critère d'information d'Akaike le plus faible.

# In[48]:


pm.auto_arima(df_bresil_dt['y'], start_p=0, d=None, start_q=0, max_p=3, max_q=3,
                      seasonal=True, m=7, D=None, test='adf', start_P=0, start_Q=0, max_P=3, max_Q=3,
                      information_criterion='aic', trace=True, error_action='ignore',
                      trend=None,with_intercept=False, stepwise=True)


# In[ ]:





# > On va donc entrainé le model SARIMA avec les parametres (0,1,1)(1,0,1)[7]

# In[49]:


model_s = SARIMAX(df_bresil_dt['y'], order=(0,1,1), seasonal_order=(1,0,1,7))
model_s_fit = model_s.fit()


# > On a ici un résumé de notre modèle ainsi que les graphe des résidus et de leur densité. On retrouve une une densité des résidus qui forment une loi Normale et un graphe quantile-quantile correct.

# In[50]:


model_s_fit.summary()


# In[51]:


model_s_fit.plot_diagnostics(figsize=(8,8))
plt.show()


# > Notre objectif est maintenant de créer une nouvelle base de données avec les prédictions de test du modèle sur les 3 dernières semaines de données et les valeurs de prévision sur les 7 prochains jours.
# Nous commençons par créer le jeu de données de la prédiction.

# In[52]:


prediction = model_s_fit.get_prediction(start=-prediction_w)
mean_prediction = prediction.predicted_mean

sarimax_prediction = pd.DataFrame({'yhat':mean_prediction})


# > Création de la base de donnée de la prédiction.

# In[53]:


forecast = model_s_fit.get_forecast(steps=forecast_w)
mean_forecast=forecast.predicted_mean

sarimax_forecast = pd.DataFrame({'yhat':mean_forecast})
sarimax_results = sarimax_prediction.append(sarimax_forecast)


# > Visualisation de la prévision du nombre de cas par jour avec le modèle SARIMA au Brésil.

# In[54]:


fig = go.Figure()

# Courbe du nombre de cas quotidiens sur les 3 dernieres semaines (jeu d'entrainemnt) 
fig.add_trace(go.Scatter(x=df_bresil_dt[-prediction_w:].index, y=df_bresil_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour Brésil'))

# Courbe de la prédiction SARIMA.
fig.add_trace(go.Scatter(x=sarimax_results.index, y=sarimax_results.yhat,
                         mode='lines+markers',
                         name='Cas/jour Brésil SARIMAX'))

# Courbe du nombre de cas quotidien lors des 7 jours prédit au Brésil
fig.add_trace(go.Scatter(x=df_covid_bra['cases_per_day']['2021-06-15':'2021-06-23'].index,
                         y=df_covid_bra['cases_per_day']['2021-06-15':'2021-06-23'],
                         mode='lines+markers',
                         name='Cas/jour Brésil '))

fig.update_layout(
    title_text="Nouveaux cas en Brésil avec SARIMAX.")

fig.show()


# > Nous allons pour finir avec le modèle SARIMA, calculer des mesures de qualités/précisions du modèle pour juger nos prévisions. 
# >
# > La métrique MAPE est à 19,13 % ce implique que le modèle est précis à environ 80,87 % pour prédire les 7 prochaines observations. Le coefficient de detremination est de 0,53, notre modèle n'est pas très bon mais reste correct. Les erreurs absolue et quadratique moyenne sont entre 10 900 et 14 400, on pourra par la suite les comparer avec les 2 autres modèles. 

# In[55]:


mape_sarimax= MAPE(df_bresil_dt[-prediction_w:]['y'], mean_prediction.values)
r2_sarimax= r2_score(df_bresil_dt[-prediction_w:]['y'],mean_prediction.values)
rmse_sarimax = mean_squared_error(df_bresil_dt[-prediction_w:]['y'], mean_prediction.values, squared=False)
mae_sarimax = mean_absolute_error(df_bresil_dt[-prediction_w:]['y'],mean_prediction.values)

print('SARIMAX MAPE: ', mape_sarimax.round(2))
print('SARIMAX MAE: ', mae_sarimax.round(2))
print('SARIMAX R2: ', r2_sarimax.round(2))
print('SARIMAX RMSE: ', rmse_sarimax.round(2))


# In[ ]:





# #### Prédiction avec Prophet (Brésil)

# > À présent nous allons faire la même prédiction mais avec un modèle different, Prophet.  On instancie dans un premier temps le modèle, on ajoute une saisonalité d'une période de 7 jours puis on entraine le model. Il y a plusieurs hyperparametres à changer lorsqu'on instancie le modèle, n_changepoints correspond à la précision des prédiction (par défaut 25), le modèle est un modèle additif et nous rajoutons une saisonnalité hebdomadaire et annuel.

# In[56]:


model_p = Prophet(n_changepoints=50,
                 seasonality_mode='additive',
                 changepoint_prior_scale=1)


# In[57]:


model_p.add_seasonality('weekly', period = 7, fourier_order = 8)
model_p.add_seasonality('yearly', period = 365, fourier_order = 19)


# > Nous entrainons notre modèle

# In[58]:


model_p_fit = model_p.fit(df_bresil)


# > Pour effectuer la prédiction, nous devons d'abord créer une base de données des prédictions des 7 prochians jours puis nous effectuons celles ci. 

# In[59]:


future = model_p.make_future_dataframe(periods=forecast_w)


# In[60]:


forecast = model_p.predict(future)


# > On affiche ainsi les prédictions du modèle puis la tendance et saisonalité de celles ci. On note une tendance à la hausse les dernieres jours et des données plus basses le dimanche et lundi. Cela est dû à la reception des données.

# In[61]:


model_p.plot(forecast);
plt.title("Nouveaux cas covid19 en Bresil et prévision by Prophet")
plt.ylabel("Cas/jour")
plt.show()


# In[62]:


fig = model_p.plot_components(forecast)


# > On crée notre dataframe avec les valeurs prédites (yhat) puis nous affichons les courbes du nombre de cas par jour 
# au Brésil ces dernieres semaines et la prédiction à l'aide du modèle Prophet. On voit que le modèle attenue légeremnt les hausses et baisses de cas.

# In[63]:


f_df = forecast[['ds', 'yhat']]
f_df.tail()


# In[64]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=df_bresil_dt[-prediction_w:].index, y=df_bresil_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour Brésil'))

fig.add_trace(go.Scatter(x=f_df[-window:]['ds'], y=f_df[-window:]['yhat'],
                         mode='lines+markers',
                         name='Cas/jour Brésil Prophet'))


fig.add_trace(go.Scatter(x=df_covid_bra['cases_per_day']['2021-06-15':'2021-06-23'].index,
                         y=df_covid_bra['cases_per_day']['2021-06-15':'2021-06-23'],
                         mode='lines+markers',
                         name='Cas/jour Brésil '))

fig.update_layout(
    title_text="Nouveaux cas en Brésil avec Prophet.")

fig.show()


# > Lorsqu'on regarde les mesures de qualité du modèle, on voit qu'elles sont proches que celle du modèle SARIMA, on a presque le meme R2 (0,52), un coefficinet MAPE legerement plus mauvais, ce qui induit un MAE plus elevé. 

# In[65]:


mape_p= MAPE(df_covid_bra['2021-05-26':'2021-06-22']['cases_per_day'],f_df[-window:]['yhat'].values)
r2_p= r2_score(df_covid_bra['2021-05-26':'2021-06-22']['cases_per_day'],f_df[-window:]['yhat'].values)
rmse_p = mean_squared_error(df_covid_bra['2021-05-26':'2021-06-22']['cases_per_day'],f_df[-window:]['yhat'].values, squared=False)
mae_p = mean_absolute_error(df_covid_bra['2021-05-26':'2021-06-22']['cases_per_day'],f_df[-window:]['yhat'].values)

print('PROPHET RMSE: ', rmse_p.round(2))
print('PROPHET MAPE: ', mape_p.round(2))
print('PROPHET R2: ', r2_p.round(2))
print('PROPHET MAE: ', mae_p.round(2))


# In[ ]:





# #### Prédiction avec Holt Winters (Brésil)

# > Pour finir, nous allons faire une derniere prédiction avec la méthode de lissage exponentielle Holt-Winters. On instancie les parametres de la fonction en ajoutant une saisonalité de 7 jours sur un modèle additif puis nous entrainons le modèle. On affiche un résumé de l'entrainement du model Holt Winters.

# In[66]:


model_hw = ExponentialSmoothing(df_bresil_dt['y'][-prediction_w:], seasonal_periods=7, trend='add', seasonal='add')
model_hw_fit = model_hw.fit()


# In[67]:


print(model_hw_fit.summary())


# > On effectue les predictions du modèle HW, puis on crée le dataframe correspondant. On trace ensuite le graphe du nombre de cas quotidien au Bresil et ces prédictions avec le modèle de lissage exponentielles.

# In[68]:


model_hw_pred = model_hw_fit.forecast((7))


# In[69]:



model_hw_pred = pd.DataFrame(model_hw_pred)
model_hw_pred.rename(columns={0:'yhat'}, inplace=True)


# In[70]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=df_bresil_dt[-prediction_w:].index, y=df_bresil_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour Brésil'))

fig.add_trace(go.Scatter(x=model_hw_pred.index, y=model_hw_pred.yhat,
                         mode='lines+markers',
                         name='Cas/jour Brésil HW'))


fig.add_trace(go.Scatter(x=df_covid_bra['cases_per_day']['2021-06-16':'2021-06-23'].index,
                         y=df_covid_bra['cases_per_day']['2021-06-16':'2021-06-23'],
                         mode='lines+markers',
                         name='Cas/jour Brésil '))

fig.update_layout(
    title_text="Nouveaux cas en Brésil avec HW.")

fig.show()


# > Au niveau des métriques, ce modèle est légèrement meilleurs que les deux autres au niveau du R2 (0,55) mais ne gagne pas sur l'erreur absolue moyenne en % à 17,58%. La fenetre entre le MAE et le RMSE est entre 11 000 et 15 000.

# In[71]:


mape_hw= MAPE(df_covid_bra['2021-06-16':'2021-06-22']['cases_per_day'], model_hw_pred.yhat)
r2_hw= r2_score(df_covid_bra['2021-06-16':'2021-06-22']['cases_per_day'],model_hw_pred.yhat)
rmse_hw = mean_squared_error(df_covid_bra['2021-06-16':'2021-06-22']['cases_per_day'], model_hw_pred.yhat, squared=False)
mae_hw = mean_absolute_error(df_covid_bra['2021-06-16':'2021-06-22']['cases_per_day'],model_hw_pred.yhat)

print('HW RMSE: ', rmse_hw.round(2))
print('HW MAPE ', mape_hw.round(2))
print('HW R2: ', r2_hw.round(2))
print('HW MAE: ', mae_hw.round(2))


# #### Bilan Brésil

# > Pour ce qui est de la prédiction du Brésil, on constate qu'il n'y a pas de grande différence de performance entre les différents modèles. 

# In[72]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=df_bresil_dt[-prediction_w:].index, y=df_bresil_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour Bresil'))

fig.add_trace(go.Scatter(x=model_hw_pred.index, y=model_hw_pred.yhat,
                         mode='lines+markers',
                         name='Cas/jour Bresil HW'))

fig.add_trace(go.Scatter(x=sarimax_results[-7:].index, y=sarimax_results[-7:].yhat,
                         mode='lines+markers',
                         name='Cas/jour Bresil SARIMAX'))

fig.add_trace(go.Scatter(x=f_df[-window:]['ds'][-7:], y=f_df[-window:]['yhat'][-7:],
                         mode='lines+markers',
                         name='Cas/jour Bresil Prophet'))


fig.add_trace(go.Scatter(x=df_covid_bra['cases_per_day']['2021-06-15':'2021-06-23'].index,
                         y=df_covid_bra['cases_per_day']['2021-06-15':'2021-06-23'],
                         mode='lines+markers',
                         name='Cas/jour Brésil '))

fig.update_layout(
    title_text="Nouveaux cas au Bresil avec HW/PROPHET/SARIMA.")

fig.show()


# In[73]:


d = [ ["SARIMA", mape_sarimax, mae_sarimax, r2_sarimax.round(2), rmse_sarimax],
     ["PROPHET", mape_p, mae_p, r2_p.round(2), rmse_p],
     ["HOLT-WINTERS", mape_hw, mae_hw, r2_hw.round(2), rmse_hw]]

print(tabulate(d, headers=["Modèle", "MAPE", "MAE", "R2", "RMSE"]))


# ### Prédiction du nombre de cas quotidiens en France

# > Nous allons faire à l'aide de nos trois modèles la prédiction sur les 7 prochains jours du nombre de cas quotidien en France. Nous n'allons pas détailler autant que précedement chaque étapes car nous reprenons les même codes. Nous alllons comparer les métriques notament le coefficient de determination et le MAPE pour voir si selon les differents pays et évolution du nombres de cas par jour un modèle de prédiction peut être plus ou moins performant.    

# In[74]:


#Création du datframe
df_france = df_model(df_covid_fr).dropna()
df_france_dt = df_france.set_index('ds')


# #### Prédiction avec SARIMA (France)

# In[75]:


results=pm.auto_arima(df_france_dt['y'], start_p=0, d=None, start_q=0, max_p=2, max_q=2,
                      seasonal=True, m=7, D=None, test='adf', start_P=0, start_Q=0, max_P=3, max_Q=3,
                      information_criterion='aic', trace=True, error_action='ignore',
                      trend=None,with_intercept=False, stepwise=True)


# In[76]:


model_s = SARIMAX(df_france_dt['y'], order=(2,1,2), seasonal_order=(2,0,1,7))
model_s_fit = model_s.fit()


# In[77]:


model_s_fit.summary()


# In[78]:


model_s_fit.plot_diagnostics(figsize=(8,8))
plt.show()


# In[79]:


prediction = model_s_fit.get_prediction(start=-prediction_w)
mean_prediction = prediction.predicted_mean
sarimax_prediction = pd.DataFrame({'yhat':mean_prediction})


# In[80]:


forecast = model_s_fit.get_forecast(steps=forecast_w)
mean_forecast=forecast.predicted_mean

sarimax_forecast = pd.DataFrame({'yhat':mean_forecast})
sarimax_results = sarimax_prediction.append(sarimax_forecast)


# In[81]:


#On enlève les prédictions inférieur à 0 (iln epeut pas y avoir moins de cas que 0)
sarimax_results.loc[sarimax_results['yhat']<0] = 0


# In[82]:


# Visualisation de la prévision du nombre de cas par jour avec le modèle SARIMA en France.
fig = go.Figure()

# Courbe du nombre de cas quotidiens sur les 3 dernieres semaines (jeu d'entrainemnt) 
fig.add_trace(go.Scatter(x=df_france_dt[-prediction_w:].index, y=df_france_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour France'))

# Courbe de la prédiction SARIMA.
fig.add_trace(go.Scatter(x=sarimax_results.index, y=sarimax_results.yhat,
                         mode='lines+markers',
                         name='Cas/jour France SARIMAX'))

# Courbe du nombre de cas quotidien lors des 7 jours prédit au France
fig.add_trace(go.Scatter(x=df_covid_fr['cases_per_day']['2021-06-15':'2021-06-23'].index,
                         y=df_covid_fr['cases_per_day']['2021-06-15':'2021-06-23'],
                         mode='lines+markers',
                         name='Cas/jour France '))

fig.update_layout(
    title_text="Nouveaux cas en France avec SARIMAX.")


# > On constate que pour la France le MAPE n'est pas correct. 

# In[83]:


mape_sarimax= MAPE(df_france_dt[-prediction_w:]['y'], mean_prediction.values)
r2_sarimax= r2_score(df_france_dt[-prediction_w:]['y'],mean_prediction.values)
rmse_sarimax = mean_squared_error(df_france_dt[-prediction_w:]['y'], mean_prediction.values, squared=False)
mae_sarimax = mean_absolute_error(df_france_dt[-prediction_w:]['y'],mean_prediction.values)

print('SARIMAX MAPE: ', mape_sarimax.round(2))
print('SARIMAX MAE: ', mae_sarimax.round(2))
print('SARIMAX R2: ', r2_sarimax.round(2))
print('SARIMAX RMSE: ', rmse_sarimax.round(2))


# #### Prédiction avec Prophet (France)

# In[84]:


model_p = Prophet(n_changepoints=50,
                 seasonality_mode='additive',
                 changepoint_prior_scale=1)
model_p.add_seasonality('weekly', period = 7, fourier_order = 8)
model_p.add_seasonality('yearly', period = 365, fourier_order = 19)
model_p_fit = model_p.fit(df_france)


# In[85]:


future = model_p.make_future_dataframe(periods=forecast_w)
forecast = model_p.predict(future)


# In[86]:


model_p.plot(forecast);
plt.title("Nouveaux cas covid19 en France et prévision by Prophet")
plt.ylabel("Cas/jour")
plt.show()


# In[87]:


fig = model_p.plot_components(forecast)


# In[88]:


f_df = forecast[['ds', 'yhat']]
f_df['yhat'].loc[f_df['yhat']<0] =0


# In[89]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=df_france_dt[-prediction_w:].index, y=df_france_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour France'))

fig.add_trace(go.Scatter(x=f_df[-window:]['ds'], y=f_df[-window:]['yhat'],
                         mode='lines+markers',
                         name='Cas/jour France Prophet'))


fig.add_trace(go.Scatter(x=df_covid_fr['cases_per_day']['2021-06-15':'2021-06-23'].index,
                         y=df_covid_fr['cases_per_day']['2021-06-15':'2021-06-23'],
                         mode='lines+markers',
                         name='Cas/jour France '))

fig.update_layout(
    title_text="Nouveaux cas en France avec Prophet.")

fig.show()


# In[90]:


mape_p= MAPE(df_covid_fr['2021-05-26':'2021-06-22']['cases_per_day'], f_df[-window:]['yhat'])
r2_p= r2_score(df_covid_fr['2021-05-26':'2021-06-22']['cases_per_day'],f_df[-window:]['yhat'])
rmse_p = mean_squared_error(df_covid_fr['2021-05-26':'2021-06-22']['cases_per_day'], f_df[-window:]['yhat'], squared=False)
mae_p = mean_absolute_error(df_covid_fr['2021-05-26':'2021-06-22']['cases_per_day'],f_df[-window:]['yhat'])

print('PROPHET RMSE: ', rmse_p.round(2))
print('PROPHET MAPE: ', mape_p.round(2))
print('PROPHET R2: ', r2_p.round(2))
print('PROPHET MAE: ', mae_p.round(2))


# In[ ]:





# #### Prédiction avec Holt Winters (France)
# 

# In[91]:


model_hw = ExponentialSmoothing(df_france_dt['y'][-prediction_w:], seasonal_periods=7, trend='add', seasonal='add')


# In[92]:



model_hw_fit = model_hw.fit()
print(model_hw_fit.summary())


# In[93]:


model_hw_pred = model_hw_fit.forecast((7))
model_hw_pred = pd.DataFrame(model_hw_pred)
model_hw_pred.rename(columns={0:'yhat'}, inplace=True)

model_hw_pred['yhat'].loc[model_hw_pred['yhat']<0]=0


# In[94]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=df_france_dt[-prediction_w:].index, y=df_france_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour France'))

fig.add_trace(go.Scatter(x=model_hw_pred.index, y=model_hw_pred.yhat,
                         mode='lines+markers',
                         name='Cas/jour France HW'))


fig.add_trace(go.Scatter(x=df_covid_fr['cases_per_day']['2021-06-15':'2021-06-23'].index,
                         y=df_covid_fr['cases_per_day']['2021-06-15':'2021-06-23'],
                         mode='lines+markers',
                         name='Cas/jour France '))

fig.update_layout(
    title_text="Nouveaux cas en France avec HW.")

fig.show()


# In[95]:


mape_hw= MAPE(df_covid_fr['2021-06-16':'2021-06-22']['cases_per_day'], model_hw_pred.yhat)
r2_hw= r2_score(df_covid_fr['2021-06-16':'2021-06-22']['cases_per_day'],model_hw_pred.yhat)
rmse_hw = mean_squared_error(df_covid_fr['2021-06-16':'2021-06-22']['cases_per_day'], model_hw_pred.yhat, squared=False)
mae_hw = mean_absolute_error(df_covid_fr['2021-06-16':'2021-06-22']['cases_per_day'],model_hw_pred.yhat)

print('HW RMSE: ', rmse_hw.round(2))
print('HW MAPE ', mape_hw.round(2))
print('HW R2: ', r2_hw.round(2))
print('HW MAE: ', mae_hw.round(2))


# #### Bilan France

# > Lorsqu'on effectue les prédictions de la France on remarque des errurs dans les métriques. On retrouve beaucoup de valeur extreme ou d'écart entre deux jours de données (entre le 30 mai et le 4 juin on a des valeurs qui ne suivent pas une tendaces précise). On ne retrouve pas une saisonnalité marquée comme on peut le voir sur les jeux de données du Brésil et du Japon notamment. On a donc une prédiction moins performante, voir mauvaise. La solution peut etre de prendre le nombre de cas quotidiens lissé sur 7 jours (à l'aide de la moyenne mobile) pour voir si l'on a une meilleur performance. 
# >
# > EDIT : On obtient de meilleurs resultats après le lissage sur 7jours pour la France seulement, on peut voir le code en après la prédiction du Japon.

# In[96]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=df_france_dt[-prediction_w:].index, y=df_france_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour France'))

fig.add_trace(go.Scatter(x=model_hw_pred.index, y=model_hw_pred.yhat,
                         mode='lines+markers',
                         name='Cas/jour France HW'))

fig.add_trace(go.Scatter(x=sarimax_results[-7:].index, y=sarimax_results[-7:].yhat,
                         mode='lines+markers',
                         name='Cas/jour France SARIMAX'))

fig.add_trace(go.Scatter(x=f_df[-window:]['ds'][-7:], y=f_df[-window:]['yhat'][-7:],
                         mode='lines+markers',
                         name='Cas/jour France Prophet'))


fig.add_trace(go.Scatter(x=df_covid_fr['cases_per_day']['2021-06-15':'2021-06-23'].index,
                         y=df_covid_fr['cases_per_day']['2021-06-15':'2021-06-23'],
                         mode='lines+markers',
                         name='Cas/jour France '))

fig.update_layout(
    title_text="Nouveaux cas au France avec HW/PROPHET/SARIMA.")

fig.show()


# In[97]:


d = [ ["SARIMA", mape_sarimax, mae_sarimax, r2_sarimax, rmse_sarimax],
     ["PROPHET", mape_p, mae_p, r2_p, rmse_p],
     ["HOLT-WINTERS", mape_hw, mae_hw, r2_hw, rmse_hw]]

print(tabulate(d, headers=["Modèle", "MAPE", "MAE", "R2", "RMSE"]))


# ### Prédiction du nombre de cas quotidien au Japon

# > Pour finir, on regarde la prédiction des cas confirmés quotidiens de la Covid-19 au Japon.

# In[98]:


#Création du datframe

df_japon = df_model(df_covid_jap).dropna()

df_japon_dt = df_japon.set_index('ds')


# #### Prédiction avec SARIMA (Japon)

# In[99]:


results=pm.auto_arima(df_japon_dt['y'], start_p=0, d=None, start_q=0, max_p=3, max_q=3,
                      seasonal=True, m=7, D=None, test='adf', start_P=0, start_Q=0, max_P=3, max_Q=3,
                      information_criterion='aic', trace=True, error_action='ignore',
                      trend=None,with_intercept=False, stepwise=True)


# In[ ]:





# In[100]:


model_s = SARIMAX(df_japon_dt['y'], order=(1,1,0), seasonal_order=(1,0,2,7))
model_s_fit = model_s.fit()
model_s_fit.summary()


# In[101]:


model_s_fit.plot_diagnostics(figsize=(8,8))
plt.show()


# In[102]:


prediction = model_s_fit.get_prediction(start=-prediction_w)
mean_prediction = prediction.predicted_mean
sarimax_prediction = pd.DataFrame({'yhat':mean_prediction})


# In[103]:


forecast = model_s_fit.get_forecast(steps=forecast_w)
mean_forecast=forecast.predicted_mean

sarimax_forecast = pd.DataFrame({'yhat':mean_forecast})
sarimax_results = sarimax_prediction.append(sarimax_forecast)


# In[104]:


sarimax_results.loc[sarimax_results['yhat']<0] = 0


# In[105]:


# Visualisation de la prévision du nombre de cas par jour avec le modèle SARIMA en Japon.
fig = go.Figure()

# Courbe du nombre de cas quotidiens sur les 3 dernieres semaines (jeu d'entrainemnt) 
fig.add_trace(go.Scatter(x=df_japon_dt[-prediction_w:].index, y=df_japon_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour Japon'))

# Courbe de la prédiction SARIMA.
fig.add_trace(go.Scatter(x=sarimax_results.index, y=sarimax_results.yhat,
                         mode='lines+markers',
                         name='Cas/jour Japon SARIMAX'))

# Courbe du nombre de cas quotidien lors des 7 jours prédit au Japon
fig.add_trace(go.Scatter(x=df_covid_jap['cases_per_day']['2021-06-15':'2021-06-23'].index,
                         y=df_covid_jap['cases_per_day']['2021-06-15':'2021-06-23'],
                         mode='lines+markers',
                         name='Cas/jour Japon '))

fig.update_layout(
    title_text="Nouveaux cas en Japon avec SARIMAX.")
fig.show()


# In[106]:


mape_sarimax= MAPE(df_japon_dt[-prediction_w:]['y'], mean_prediction.values)
r2_sarimax= r2_score(df_japon_dt[-prediction_w:]['y'],mean_prediction.values)
rmse_sarimax = mean_squared_error(df_japon_dt[-prediction_w:]['y'], mean_prediction.values, squared=False)
mae_sarimax = mean_absolute_error(df_japon_dt[-prediction_w:]['y'],mean_prediction.values)

print('SARIMAX MAPE: ', mape_sarimax.round(2))
print('SARIMAX MAE: ', mae_sarimax.round(2))
print('SARIMAX R2: ', r2_sarimax.round(2))
print('SARIMAX RMSE: ', rmse_sarimax.round(2))


# #### Prédiction avec Prophet (Japon)

# In[107]:


model_p = Prophet(n_changepoints=50,
                 seasonality_mode='additive',
                 changepoint_prior_scale=1)
model_p.add_seasonality('weekly', period = 7, fourier_order = 8)
model_p.add_seasonality('yearly', period = 365, fourier_order = 19)
model_p_fit = model_p.fit(df_japon)


# In[108]:


future = model_p.make_future_dataframe(periods=forecast_w)
forecast = model_p.predict(future)

model_p.plot(forecast);
plt.title("Nouveaux cas covid19 en Japon et prévision by Prophet")
plt.ylabel("Cas/jour")
plt.show()


# In[109]:


fig = model_p.plot_components(forecast)
f_df = forecast[['ds', 'yhat']]
f_df['yhat'].loc[f_df['yhat']<0] =0


# In[110]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=df_japon_dt[-prediction_w:].index, y=df_japon_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour Japon'))

fig.add_trace(go.Scatter(x=f_df[-window:]['ds'], y=f_df[-
                                                        window:]['yhat'],
                         mode='lines+markers',
                         name='Cas/jour Japon Prophet'))


fig.add_trace(go.Scatter(x=df_covid_jap['cases_per_day']['2021-06-15':'2021-06-23'].index,
                         y=df_covid_jap['cases_per_day']['2021-06-15':'2021-06-23'],
                         mode='lines+markers',
                         name='Cas/jour Japon '))

fig.update_layout(
    title_text="Nouveaux cas en Japon avec Prophet.")

fig.show()


# In[111]:


mape_p= MAPE(df_covid_jap['2021-05-26':'2021-06-22']['cases_per_day'], f_df[-window:]['yhat'])
r2_p= r2_score(df_covid_jap['2021-05-26':'2021-06-22']['cases_per_day'],f_df[-window:]['yhat'])
rmse_p = mean_squared_error(df_covid_jap['2021-05-26':'2021-06-22']['cases_per_day'], f_df[-window:]['yhat'], squared=False)
mae_p = mean_absolute_error(df_covid_jap['2021-05-26':'2021-06-22']['cases_per_day'],f_df[-window:]['yhat'])

print('PROPHET RMSE: ', rmse_p.round(2))
print('PROPHET MAPE: ', mape_p.round(2))
print('PROPHET R2: ', r2_p.round(2))
print('PROPHET MAE: ', mae_p.round(2))


# #### Prédiction avec Holt Winters (Japon)

# In[112]:


model_hw = ExponentialSmoothing(df_japon_dt['y'][-prediction_w:], seasonal_periods=7, trend='add', seasonal='add')
model_hw_fit = model_hw.fit()
print(model_hw_fit.summary())


# In[113]:


model_hw_pred = model_hw_fit.forecast((7))
model_hw_pred = pd.DataFrame(model_hw_pred)
model_hw_pred.rename(columns={0:'yhat'}, inplace=True)


# In[114]:


model_hw_pred['yhat'].loc[model_hw_pred['yhat']<0]=0


# In[115]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=df_japon_dt[-prediction_w:].index, y=df_japon_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour Japon'))

fig.add_trace(go.Scatter(x=model_hw_pred.index, y=model_hw_pred.yhat,
                         mode='lines+markers',
                         name='Cas/jour Japon HW'))


fig.add_trace(go.Scatter(x=df_covid_jap['cases_per_day']['2021-06-16':'2021-06-23'].index,
                         y=df_covid_jap['cases_per_day']['2021-06-16':'2021-06-23'],
                         mode='lines+markers',
                         name='Cas/jour Japon '))

fig.update_layout(
    title_text="Nouveaux cas en Japon avec HW.")

fig.show()


# In[116]:


mape_hw= MAPE(df_covid_jap['2021-06-16':'2021-06-22']['cases_per_day'], model_hw_pred.yhat)
r2_hw= r2_score(df_covid_jap['2021-06-16':'2021-06-22']['cases_per_day'],model_hw_pred.yhat)
rmse_hw = mean_squared_error(df_covid_jap['2021-06-16':'2021-06-22']['cases_per_day'], model_hw_pred.yhat, squared=False)
mae_hw = mean_absolute_error(df_covid_jap['2021-06-16':'2021-06-22']['cases_per_day'],model_hw_pred.yhat)

print('HW RMSE: ', rmse_hw.round(2))
print('HW MAPE ', mape_hw.round(2))
print('HW R2: ', r2_hw.round(2))
print('HW MAE: ', mae_hw.round(2))


# #### Bilan Japon

# > Au japon, les trois modèles fonctionnent très bien, notamment SARIMA avec un coefficient de determination de plus de 0,9 et une prédiction des 7 pchains jours précis à plus de 90%. Les données sont très marquées sur la semaine et le nombre de cas pas trop élevés ce qui explique la bonne performance des prédictions.

# In[117]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=df_japon_dt[-prediction_w:].index, y=df_japon_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour Japon'))

fig.add_trace(go.Scatter(x=model_hw_pred.index, y=model_hw_pred.yhat,
                         mode='lines+markers',
                         name='Cas/jour Japon HW'))

fig.add_trace(go.Scatter(x=sarimax_results[-7:].index, y=sarimax_results[-7:].yhat,
                         mode='lines+markers',
                         name='Cas/jour Japon SARIMAX'))

fig.add_trace(go.Scatter(x=f_df[-window:]['ds'][-7:], y=f_df[-window:]['yhat'][-7:],
                         mode='lines+markers',
                         name='Cas/jour Japon Prophet'))


fig.add_trace(go.Scatter(x=df_covid_jap['cases_per_day']['2021-06-15':'2021-06-23'].index,
                         y=df_covid_jap['cases_per_day']['2021-06-15':'2021-06-23'],
                         mode='lines+markers',
                         name='Cas/jour Japon '))

fig.update_layout(
    title_text="Nouveaux cas au Japon avec HW/PROPHET/SARIMA.")

fig.show()


# In[118]:


d = [ ["SARIMA", mape_sarimax, mae_sarimax, r2_sarimax, rmse_sarimax],
     ["PROPHET", mape_p, mae_p, r2_p, rmse_p],
     ["HOLT-WINTERS", mape_hw, mae_hw, r2_hw, rmse_hw]]

print(tabulate(d, headers=["Modèle", "MAPE", "MAE", "R2", "RMSE"]))


# ### EDIT : Prédiction France avec les données lissées sur 7 jours

# > Nous allons voir avant de conclure que nos modèles de prédictions pour certains pays peuvent être plus performant avec des données lissées sur 7 jours. C'est le cas pour la France à cause comme on l'a dit précedemment des données pas assez saisonniées avec des trop grandes differences d'un jour à l'aure, des valeurs abérantes possiblemment. Le principe reste le même, on change le dataframe de départ en prenant la moyenne mobile des cas par jour. 

# In[119]:


#La fonction df_model() permet de creer un dataframe contenant seulement la variable date et le nombre de cas par jour.
def df_model_roll(df):
    df = pd.DataFrame(df['cases_per_day']['2020-01-22':'2021-06-15'].rolling(window = 7, center = True).mean().round())
    df = df.reset_index()
    df = df.rename(columns={"date":"ds","cases_per_day":"y"})
    return df


# In[120]:


#Création du datframe
df_france = df_model_roll(df_covid_fr).dropna()
df_france_dt = df_france.set_index('ds')


# #### Prédiction avec SARIMA (France données lissées)

# In[121]:


model_s = SARIMAX(df_france_dt['y'], order=(1,1,1), seasonal_order=(2,0,1,7))
model_s_fit = model_s.fit()


# In[122]:


prediction = model_s_fit.get_prediction(start=-prediction_w)
mean_prediction = prediction.predicted_mean
sarimax_prediction = pd.DataFrame({'yhat':mean_prediction})


# In[123]:


forecast = model_s_fit.get_forecast(steps=forecast_w)
mean_forecast=forecast.predicted_mean

sarimax_forecast = pd.DataFrame({'yhat':mean_forecast})
sarimax_results = sarimax_prediction.append(sarimax_forecast)


# In[124]:


# Visualisation de la prévision du nombre de cas par jour avec le modèle SARIMA en France.
fig = go.Figure()

# Courbe du nombre de cas quotidiens sur les 3 dernieres semaines (jeu d'entrainemnt) 
fig.add_trace(go.Scatter(x=df_france_dt[-prediction_w:].index, y=df_france_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour France'))

# Courbe de la prédiction SARIMA.
fig.add_trace(go.Scatter(x=sarimax_results.index, y=sarimax_results.yhat,
                         mode='lines+markers',
                         name='Cas/jour France SARIMAX'))

# Courbe du nombre de cas quotidien lors des 7 jours prédit au France
fig.add_trace(go.Scatter(x=df_covid_fr['cases_per_day']['2021-06-8':'2021-06-23'].index,
                         y=df_covid_fr['cases_per_day']['2021-06-8':'2021-06-23'].rolling(window = 7, center = True).mean().round(),
                         mode='lines+markers',
                         name='Cas/jour France '))

fig.update_layout(
    title_text="Nouveaux cas en France avec SARIMAX.")


# In[125]:


mape_sarimax= MAPE(df_france_dt[-prediction_w:]['y'], mean_prediction.values)
r2_sarimax= r2_score(df_france_dt[-prediction_w:]['y'],mean_prediction.values)
rmse_sarimax = mean_squared_error(df_france_dt[-prediction_w:]['y'], mean_prediction.values, squared=False)
mae_sarimax = mean_absolute_error(df_france_dt[-prediction_w:]['y'],mean_prediction.values)

print('SARIMAX MAPE: ', mape_sarimax.round(2))
print('SARIMAX MAE: ', mae_sarimax.round(2))
print('SARIMAX R2: ', r2_sarimax.round(2))
print('SARIMAX RMSE: ', rmse_sarimax.round(2))


# #### Prédiction avec Prophet (France données lissées)

# In[126]:


model_p = Prophet(n_changepoints=50,
                 seasonality_mode='additive',
                 changepoint_prior_scale=1)
model_p.add_seasonality('weekly', period = 7, fourier_order = 8)
model_p.add_seasonality('yearly', period = 365, fourier_order = 19)
model_p_fit = model_p.fit(df_france)


# In[127]:


future = model_p.make_future_dataframe(periods=forecast_w)
forecast = model_p.predict(future)


# In[128]:


f_df = forecast[['ds', 'yhat']]


# In[129]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=df_france_dt[-prediction_w:].index, y=df_france_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour France'))

fig.add_trace(go.Scatter(x=f_df[-window:]['ds'], y=f_df[-window:]['yhat'],
                         mode='lines+markers',
                         name='Cas/jour France Prophet'))


fig.add_trace(go.Scatter(x=df_covid_fr['cases_per_day']['2021-06-8':'2021-06-23'].index,
                         y=df_covid_fr['cases_per_day']['2021-06-8':'2021-06-23'].rolling(window = 7, center = True).mean().round(),
                         mode='lines+markers',
                         name='Cas/jour France '))

fig.update_layout(
    title_text="Nouveaux cas en France avec Prophet.")

fig.show()


# In[130]:


r = df_covid_fr['cases_per_day']['2021-05-20':'2021-06-23'].rolling(window = 7, center = True).mean().round().dropna()

mape_p= MAPE(r, f_df[-window:]['yhat'])
r2_p= r2_score(r,f_df[-window:]['yhat'])
rmse_p = mean_squared_error(r, f_df[-window:]['yhat'], squared=False)
mae_p = mean_absolute_error(r,f_df[-window:]['yhat'])

print('PROPHET RMSE: ', rmse_p.round(2))
print('PROPHET MAPE: ', mape_p.round(2))
print('PROPHET R2: ', r2_p.round(2))
print('PROPHET MAE: ', mae_p.round(2))


# #### Prédiction avec Holt Winters (France données lissées)
# 

# In[131]:



model_hw = ExponentialSmoothing(df_france_dt['y'][-prediction_w:], seasonal_periods=7, trend='add', seasonal='add')
model_hw_fit = model_hw.fit()


# In[132]:


model_hw_pred = model_hw_fit.forecast((7))
model_hw_pred = pd.DataFrame(model_hw_pred)
model_hw_pred.rename(columns={0:'yhat'}, inplace=True)


# In[133]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=df_france_dt[-prediction_w:].index, y=df_france_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour France'))

fig.add_trace(go.Scatter(x=model_hw_pred.index, y=model_hw_pred.yhat,
                         mode='lines+markers',
                         name='Cas/jour France HW'))


fig.add_trace(go.Scatter(x=df_covid_fr['cases_per_day']['2021-06-8':'2021-06-23'].index,
                         y=df_covid_fr['cases_per_day']['2021-06-8':'2021-06-23'].rolling(window = 7, center = True).mean().round(),
                         mode='lines+markers',
                         name='Cas/jour France '))

fig.update_layout(
    title_text="Nouveaux cas en France avec HW.")

fig.show()


# In[134]:


r = df_covid_fr['cases_per_day']['2021-06-10':'2021-06-23'].rolling(window = 7, center = True).mean().round().dropna()

mape_hw= MAPE(r, model_hw_pred.yhat)
r2_hw= r2_score(r,model_hw_pred.yhat)
rmse_hw = mean_squared_error(r, model_hw_pred.yhat, squared=False)
mae_hw = mean_absolute_error(r,model_hw_pred.yhat)

print('HW RMSE: ', rmse_hw.round(2))
print('HW MAPE ', mape_hw.round(2))
print('HW R2: ', r2_hw.round(2))
print('HW MAE: ', mae_hw.round(2))


# #### Bilan  France (données lissées)

# > On voit à l'aide des métriques la nette amélioration des perfomance de nos modèles, même si la prédiction SARIMA n'est pas vraiment ce ue nous attendions comme on peu le voir ce le graphique.

# In[135]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=df_france_dt[-prediction_w:].index, y=df_france_dt[-prediction_w:]['y'],
                         mode='lines+markers',
                         name='Cas/jour France'))

fig.add_trace(go.Scatter(x=model_hw_pred.index, y=model_hw_pred.yhat,
                         mode='lines+markers',
                         name='Cas/jour France HW'))

fig.add_trace(go.Scatter(x=sarimax_results[-7:].index, y=sarimax_results[-7:].yhat,
                         mode='lines+markers',
                         name='Cas/jour France SARIMAX'))

fig.add_trace(go.Scatter(x=f_df[-window:]['ds'][-7:], y=f_df[-window:]['yhat'][-7:],
                         mode='lines+markers',
                         name='Cas/jour France Prophet'))


fig.add_trace(go.Scatter(x=df_covid_fr['cases_per_day']['2021-06-8':'2021-06-23'].index,
                         y=df_covid_fr['cases_per_day']['2021-06-8':'2021-06-23'].rolling(window = 7, center = True).mean().round(),
                         mode='lines+markers',
                         name='Cas/jour France '))

fig.update_layout(
    title_text="Nouveaux cas au France avec HW/PROPHET/SARIMA.")

fig.show()


# In[ ]:





# In[136]:


d = [ ["SARIMA", mape_sarimax, mae_sarimax, r2_sarimax, rmse_sarimax],
     ["PROPHET", mape_p, mae_p, r2_p, rmse_p],
     ["HOLT-WINTERS", mape_hw, mae_hw, r2_hw, rmse_hw]]

print(tabulate(d, headers=["Modèle", "MAPE", "MAE", "R2", "RMSE"]))


# In[ ]:





# In[ ]:





# ## Fonction de prédion pour chaque modèle

# > Pour chaque modèle de prédiction, nous avons crée une fonction qui comprend en parametre le dataframe original du pays, un boléen *rolling* si l'on veut lissée nos données sur 7 jours et pour le modèle SARIMA, les hyperparametres à remplir. Dans le notebook, nous n'utilisons pas ces fonctions pour une question d'esthetisme et de facilité de compréhension, mais voici les fonctions et un exemple avec nos pays choisie.  

# ### SARIMA

# In[137]:


def SARIMA(df_covid, p, d, q, a_1, a_2, a_3, rolling):
    
    if rolling == False:
        
        df = pd.DataFrame(df_covid['cases_per_day']['2020-01-22':'2021-06-15'])
        df = df.reset_index()
        df = df.rename(columns={"date":"ds","cases_per_day":"y"})

        df_s = df.dropna()
        df_s_dt = df_s.set_index('ds')

        model_s = SARIMAX(df_s_dt['y'], order=(p,d,q), seasonal_order=(a_1,a_2,a_3,7))
        model_s_fit = model_s.fit()

        print(model_s_fit.summary())

        model_s_fit.plot_diagnostics(figsize=(8,8))
        plt.show()

        prediction = model_s_fit.get_prediction(start=-prediction_w)
        mean_prediction = prediction.predicted_mean

        sarimax_prediction = pd.DataFrame({'yhat':mean_prediction})

        forecast = model_s_fit.get_forecast(steps=forecast_w)
        mean_forecast=forecast.predicted_mean

        sarimax_forecast = pd.DataFrame({'yhat':mean_forecast})
        sarimax_results = sarimax_prediction.append(sarimax_forecast)
        
        sarimax_results.loc[sarimax_results['yhat']<0] = 0

        fig = go.Figure()

        # Courbe du nombre de cas quotidiens sur les 3 dernieres semaines (jeu d'entrainemnt) 
        fig.add_trace(go.Scatter(x=df_s_dt[-prediction_w:].index, y=df_s_dt[-prediction_w:]['y'],
                             mode='lines+markers',
                             name='Cas/jour '))

        # Courbe de la prédiction SARIMA.
        fig.add_trace(go.Scatter(x=sarimax_results.index, y=sarimax_results.yhat,
                             mode='lines+markers',
                             name='Cas/jour  SARIMAX'))

        # Courbe du nombre de cas quotidien lors des 7 jours prédit au Brésil
        fig.add_trace(go.Scatter(x=df_covid['cases_per_day']['2021-06-15':'2021-06-23'].index,
                             y=df_covid['cases_per_day']['2021-06-15':'2021-06-23'],
                             mode='lines+markers',
                             name='Cas/jour  '))

        fig.update_layout(
            title_text="Nouveaux cas quotidien avec SARIMAX.")

        fig.show()

        mape_sarimax= MAPE(df_s_dt[-prediction_w:]['y'], mean_prediction.values)
        r2_sarimax= r2_score(df_s_dt[-prediction_w:]['y'],mean_prediction.values)
        rmse_sarimax = mean_squared_error(df_s_dt[-prediction_w:]['y'], mean_prediction.values, squared=False)
        mae_sarimax = mean_absolute_error(df_s_dt[-prediction_w:]['y'],mean_prediction.values)

        print('SARIMAX MAPE: ', mape_sarimax.round(2))
        print('SARIMAX MAE: ', mae_sarimax.round(2))
        print('SARIMAX R2: ', r2_sarimax.round(2))
        print('SARIMAX RMSE: ', rmse_sarimax.round(2))
    
    else:
        
        df = pd.DataFrame(df_covid['cases_per_day']['2020-01-22':'2021-06-15'].rolling(window = 7, center = True).mean().round())
        df = df.reset_index()
        df = df.rename(columns={"date":"ds","cases_per_day":"y"})

        df_s = df.dropna()
        df_s_dt = df_s.set_index('ds')

        model_s = SARIMAX(df_s_dt['y'], order=(p,d,q), seasonal_order=(a_1,a_2,a_3,7))
        model_s_fit = model_s.fit()

        print(model_s_fit.summary())

        model_s_fit.plot_diagnostics(figsize=(8,8))
        plt.show()

        prediction = model_s_fit.get_prediction(start=-prediction_w)
        mean_prediction = prediction.predicted_mean

        sarimax_prediction = pd.DataFrame({'yhat':mean_prediction})

        forecast = model_s_fit.get_forecast(steps=forecast_w)
        mean_forecast=forecast.predicted_mean

        sarimax_forecast = pd.DataFrame({'yhat':mean_forecast})
        sarimax_results = sarimax_prediction.append(sarimax_forecast)
        
        sarimax_results.loc[sarimax_results['yhat']<0] = 0

        # Visualisation de la prévision du nombre de cas par jour avec le modèle SARIMA en France.
        fig = go.Figure()
    
        # Courbe du nombre de cas quotidiens sur les 3 dernieres semaines (jeu d'entrainemnt) 
        fig.add_trace(go.Scatter(x=df_s_dt[-prediction_w:].index, y=df_s_dt[-prediction_w:]['y'],
                                 mode='lines+markers',
                                 name='Cas/jour '))

        # Courbe de la prédiction SARIMA.
        fig.add_trace(go.Scatter(x=sarimax_results.index, y=sarimax_results.yhat,
                                 mode='lines+markers',
                                 name='Cas/jour  SARIMAX'))

        # Courbe du nombre de cas quotidien lors des 7 jours prédit au France
        fig.add_trace(go.Scatter(x=df_covid['cases_per_day']['2021-06-8':'2021-06-23'].index,
                                 y=df_covid['cases_per_day']['2021-06-8':'2021-06-23'].rolling(window = 7, center = True).mean().round(),
                                 mode='lines+markers',
                                 name='Cas/jour  '))

        fig.update_layout(
            title_text="Nouveaux cas en  avec SARIMAX.")
        fig.show()

        mape_sarimax= MAPE(df_s_dt[-prediction_w:]['y'], mean_prediction.values)
        r2_sarimax= r2_score(df_s_dt[-prediction_w:]['y'],mean_prediction.values)
        rmse_sarimax = mean_squared_error(df_s_dt[-prediction_w:]['y'], mean_prediction.values, squared=False)
        mae_sarimax = mean_absolute_error(df_s_dt[-prediction_w:]['y'],mean_prediction.values)

        print('SARIMAX MAPE: ', mape_sarimax.round(2))
        print('SARIMAX MAE: ', mae_sarimax.round(2))
        print('SARIMAX R2: ', r2_sarimax.round(2))
        print('SARIMAX RMSE: ', rmse_sarimax.round(2))


# In[138]:


SARIMA(df_covid_bra, 0, 1, 1, 1, 0, 1, True)


# ### Prophet

# In[139]:


def PROPHET(df_covid, rolling):
    
    if rolling == False:
        
        df = pd.DataFrame(df_covid['cases_per_day']['2020-01-22':'2021-06-15'])
        df = df.reset_index()
        df = df.rename(columns={"date":"ds","cases_per_day":"y"})

        df_s = df.dropna()
        df_s_dt = df_s.set_index('ds')

        
        model_p = Prophet(n_changepoints=50,
                         seasonality_mode='additive',
                         changepoint_prior_scale=1)

        model_p.add_seasonality('weekly', period = 7, fourier_order = 8)
        model_p.add_seasonality('yearly', period = 365, fourier_order = 19)


        model_p_fit = model_p.fit(df_s)

        future = model_p.make_future_dataframe(periods=forecast_w)

        forecast = model_p.predict(future)

        
        model_p.plot(forecast);
        plt.title("Nouveaux cas covid19 et prévision by Prophet")
        plt.ylabel("Cas/jour")
        plt.show()

        #model_p.plot_components(forecast)

        f_df = forecast[['ds', 'yhat']]
        
        f_df = forecast[['ds', 'yhat']]
        f_df['yhat'].loc[f_df['yhat']<0] =0
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_s_dt[-prediction_w:].index, y=df_s_dt[-prediction_w:]['y'],
                                 mode='lines+markers',
                                 name='Cas/jour'))

        fig.add_trace(go.Scatter(x=f_df[-window:]['ds'], y=f_df[-window:]['yhat'],
                                 mode='lines+markers',
                                 name='Cas/jour Prophet'))


        fig.add_trace(go.Scatter(x=df_covid['cases_per_day']['2021-06-15':'2021-06-23'].index,
                                 y=df_covid['cases_per_day']['2021-06-15':'2021-06-23'],
                                 mode='lines+markers',
                                 name='Cas/jour  '))

        fig.update_layout(
            title_text="Nouveaux cas avec Prophet.")

        fig.show()

        mape_p= MAPE(df_covid['2021-05-26':'2021-06-22']['cases_per_day'], f_df[-window:]['yhat'])
        r2_p= r2_score(df_covid['2021-05-26':'2021-06-22']['cases_per_day'],f_df[-window:]['yhat'])
        rmse_p = mean_squared_error(df_covid['2021-05-26':'2021-06-22']['cases_per_day'], f_df[-window:]['yhat'], squared=False)
        mae_p = mean_absolute_error(df_covid['2021-05-26':'2021-06-22']['cases_per_day'],f_df[-window:]['yhat'])

        print('PROPHET RMSE: ', rmse_p.round(2))
        print('PROPHET MAPE: ', mape_p.round(2))
        print('PROPHET R2: ', r2_p.round(2))
        print('PROPHET MAE: ', mae_p.round(2))
    else:
        
        df = pd.DataFrame(df_covid['cases_per_day']['2020-01-22':'2021-06-15'].rolling(window = 7, center = True).mean().round())
        df = df.reset_index()
        df = df.rename(columns={"date":"ds","cases_per_day":"y"})

        df_s = df.dropna()
        df_s_dt = df_s.set_index('ds')
        
        model_p = Prophet(n_changepoints=50,
                         seasonality_mode='additive',
                         changepoint_prior_scale=1)

        model_p.add_seasonality('weekly', period = 7, fourier_order = 8)
        model_p.add_seasonality('yearly', period = 365, fourier_order = 19)


        model_p_fit = model_p.fit(df_s)

        future = model_p.make_future_dataframe(periods=forecast_w)

        forecast = model_p.predict(future)


        model_p.plot(forecast);
        plt.title("Nouveaux cas covid19 et prévision by Prophet")
        plt.ylabel("Cas/jour")
        plt.show()

        #model_p.plot_components(forecast)

        f_df = forecast[['ds', 'yhat']]
        
        f_df = forecast[['ds', 'yhat']]
        f_df['yhat'].loc[f_df['yhat']<0] =0

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_s_dt[-prediction_w:].index, y=df_s_dt[-prediction_w:]['y'],
                                 mode='lines+markers',
                                 name='Cas/jour'))

        fig.add_trace(go.Scatter(x=f_df[-window:]['ds'], y=f_df[-window:]['yhat'],
                                 mode='lines+markers',
                                 name='Cas/jour Prophet'))


        fig.add_trace(go.Scatter(x=df_covid['cases_per_day']['2021-06-8':'2021-06-23'].index,
                                 y=df_covid['cases_per_day']['2021-06-8':'2021-06-23'].rolling(window = 7, center = True).mean().round(),
                                 mode='lines+markers',
                                 name='Cas/jour '))

        fig.update_layout(
            title_text="Nouveaux cas avec Prophet.")

        fig.show()

        r = df_covid['cases_per_day']['2021-05-20':'2021-06-23'].rolling(window = 7, center = True).mean().round().dropna()

        mape_p= MAPE(r, f_df[-window:]['yhat'])
        r2_p= r2_score(r,f_df[-window:]['yhat'])
        rmse_p = mean_squared_error(r, f_df[-window:]['yhat'], squared=False)
        mae_p = mean_absolute_error(r,f_df[-window:]['yhat'])

        print('PROPHET RMSE: ', rmse_p.round(2))
        print('PROPHET MAPE: ', mape_p.round(2))
        print('PROPHET R2: ', r2_p.round(2))
        print('PROPHET MAE: ', mae_p.round(2))


# In[140]:


PROPHET(df_covid_bra, True)


# ### Holt-Winters

# In[141]:


def HOLT_WINTERS(df_covid, rolling):
    
    if rolling == False:
        df = pd.DataFrame(df_covid['cases_per_day']['2020-01-22':'2021-06-15'])
        df = df.reset_index()
        df = df.rename(columns={"date":"ds","cases_per_day":"y"})

        df_s = df.dropna()
        df_s_dt = df_s.set_index('ds')
        
        model_hw = ExponentialSmoothing(df_s_dt['y'][-prediction_w:], seasonal_periods=7, trend='add', seasonal='add')
        model_hw_fit = model_hw.fit()

        print(model_hw_fit.summary())
        model_hw_pred = model_hw_fit.forecast((7))


        model_hw_pred = pd.DataFrame(model_hw_pred)
        model_hw_pred.rename(columns={0:'yhat'}, inplace=True)


        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_s_dt[-prediction_w:].index, y=df_s_dt[-prediction_w:]['y'],
                                 mode='lines+markers',
                                 name='Cas/jour'))

        fig.add_trace(go.Scatter(x=model_hw_pred.index, y=model_hw_pred.yhat,
                                 mode='lines+markers',
                                 name='Cas/jour HW'))


        fig.add_trace(go.Scatter(x=df_covid['cases_per_day']['2021-06-16':'2021-06-23'].index,
                                 y=df_covid['cases_per_day']['2021-06-16':'2021-06-23'],
                                 mode='lines+markers',
                                 name='Cas/jour'))

        fig.update_layout(
            title_text="Nouveaux cas en Brésil avec HW.")

        fig.show()

        mape_hw= MAPE(df_covid['2021-06-16':'2021-06-22']['cases_per_day'], model_hw_pred.yhat)
        r2_hw= r2_score(df_covid['2021-06-16':'2021-06-22']['cases_per_day'],model_hw_pred.yhat)
        rmse_hw = mean_squared_error(df_covid['2021-06-16':'2021-06-22']['cases_per_day'], model_hw_pred.yhat, squared=False)
        mae_hw = mean_absolute_error(df_covid['2021-06-16':'2021-06-22']['cases_per_day'],model_hw_pred.yhat)

        print('HW RMSE: ', rmse_hw.round(2))
        print('HW MAPE ', mape_hw.round(2))
        print('HW R2: ', r2_hw.round(2))
        print('HW MAE: ', mae_hw.round(2))
    else: 
        df = pd.DataFrame(df_covid['cases_per_day']['2020-01-22':'2021-06-15'].rolling(window = 7, center = True).mean().round())
        df = df.reset_index()
        df = df.rename(columns={"date":"ds","cases_per_day":"y"})

        df_s = df.dropna()
        df_s_dt = df_s.set_index('ds')
        
        model_hw = ExponentialSmoothing(df_s_dt['y'][-prediction_w:], seasonal_periods=7, trend='add', seasonal='add')
        model_hw_fit = model_hw.fit()

        print(model_hw_fit.summary())
        model_hw_pred = model_hw_fit.forecast((7))


        model_hw_pred = pd.DataFrame(model_hw_pred)
        model_hw_pred.rename(columns={0:'yhat'}, inplace=True)
        model_hw_pred['yhat'].loc[model_hw_pred['yhat']<0]=0

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_s_dt[-prediction_w:].index, y=df_s_dt[-prediction_w:]['y'],
                                 mode='lines+markers',
                                 name='Cas/jour'))

        fig.add_trace(go.Scatter(x=model_hw_pred.index, y=model_hw_pred.yhat,
                                 mode='lines+markers',
                                 name='Cas/jour HW'))


        fig.add_trace(go.Scatter(x=df_covid['cases_per_day']['2021-06-10':'2021-06-23'].index,
                                 y=df_covid['cases_per_day']['2021-06-10':'2021-06-23'].rolling(window = 7, center = True).mean().round(),
                                 mode='lines+markers',
                                 name='Cas/jour '))

        fig.update_layout(
            title_text="Nouveaux cas avec HW.")

        fig.show()



        r = df_covid['cases_per_day']['2021-06-10':'2021-06-23'].rolling(window = 7, center = True).mean().round().dropna()

        mape_hw= MAPE(r, model_hw_pred.yhat)
        r2_hw= r2_score(r,model_hw_pred.yhat)
        rmse_hw = mean_squared_error(r, model_hw_pred.yhat, squared=False)
        mae_hw = mean_absolute_error(r,model_hw_pred.yhat)

        print('HW RMSE: ', rmse_hw.round(2))
        print('HW MAPE ', mape_hw.round(2))
        print('HW R2: ', r2_hw.round(2))
        print('HW MAE: ', mae_hw.round(2))


# In[142]:


HOLT_WINTERS(df_covid_bra, True)


# > On a testé ici des prédictions pour le Brésil avec des données lissées. Le resultat reste en dessous de ce qu'on peut avoir avec des données non lissées pour nos prédictions.

# ## Conclusion

# > Nous avons vu tout au long de ce projet l'impact de la Covid 19 sur le monde entre les premiers cas début 2020 jusqu'à juin 2021. Chaque pays ne gère pas la crise de la même facon, et certains sont plus ou moins séverement touchés. On a vu que le Brésil à un taux de mortalité bien plus important que la France pour un nombre de cas par million d'habitants presque équivalent. Le Japon quant à lui, s'en sort mieux avec très peu de cas confirmés. 
# Pour ce qui est des prédictions, nous avons prédit qu'elle sera le nombre de cas quotidiens dans ces trois pays lors des 7 jours suivants. Le modèle SARIMA et Prophet on des capacités de prédiction assez proche, contrairement à Holt Winter qui varie beaucoup plus. On voit aussi que ces modèles s'adaptent bien pour des données dont la saisonnalité est bien marquée. En effet lorsqu'on prend la cas de la France, les données sont peu homogènes et on retrouve beaucoup de valeurs extrèmes ou des larges écarts entre les valeurs, nos modèles ne sont alors plus performants. On arrive a corriger cette performnce en effectuant les prédictions avec les données lissées sur 7 jours par la moyenne mobile. 
# 

# In[ ]:





# In[ ]:




