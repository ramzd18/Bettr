"""""
import pandas as pd
##cols=["R_fighter","B_fighter","R_odds","B_odds","date","location","country","Winner","title_bout","weight_class","gender"]
ufcDF=pd.read_csv("C:/Users/rbped_w7mjkip/Downloads/archive/ufc.csv")
ufcDF["R_odds"]=ufcDF["R_odds"].astype(int)
ufcDF["B_odds"]=ufcDF["B_odds"].astype(int)
ufcDF["title_bout"] = (ufcDF["title_bout"] == "True").astype(int)
ufcDF["gender"] = (ufcDF["gender"] == "MALE").astype(int)
ufcDF["Winner"] = (((ufcDF["Winner"]=="Red") & (ufcDF["R_odds"] >0))| ((ufcDF["Winner"]=="Blue") & (ufcDF["B_odds"] >0))).astype(int)
#########

from geopy.geocoders import ArcGIS
from geopy import distance
geolocator = ArcGIS(scheme="https")
ufcDF=ufcDF.head(1000)
ufcDF = ufcDF.dropna(axis=0, subset=['date'])
ufcDF["Spread"]=ufcDF["R_odds"]-ufcDF["B_odds"]
##ufcDF['distance']=distance.geodesic((geolocator.geocode(ufcDF["country"]).latitude, geolocator.geocode(ufcDF["country"]).llngitude),(38.9072,77.0369))
#ufcDF["minimizedlocation"]=ufcDF["location"].str
##ufcDF["latitude"]=geolocator.geocode(ufcDF["location"]).latitude
ufcDF["actuallocation"] = ufcDF.apply(lambda row: geolocator.geocode(row["location"]), axis=1)
print(ufcDF)
#########

ufcDF["actuallocation"] = ufcDF.apply(lambda row: geolocator.geocode(row["location"]), axis=1)

##df.to_csv(r'C:\Users\Admin\Desktop\file3.csv')
#ufcDF.to_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp\ufcDF.csv')
ufcnewDF=pd.read_csv("C:/Users/rbped_w7mjkip/OneDrive/Desktop/mlwebapp/ufcDF.csv")
ufcDF['latitude']=ufcDF.apply(lambda row: row["actuallocation"].latitude,axis=1)
ufcDF['longitude']=ufcDF.apply(lambda row: row["actuallocation"].longitude,axis=1)
ufcnewDF['latitude']=ufcDF['latitude']
ufcnewDF['longitude']=ufcDF['latitude']
ufcnewDF["Distance"]=ufcnewDF.apply(lambda row: distance.geodesic((row["latitude"], row["longitude"]),(38.9072,77.0369)),axis=1)
#ufcnewDF.to_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp\updatedufcDF.csv')

###############

ufcnew1DF=pd.read_csv("C:/Users/rbped_w7mjkip/OneDrive/Desktop/mlwebapp/updatedufcDF.csv")
ufcnew1DF['date']=pd.to_datetime(ufcnew1DF['date'])
##ufcnew1DF.to_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp\dateupdatedufcDF.csv')

########

import pandas as pd
first100reddf=pd.read_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp\100bluesocialufcDF.csv')
first200reddf=pd.read_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp\200bluesocialufcDF.csv')
first300reddf=pd.read_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp\300bluesocialufcDF.csv')
first400reddf=pd.read_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp\400bluesocialufcDF.csv')
first500reddf=pd.read_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp\500bluesocialufcDF.csv')
first600reddf=pd.read_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp\600bluesocialufcDF.csv')
first700reddf=pd.read_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp\700bluesocialufcDF.csv')
first800reddf=pd.read_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp\800bluesocialufcDF.csv')
first900reddf=pd.read_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp\900bluesocialufcDF.csv')
first1000reddf=pd.read_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp\1000bluesocialufcDF.csv')
redDF = pd.concat([first100reddf, first200reddf,first300reddf,first400reddf,first500reddf,first600reddf,first700reddf,first800reddf,first900reddf,first1000reddf], axis=0)
bluecolumnDF=redDF["B_fightersocial"]
#######
#df['favorite']=df.apply(lambda row: row['R_fighter'] if row['R_odds']>0 else row['B_fighter'])

df = pd.read_csv("C:/Users/rbped_w7mjkip/OneDrive/Desktop/mlwebapp/finalcompleteufcDF.csv")
#df=df[["R_odds", "B_odds", "Spread", "Distance", "R_fightersocial", "B_fightersocial","Winner"]]
df['favorite']=df.apply((lambda row: row['R_fighter'] if row['R_odds']>0 else row['B_fighter']),axis=1)
df['favorite_odds']=df.apply((lambda row: row['R_odds'] if row['R_odds']>0 else row['B_odds']),axis=1)
df['underdog']=df.apply((lambda row: row['R_fighter'] if row['R_odds']<=0 else row['B_fighter']),axis=1)
df['underdog_odds']=df.apply((lambda row: row['R_odds'] if row['R_odds']<=0 else row['B_odds']),axis=1)
df['favoritesocial']=df.apply((lambda row: row['R_fightersocial'] if row['R_odds']>0 else row['B_fightersocial']),axis=1)
df['underdogsocial']=df.apply((lambda row: row['R_fightersocial'] if row['R_odds']<=0 else row['B_fightersocial']),axis=1)
df=df[["favorite", "underdog", "favorite_odds", "underdog_odds", "favoritesocial", "underdogsocial","Distance","Winner"]]
df.to_csv(r"C:/Users/rbped_w7mjkip/OneDrive/Desktop/mlwebapp/finalufcDF.csv")
#print(df)
#######
import tensorflow as tf
import re
def scale_dataset(dataframe, oversample=False):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X, y)

  data = np.hstack((X, np.reshape(y, (-1, 1))))

  return data, X, y
def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
  nn_model = tf.keras.Sequential([
      tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(7,)),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(num_nodes, activation='relu'),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                  metrics=['accuracy'])
  history = nn_model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
  )

  return nn_model, history
def convert_to_km(distance):
   return distance.split(" ", 1)[0]
"""