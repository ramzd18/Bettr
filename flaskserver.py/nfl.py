"""""
import pandas as pd
nfldf=pd.read_csv("C:/Users/rbped_w7mjkip/OneDrive/Desktop/mlwebapp-nfl/nfl.csv")
#nfldf=nfldf.sample(frac=1)
nfldf=nfldf.head(700)
nfldf['Line Differential']=(nfldf['Home Line Close']-nfldf['Away Line Close']).abs()
nfldf=nfldf[["Home Team","Away Team","Home Line Close","Away Line Close","Line Differential","Total Score Close"]]
print(nfldf)
nfldf.to_csv(r"C:/Users/rbped_w7mjkip/OneDrive/Desktop/mlwebapp-nfl/nflcolumns.csv")
######
from geopy.geocoders import ArcGIS
from geopy import distance
import pandas as pd
geolocator = ArcGIS(scheme="https")
print(geolocator.geocode("Los Angeles").latitude)
#text.rsplit(' ', 1)[0]
nfldf=pd.read_csv("C:/Users/rbped_w7mjkip/OneDrive/Desktop/mlwebapp-nfl/nflcolumns.csv")
nfldf['Home latitude'] = nfldf.apply(lambda row: geolocator.geocode(row["Home Team"].rsplit(' ', 1)[0]).latitude, axis=1)

nfldf['Home longitude'] = nfldf.apply(lambda row: geolocator.geocode(row["Home Team"].rsplit(' ', 1)[0]).longitude, axis=1)
nfldf['Away latitude'] = nfldf.apply(lambda row: geolocator.geocode(row["Away Team"].rsplit(' ', 1)[0]).latitude, axis=1)
nfldf['Away longitude'] = nfldf.apply(lambda row: geolocator.geocode(row["Away Team"].rsplit(' ', 1)[0]).longitude, axis=1)
nfldf["Distance"]=nfldf.apply(lambda row: distance.geodesic((row["Home latitude"], row["Home longitude"]),(row["Away latitude"], row["Away longitude"])),axis=1)

########
df1 = nfldf.iloc[:350]
df2 = nfldf.iloc[350:]
nfldf['Home latitdude1']= df1.apply(lambda row: geolocator.geocode(row["Home Team"].rsplit(' ', 1)[0]).latitude, axis=1)
nfldf['Home latitude2'] = df2.apply(lambda row: geolocator.geocode(row["Home Team"].rsplit(' ', 1)[0]).latitude, axis=1)
#nfldf['Home longitude'] = nfldf.apply(lambda row: geolocator.geocode(row["Home Team"].rsplit(' ', 1)[0]).longitude, axis=1)
#nfldf['Away latitude'] = nfldf.apply(lambda row: geolocator.geocode(row["Away Team"].rsplit(' ', 1)[0]).latitude, axis=1)
#nfldf['Away longitude'] = nfldf.apply(lambda row: geolocator.geocode(row["Away Team"].rsplit(' ', 1)[0]).longitude, axis=1)
nfldf['Home latitude2'] = df2.apply(lambda row: geolocator.geocode(row["Home Team"].rsplit(' ', 1)[0]).latitude, axis=1)
#nfldf['Home longitude'] = nfldf.apply(lambda row: geolocator.geocode(row["Home Team"].rsplit(' ', 1)[0]).longitude, axis=1)
#nfldf['Away latitude1'] = df1.apply(lambda row: geolocator.geocode(row["Away Team"].rsplit(' ', 1)[0]).latitude, axis=1)
#nfldf['Away latitude2'] = df2.apply(lambda row: geolocator.geocode(row["Away Team"].rsplit(' ', 1)[0]).latitude, axis=1)
#nfldf['Away longitude1'] = df1.apply(lambda row: geolocator.geocode(row["Away Team"].rsplit(' ', 1)[0]).longitude, axis=1)
nfldf['Away longitude2'] = df1.apply(lambda row: geolocator.geocode(row["Away Team"].rsplit(' ', 1)[0]).longitude, axis=1
nfldf.to_csv(r"C:/Users/rbped_w7mjkip/OneDrive/Desktop/mlwebapp-nfl/nfllatlong.csv")
print(nfldf)
##########
print(df)
df3=pd.concat([df['Away longitude1'],df['Away longitude2']])
df3=df3.dropna(axis=0)
df3=df3.to_frame(name="Away longitude")
print(df3)
df4=pd.concat([df['Away latitude1'],df['Away latitude2']])
df4=df4.dropna(axis=0)
df4=df4.to_frame(name="Away latitude")
print(df4)
df5=pd.concat([df['Home latitdude1'],df['Home latitude2']])
df5=df5.dropna(axis=0)
df5=df5.to_frame(name="Home latitude")

df['Away longitude']=df3
df['Away latitude']=df4
df['Home latitude']=df5
df.to_csv(r"C:/Users/rbped_w7mjkip/OneDrive/Desktop/mlwebapp-nfl/updatednfllatlong.csv")
####### df=pd.read_csv("C:/Users/rbped_w7mjkip/OneDrive/Desktop/mlwebapp-nfl/fullnfllatlong.csv")
df3=pd.concat([df['Away longitude1'],df['Away longitude2']])
df3=df3.dropna(axis=0)
df3=df3.to_frame(name="Away longitude")
df4=pd.concat([df['Away latitude1'],df['Away latitude2']])
df4=df4.dropna(axis=0)
df4=df4.to_frame(name="Away latitude")
df5=pd.concat([df['Home latitdude1'],df['Home latitude2']])
df5=df5.dropna(axis=0)
df5=df5.to_frame(name="Home latitude")
df6=pd.concat([df['Home longitude2'],df['Home longitude1']])
df6=df6.dropna(axis=0)
df6=df6.to_frame(name="Home longitude")
df['Away longitude']=df3
df['Away latitude']=df4
df['Home latitude']=df5
df['Home longitude']=df6
df["Distance"]=df.apply(lambda row: distance.geodesic((row["Home latitude"], row["Home longitude"]),(row["Away latitude"], row["Away longitude"])),axis=1)
#########def socialmediaData(fighter1):
    # Authentication Keys
    consumerKey = "7xzXFbwLqoiDO7lyJgNxGwdFG"   # enter your own credentials
    consumerSecret = "JiC0zq4wCiXLXzXFI93ijZNRcCkQamQZhN3GoZZOb790fT9kdB"  # enter your own credentials

    accessToken = "1278718386143399936-3fZE59ZIuHgG23XUVajG3IHkCCvHpD"    # enter your own credentials
    accessTokenSecret = "0cv5oxL8XzfoPu90q8Lc36IApYEqr5KIKbZoZi8lw1LOw"    # enter your own credentials
    auth = tweepy.OAuthHandler(consumerKey, consumerSecret) 
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)
    
    tweetnum = 25
    tweets=api.search_tweets(q=fighter1,count=tweetnum)
   
    tweet_list = []
    for tweet in tweets:
        tweet_list.append(tweet.text)  # transfer all text into variable tweet list
   
    totalnumber=tweetnum
    
    df = pd.DataFrame(tweet_list, columns =['initial text'])
    df['cleaned text']= df['initial text']
   # print(df['initial text'])
    #Removing RT, Punctuation etc
    clean_tweets = []
    for tweet in df['cleaned text']:
        tweet = re.sub("RT @[A-Za-z0-9]+","",tweet) #Remove @ sign
        ##Here's where all the cleaning takes place
        clean_tweets.append(tweet)
    df['cleaned text'] = clean_tweets
    df.drop_duplicates(subset ="cleaned text",
                         keep = False, inplace = True)

    #Specify location
    #Calculating Negative, Positive, Neutral and Compound values
    if(len(df['cleaned text'])==0):
        return 0

    cols=["neg","neu","pos","compound"]
    sentimentdf=pd.DataFrame(columns=cols)
    sia= SentimentIntensityAnalyzer()
    sentimentdf["neg"]=  df.apply(lambda row: (sia.polarity_scores(row["cleaned text"])['neg']*100), axis=1)
    sentimentdf["pos"]=  df.apply(lambda row: (sia.polarity_scores(row["cleaned text"])['pos']*100), axis=1)

  
    
    totaltot =(df["cleaned text"].count())
  
    negneg= sentimentdf['neg'].sum()
  
    pospos=sentimentdf['pos'].sum()
    
    totalpol= pospos-negneg
    return totalpol
 ################
 import pandas as pd
df=pd.read_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp-nfl\fullnfldataset.csv')
df['favorite']=df.apply((lambda row: row['Home Team'] if row['Home Line Close']<=0 else row['Away Team']),axis=1)
df['favorite_odds']=df.apply((lambda row: row['Home Line Close'] if row['Home Line Close']<=0 else row['Away Line Close']),axis=1)
df['underdog']=df.apply((lambda row: row['Home Team'] if row['Home Line Close']>0 else row['Away Team']),axis=1)
df['underdog_odds']=df.apply((lambda row: row['Home Line Close'] if row['Home Line Close']>0 else row['Away Line Close']),axis=1)
df['favoritesocial']=df.apply((lambda row: row['homesocial'] if row['Home Line Close']<=0 else row['awaysocial']),axis=1)
df['underdogsocial']=df.apply((lambda row: row['homesocial'] if row['Home Line Close']>0 else row['awaysocial']),axis=1)
df['favortiedistance']=df.apply((lambda row: 0 if row['Home Line Close']<=0 else row['Distance']),axis=1)
df['underdogdistance']=df.apply((lambda row: 0 if row['Home Line Close']>0 else row['Distance']),axis=1)
df=df[["favorite", "underdog", "favorite_odds", "underdog_odds", "favoritesocial", "underdogsocial","favortiedistance","underdogdistance"]]
df.to_csv(r'C:\Users\rbped_w7mjkip\OneDrive\Desktop\mlwebapp-nfl\nflmldataset.csv')
#############   
 import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
for label in cols[:-1]:
  plt.hist(df[df["Winner"]==1][label], color='blue', label='favorite', alpha=0.7, density=True)
  plt.hist(df[df["Winner"]==0][label], color='red', label='underdog', alpha=0.7, density=True)
  plt.title(label)
  plt.ylabel("Probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()                                    

  #####
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