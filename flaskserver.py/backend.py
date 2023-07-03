from flask import Flask
import distance
import socialmediadata
import oddsapi
import pickle
app = Flask(__name__)

@app.route('/ufcmodel')
def ufc_model(fighter1, fighter2, location):
    distanceto=distance.distanceUS(location)
    fighter1social= socialmediadata.socialmediaData(fighter1)
    fighter2social=socialmediadata.socialmediaData(fighter2)
    ##list= oddsapi.ufcOdds(fighter1,fighter2)
    if(list[0]=="error"):
        return "error"
    if(list[0].equals(fighter1)):
        fighter1odds=list[1]
        fighter2odds=list[3]
    else:
        fighter1odds=list[3]
        fighter2odds=list[1]
    model = pickle.load(open("C:/Users/rbped_w7mjkip/OneDrive/Desktop/mlwebapp-ufc/neuralnetworkmodel.pkl",'rb'))
    print(model.predict([[135,-145,-48.2,73.3,11336]]))

def nfl_model(team1,team2):
    team1social=socialmediadata.socialmediaData(team1)    
    team2social=socialmediadata.socialmediaData(team2)  
ufc_model("d","df","dfd")  