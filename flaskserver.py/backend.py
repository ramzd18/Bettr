from flask import Flask
import distance
import socialmediadata
import oddsapi
app = Flask(__name__)

@app.route('/ufcmodel')
def ufc_model(fighter1, fighter2, location):
    distanceto=distance.distanceUS(location)
    fighter1social= socialmediadata.socialmediaData(fighter1)
    fighter2social=socialmediadata.socialmediaData(fighter2)
    list= oddsapi.ufcOdds(fighter1,fighter2)
    if(list[0]=="error"):
        return "error"
    if(list[0].equals(fighter1)):
        fighter1odds=list[1]
        fighter2odds=list[3]
    else:
        fighter1odds=list[3]
        fighter2odds=list[1]

def nfl_model(team1,team2):
    team1social=socialmediadata.socialmediaData(team1)    
    team2social=socialmediadata.socialmediaData(team2)  
  