from flask import Flask
import distance
import socialmediadata

app = Flask(__name__)

@app.route('/ufcmodel')
def ufc_model(fighter1, fighter2, location):
    distanceto=distance.distanceUS(location)
    fighter1social= socialmediadata.socialmediaData(fighter1)
    fighter2social=socialmediadata.socialmediaData(fighter2)
    