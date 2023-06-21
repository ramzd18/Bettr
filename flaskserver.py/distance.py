from geopy.geocoders import ArcGIS
from geopy import distance
def distanceBetween(home,away):
    geolocator = ArcGIS(scheme="https")
    geol= 2
    home_lat= geolocator.geocode(home).longitude
    home_lang=geolocator.geocode(home).latitude
    away_lat= geolocator.geocode(away).longitude
    away_lang=geolocator.geocode(away).latitude
    return distance.geodesic((home_lat,home_lang),(away_lat,away_lang))
def distanceUS(name):
   name_lat= geolocator.geocode(name).longitude
   name_long=geolocator.geocode(name).latitude
   return distance.geodesic((name_lat,name_long),(38.9072,77.0369))