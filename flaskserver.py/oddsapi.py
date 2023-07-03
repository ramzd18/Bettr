import requests
import json

def ufcOdds(fighter1, fighter2):
    api_key='cb30a81c314d379e619131aa94c365e7'
    SPORT='mma_mixed_martial_arts'
    regions='us'
    markets='h2h'
    odds_format='decimal'
    date_format='iso'
    odds_response = requests.get(f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds', params={
        'api_key': api_key,
        'regions': regions,
        'markets': markets,
        'oddsFormat': odds_format,
        'dateFormat': date_format,
    })
    response_info =(odds_response.json())
    list=[]
    for x in range(len(response_info)-1):
        dict=response_info[x]
        home=dict.get('home_team')
        print(home)
        away= dict.get('away_team')
        print(away)
        if((home.lower()==fighter1.lower() and away.lower()==fighter2.lower()) or(away.lower()==fighter1.lower() and home.lower()==fighter2.lower())):
            dictbook=dict.get('bookmakers')
            print("reached here")
            dictbookoutcomes=dictbook[0].get('markets')
            dictbookoutcomes=dictbookoutcomes[0].get('outcomes')
            print(dictbookoutcomes[0].get('name'))
            list.append(dictbookoutcomes[0].get('name'))
            list.append(dictbookoutcomes[0].get('price'))
            list.append(dictbookoutcomes[1].get('name'))
            list.append(dictbookoutcomes[1].get('price'))
            break; 
    if(len(list)<3):
        list.append("error")

def fixOdds(list1:list ):
    if(list[0]=="error"):
        return
    if(list[1]>list[3]):
        list[1]=(list[1]-1)/400
        list[3]=(100/(list[3]-1))*(-1)
    else:
        list[3]=(list[3]-1)/400  
        list[1]=(100/(list[1]-1))*(-1)
    return list    

def nflOdds(team1, team2):
    api_key='cb30a81c314d379e619131aa94c365e7'
    SPORT='americanfootball_nfl'
    regions='us'
    markets='spreads,totals'
    odds_format='decimal'
    date_format='iso'
    odds_response = requests.get(f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds', params={
            'api_key': api_key,
            'regions': regions,
            'markets': markets,
            'oddsFormat': odds_format,
            'dateFormat': date_format,
        })
    response_info =(odds_response.json())
    list=[]
    for x in range(len(response_info)-1):
        dict=response_info[x]
        home=dict.get('home_team')
        print(home)
        away= dict.get('away_team')
        print(away)
        if((home.lower()==team1.lower() and away.lower()==team2.lower()) or(away.lower()==team1.lower() and home.lower()==team2.lower())):
            dictbook=dict.get('bookmakers')
            print("reached here")
            dictbookoutcomes=dictbook[0].get('markets')
            print(dictbookoutcomes[0])
            dictbookoutcomes1=dictbookoutcomes[0].get('outcomes')
            print(dictbookoutcomes1)
            print(dictbookoutcomes1[0].get('name'))
            list.append(dictbookoutcomes1[0].get('name'))
            list.append(dictbookoutcomes1[0].get('point'))
            list.append(dictbookoutcomes1[1].get('name'))
            list.append(dictbookoutcomes1[1].get('point'))
            if(len(dictbookoutcomes)>1):
                dictbookoutcomes2=dictbookoutcomes[1].get('outcomes')
                print(dictbookoutcomes2[0].get('point'))
                list.append(dictbookoutcomes2[0].get('point'))
            else:
                list.append("error")    
            break; 
    if(len(list)<3):
        list.append("error")
    print(list)
    

nflOdds("Los Angeles Chargers","Detroit Lons")        
