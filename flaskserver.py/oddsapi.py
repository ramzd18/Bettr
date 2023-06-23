import requests
import json



def ufcOdds(fighter1, fighter2):
 api_key='cb30a81c314d379e619131aa94c365e7'
 sport='mma_mixed_martial_arts'
 regions='us'
 markets='h2h'
 odds_format='decimal'
 date_format='iso'
 odds_response = requests.get(f'https://api.the-odds-api.com/v4/sports/{sport}/odds', params={
    'api_key': api_key,
    'regions': regions,
    'markets': markets,
    'oddsFormat': odds_format,
    'dateFormat': date_format,
})
 response_info = json.loads(odds_response)
 fighter_list = []
 ##fighter_list.append([response_info[‘Country’], country_info[‘TotalConfirmed’]])

