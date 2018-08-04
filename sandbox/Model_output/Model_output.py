
from __future__ import print_function
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import googlemaps  # pip install -U googlemaps
from geopy.distance import geodesic
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd

#sklearn libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

def get_user_scores():
    
    # Setup the Sheets API
    SCOPES = 'https://www.googleapis.com/auth/spreadsheets.readonly'
    store = file.Storage('credentials.json')
    creds = store.get()
    if not creds or creds.invalid:
        flags = tools.argparser.parse_args('--auth_host_name localhost --logging_level INFO'.split())
        flow = client.flow_from_clientsecrets('client_secret.json', SCOPES)
        creds = tools.run_flow(flow, store, flags)
    service = build('sheets', 'v4', http=creds.authorize(Http()))
    
    # Call the Sheets API
    SPREADSHEET_ID = '1jG05kbXMDiw6N-Ic5xuAl1MdR6NS3VwDYxGAKMoQ_FI'
    RANGE_NAME = 'Form Responses 1!A1:X'
    result = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID,
                                                 range=RANGE_NAME).execute()
    
    # Get values
    values = result.get('values', [])
    users_df = pd.DataFrame(values)
    users_df.columns = ['Timestamp','username','SQFTmain', 'lot_area', 'EffectiveYearBuilt', \
                     'closest_school_rating', 'HH_Kids', 'Owner', 'crime', 'geography', \
                     'walkscore', 'transit_score',  'closest_schools', 'groceries', 'parks', \
                     'House', 'Neighborhood', 'Location', 'Max_Price', 'Min_Bed', 'Max_Bed', 'Type', 'Address', 'Max_Distance']

    # Get user input and standardize
    user_input = users_df.iloc[-1,:].values
    username = user_input[1]
    user_numbers = list(map(int, user_input[2:18]))
    filter_answers = user_input[18:]
    preferences = np.array(user_numbers[-3:])/sum(user_numbers[-3:])
    user_scores = list(np.array(user_numbers[0:3])/sum(user_numbers[0:3]) * 100 * preferences[0]) + \
                        list(np.array(user_numbers[3:8])/sum(user_numbers[3:8]) * 100 * preferences[1]) + \
                        list(np.array(user_numbers[8:13])/sum(user_numbers[8:13]) * 100 * preferences[2])
    
    return [username, user_scores, filter_answers]

def rf_score(price_wt, df, personal_wt):
    rf_score = price_wt * df['TotalValue']
    df_feature_score = pd.read_csv('df_feature_score.csv')
    for feature in personal_wt.keys():
        rf_score += df_feature_score[feature] * personal_wt[feature]
    return rf_score

def gmap_coordinates(address):
    # Google Map API
    gmaps = googlemaps.Client(key='AIzaSyCvdQFsQNisoXrltPkiVgiVKQEu-EZaoog')
    geocode_result = gmaps.geocode(address)
    property_address = geocode_result[0]['formatted_address']
    coordinates = (geocode_result[0]['geometry']['location']['lat'], geocode_result[0]['geometry']['location']['lng'])
    return coordinates

def personal_listings_rf(price_wt, personal_scores, filter_answers):
    personal_dict = {}
    personal_wt = {}
    
    # read original dataset and standardized dataset
    df_transform = pd.read_csv('df_transform_July2016.csv')
    df = pd.read_csv('df_integrated.csv', sep=';')
    
    # filter dataset
    price, min_bed, max_bed, house_type, address, max_distance = filter_answers
    df = df[df['TotalValue'] <= float(price)*1000]
    if min_bed != '':
        df = df[df['Bedrooms'] >= int(min_bed)]
    if max_bed != '':
        df = df[df['Bedrooms'] <= int(max_bed)]
    if house_type != None:
        text = house_type.split(", ")
        house_types = [t.replace('House', 'house').replace('Condo', 'condo').replace('PUD (planned unit development)', 'pud') for t in text]
        df = df[df['house_type'].isin(house_types)]
    if max_distance != None:
        max_distance = float(max_distance.split(" ")[1])
    else:
        max_distance = 50.0
    if address != None:
        place_coordinates = gmap_coordinates(address)
#             df = df[(abs(df['CENTER_LAT'] - waylat) <= (distance/69.0)) & (abs(df['CENTER_LON'] - waylon) <= (distance/69.0))]
        for i in df.index:
            df.loc[i, 'distance'] = geodesic((df.loc[i, 'CENTER_LAT'], df.loc[i, 'CENTER_LON']), place_coordinates).miles
        if 'distance' in df.columns:
            df = df[df['distance'] <= max_distance]
        
    df_transform = df_transform.iloc[df.index, :]
    df_features = ['zip_rank', 'SQFTmain', 'Units', 'Bedrooms', 'EffectiveYearBuilt', \
                    'house', 'condo', 'pud', 'pool', 'HH_Kids', 'Owner', 'lot_area', \
                    'num_school_choices', 'closest_school_rating', 'geography', 'parks', 'groceries', 'walkscore', 'transit_score']
    personal_features = ['SQFTmain', 'lot_area', 'EffectiveYearBuilt', \
                     'closest_school_rating', 'HH_Kids', 'Owner', 'crime', 'geography', \
                     'walkscore', 'transit_score',  'closest_schools', 'groceries', 'parks']

    for i in range(len(personal_scores)):
        personal_dict[personal_features[i]] = personal_scores[i]
    for f in df_features:
        if f in personal_dict.keys():
            personal_wt[f] = personal_dict[f] / 100
        else:
            personal_wt[f] = 0
            
    rf_scores = rf_score(price_wt, df_transform, personal_wt)

    for i in tqdm(df.index):
        df.loc[i, 'rf_scores'] = rf_scores[i]
        
    return df.sort_values(by=['rf_scores'], ascending=False)

def get_top10():
    price_wt = 6.7 # best top accuracy
    username, user_scores, filter_answers = get_user_scores()
    personal_df = personal_listings_rf(price_wt, user_scores, filter_answers)
    top10 = personal_df['Post_ID'].astype(str)[:10].values
    return (username, top10)

username, top10 = get_top10()
# send PUT request
url = 'http://127.0.0.1:5000/user/' + username + '?rank_01=' + top10[0] +'&rank_02=' + top10[1] +\
        '&rank_03=' + top10[2] + '&rank_04=' + top10[3] + '&rank_05=' + top10[4] +\
        '&rank_06=' + top10[5] + '&rank_07=' + top10[6] + '&rank_08=' + top10[7] +\
        '&rank_09=' + top10[8] + '&rank_10=' + top10[9]
        
r = requests.put(url)

print (username, top10)