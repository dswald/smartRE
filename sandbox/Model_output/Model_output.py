
from __future__ import print_function
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
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
    RANGE_NAME = 'Form Responses 1!A1:R'
    result = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID,
                                                 range=RANGE_NAME).execute()
    
    # Get values
    values = result.get('values', [])
    users_df = pd.DataFrame(values)
    users_df.columns = ['Timestamp','username','SQFTmain', 'lot_area', 'EffectiveYearBuilt', \
                     'closest_school_rating', 'HH_Kids', 'Owner', 'crime', 'geography', \
                     'walkscore', 'transit_score',  'closest_schools', 'groceries', 'parks', \
                     'House', 'Neighborhood', 'Location']

    # Get user input and standardize
    user_input = users_df.iloc[-1,:].values
    username = user_input[1]
    user_numbers = list(map(int, user_input[2:]))
    preferences = np.array(user_numbers[-3:])/sum(user_numbers[-3:])
    user_scores = list(np.array(user_numbers[0:3])/sum(user_numbers[0:3]) * 100 * preferences[0]) + \
                        list(np.array(user_numbers[3:8])/sum(user_numbers[3:8]) * 100 * preferences[1]) + \
                        list(np.array(user_numbers[8:13])/sum(user_numbers[8:13]) * 100 * preferences[2])
    
    return (username, user_scores)

def rf_feature_score(df, feature):
    rf = RandomForestRegressor()
    X = df.loc[:, df.columns.difference(['TotalValue', 'rowID'])]
    Y = df['TotalValue']
    rf.fit(X, Y)
    feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X.columns,
                                    columns=['importance'])
    rf.fit(df[feature].values.reshape(-1, 1), df['TotalValue'])
    feature_coef = feature_importances.loc[feature]
    feature_score = rf.predict(df[feature].values.reshape(-1, 1)) * feature_coef.values[0]
    return feature_score

def rf_score(price_wt, df, personal_wt):
    rf_score = price_wt * df['TotalValue']
    for feature in personal_wt.keys():
        rf_score += rf_feature_score(df, feature) * personal_wt[feature]
    return rf_score

def personal_listings_rf(price_wt, personal_scores):
    personal_dict = {}
    personal_wt = {}
    
    df_transform = pd.read_csv('df_transform_July2016.csv')
    df = pd.read_csv('df_integrated.csv', sep=';')
    
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
    price_wt = 0.01 # to be tuned in model evaluation
    username, user_scores = get_user_scores()
    personal_df = personal_listings_rf(price_wt, user_scores)
    top10 = personal_df['Post_ID'].astype(str)[:10].values
    return (username, top10)

username, top10 = get_top10()
# send PUT request
url = 'http://amronline.net:9000/user/' + username + '?rank_01=' + top10[0] +'&rank_02=' + top10[1] +\
        '&rank_03=' + top10[2] + '&rank_04=' + top10[3] + '&rank_05=' + top10[4] +\
        '&rank_06=' + top10[5] + '&rank_07=' + top10[6] + '&rank_08=' + top10[7] +\
        '&rank_09=' + top10[8] + '&rank_10=' + top10[9]
        
r = requests.put(url)

print (username, top10)