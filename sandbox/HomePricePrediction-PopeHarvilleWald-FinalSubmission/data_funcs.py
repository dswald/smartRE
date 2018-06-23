# general libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import *
import pickle
import math
import csv
from itertools import izip

#sklearn libraries
from sklearn.feature_extraction.text import *
from sklearn.neural_network import *
from sklearn import tree
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

#sklearn regressors
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LogisticRegression

#sklearn classifiers
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier

# functions supporting mean normalization
def find_mu_std(df, feature):
    # finds mean and standard deviation of a given feature vector
    # this is run against training data and stored for use when conducting mean normalization
    # on dev and test
    mu = df[feature].mean()
    std = df[feature].std()
    return([mu, std])

def mean_norm_sub(x, mu, std):
    # simple function called in comprehension / lamba statement, that calculations
    # mean normalization for a given x in a feature vector
    if std == 0:
        return x
    else:
        return ((x-mu)/std)

def mean_normalization(df, features_to_normalize, mn_values):
    # pass in the dataframe with the features to normalize, the set of features to run mean normalization and
    # the set of mu and std values (previously calculated on training data)
    for feature in features_to_normalize:
        mu=mn_values[feature][0]
        std=mn_values[feature][1]
        newfeaturename = feature + '_MN'
        df[newfeaturename] = df[feature].apply(lambda x: mean_norm_sub(x, mu, std))
        df.drop(feature, axis=1, inplace=True)
    return df


# functions supporting feature scaling (not sure if we are using these)
def find_min_max(df, feature):
    # finds mean and standard deviation of a given feature vector
    # this is run against training data and stored for use when conducting mean normalization
    # on dev and test
    fs_min = df[feature].min()
    fs_max = df[feature].max()
    return([fs_min, fs_max])

def feature_scaling_sub(x, fmin, fmax):
    # simple function called in comprehension / lamba statement, that calculations
    # mean normalization for a given x in a feature vector
    return ((x-fmin)/(fmax-fmin))

def feature_scaling(df, cont_features, fs_values):
    # pass in the dataframe with the features to normalize, the set of features to run mean normalization and
    # the set of mu and std values (previously calculated on training data)
    for feature in cont_features:
        fmin=fs_values[feature][0]
        fmax=fs_values[feature][1]
        newfeaturename = 'FS' + feature
        df[newfeaturename] = df[feature].apply(lambda x: feature_scaling_sub(x, fmin, fmax))


# functions for turning categorical features into separate dummy variables
def get_train_cat_features(train_data, used_cat_features):
    '''
    Gets dictionary of categorical features with their categories from the training data. This is necessary because we
    need to ensure we have the same list of new features when cleaning dev and test data.
    '''
    feature_dict = {}
    for feature in used_cat_features:
        feature_dict[feature] = list(train_data[feature].unique())
    return feature_dict

def clean_categorical_features(df, feature_dict, normal_cat_features, used_cat_features):
    '''
    Takes a dataframe and cleans categorical features by splitting into dummy variables.
    Also removes unwanted features and dummy variables.
    '''
    ## Splitting features that don't need special attention
    for feature in normal_cat_features:
        # create a new column for each category of the feature
        for cat in feature_dict[feature]:
            new_feature_name = feature + '_' + str(cat)
            df[new_feature_name] = (df[feature] == cat).astype(int)

    ## Central air is already boolean, it just needs Y/N to be changed to 1/0
    df['CentralAir_bool'] = (df['CentralAir'] == 'Y').astype(int)

    ## Dealing with double features (e.g. Condition1 and Condition2)
    for cat in feature_dict['Condition1']:
        new_feature_name = 'Condition_' + str(cat)
        df[new_feature_name] = ((df['Condition1'] == cat) | (df['Condition2'] == cat)).astype(int)
    for cat in feature_dict['Exterior1st']:
        new_feature_name = 'Exterior_' + str(cat)
        df[new_feature_name] = ((df['Exterior1st'] == cat) | (df['Exterior2nd'] == cat)).astype(int)
    for cat in feature_dict['BsmtFinType1']:
        new_feature_name = 'BsmtFinType_' + str(cat)
        df[new_feature_name] = ((df['BsmtFinType1'] == cat) | (df['BsmtFinType2'] == cat)).astype(int)

    ## combining some feature categories
    # combining irregular lot shape types
    df['LotShape_IR'] = ((df['LotShape_IR1'] == 1) | (df['LotShape_IR2'] == 1) | (df['LotShape_IR3'] == 1)).astype(int)
    # combining FR2 and FR3 for frontage sides in LotConfig
    df['LotConfig_FR23'] = ((df['LotConfig_FR2'] == 1) | (df['LotConfig_FR3'] == 1)).astype(int)
    return df

# functions for dealing with likert scale features
def likert2cont(x):
    '''
    Likert conversion function to apply to all data in the DataFrame.
    Due to extremely tight filtering in the if/else conditions, no inadvertant
    conversions should exist in the output.

    Note Basement exposure uses different variables to the same effect.
    Consider removing basement exposure from the calculation
    '''
    if x == 'Ex': x = 5
    elif x == 'Gd': x = 4
    elif x == 'TA' or x == 'Av': x = 3 #or statement to handle basement exposure
    elif x == 'Fa' or x == 'Mn': x = 2 #or statement to handle basement exposure
    elif x == 'Po' or x == 'No': x = 1 #or statement to handle basement exposure
    elif x == 'NA': x = 0
    else: x = x #probably not necessary, but just covering my bases
    return x

def convert_likerts(df, likert_features):
    '''
    Creates a new feature scaled 0-5 for each feature currently represented as a likert scale.
    '''
    for feature in likert_features:
        new_feature_name = feature + '_lik'
        df[new_feature_name] = df[feature].apply(likert2cont)
    return df


# Combining feature transformations
def transform_features(df,feature_dict,normal_cat_features,used_cat_features,likert_features,drop_dummies,drop_features):
    '''
    Combines all feature transformation code to a single function.
    '''
    # transforming categorical features
    df = clean_categorical_features(df=df, feature_dict=feature_dict, normal_cat_features=normal_cat_features,
                                    used_cat_features=used_cat_features)

    # transforming likert scale features
    df = convert_likerts(df=df, likert_features=likert_features)

    ## dropping unwanted/uneeded features at this point, before mean normalizing
    # dropping original categorical features
    df.drop(used_cat_features, axis=1, inplace=True)
    # dropping new categorical dummy features that we don't want
    df.drop(drop_dummies, axis=1, inplace=True)
    # dropping original likert features
    df.drop(likert_features, axis=1, inplace=True)
    # dropping unwanted features
    df.drop(drop_features, axis=1, inplace=True)
    return df

# Evaluating model errors
def rmsle(y, y_pred):
    '''
    Calculates root mean square logarithmic error, the metric used in this Kaggle competition.
    '''
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

def get_errors(model, train_data, train_labels, dev_data, dev_labels):
    '''
    Uses given model and data/labels to test model predictions using a variety of metrics.
    Returns dictionary containing predictions and the resulting scores.
    '''
    predictions = model.predict(dev_data)
    train_predictions = model.predict(train_data)

    # Explained variance score: 1 is perfect prediction
    train_var_score = model.score(train_data, train_labels)
    print('Variance score, Training: %.2f' % train_var_score)

    var_score = model.score(dev_data, dev_labels)
    print('Variance score, Dev: %.2f' % var_score)

    train_abs_mean_error = np.mean(np.abs((train_predictions - train_labels)))
    print("Abs Mean Error - Train: %.2f" % train_abs_mean_error)

    #mean error
    abs_mean_error = np.mean(np.abs(predictions - dev_labels))
    print("Abs Mean Error: %.2f" % abs_mean_error)

    # The mean squared error
    mean_square_error = np.mean((predictions - dev_labels) ** 2)
    print("Mean squared error: %.2f" % mean_square_error)


    #kaggle measure of success - root mean squared Logarithmic error
    root_mean_square_log_error = rmsle(dev_labels, predictions)
    print("Root Mean Squared Logarithmic Error: %.2f" % root_mean_square_log_error)

    return {'dev_predictions':predictions, 'train_predictions':train_predictions,
            'train_var_score':train_var_score, 'dev_var_score':var_score,
            'train_abs_mean_error':train_abs_mean_error, 'dev_mean_square_error':mean_square_error,
            'dev_root_mean_square_log_error':root_mean_square_log_error}


# function finds non-zero weights
def find_nonzero(data, fnames, num_features):
    
    nz_features_idx = []
    nz_features = []
    num_classes = len(data)
    #loop through list of 26K+ tokens and check if weight is non-zero for any class
    # more efficient ways to write, but this was easy for me to understand
    for f in range(0, num_features):
        for c in range(0, num_classes):
            if data[c][f] <> 0:
                nz_features_idx.append(f)
                nz_features.append(fnames[f])       
        
    nz_features = list(set(nz_features)) 

    return (nz_features)



'''
function for splitting our price label series into a series of categories. 
used categories [0,1,2] instead of [low, med, high] since using numbers means we can easily change our number of thresholds
If you give it only 1 or 3 thresholds for example, it will return [0,1] or [0,1,2,3] for your categories. 
'''

def categorize_prices(labels, thresholds=[275665, 424100] ):
    '''
    Function takes a list of numeric labels and thresholds for categorizing them in to buckets.
    Iteritively identifies categories from lowest bucket to highest. This is probably inefficient,
    but hey it's easy and doesn't matter on vectors this small.
    '''
    cat = 0 # using integers for categories, since this can be adjusted dynamically
    cat_labels = pd.Series([cat]*len(labels)) # initializing category vector with all as loweset category
    for threshold in thresholds:
        cat += 1
        next_cat_indices = labels > threshold # identify all labels above threshold
        cat_labels[next_cat_indices] = cat
        
    return cat_labels

def run_logr_l1(train_data, train_cat_labels, dev_data, final_features):
    '''
    L1 regression
    Find low value add features (to them remove)
    return subsetted data set for training and dev
    '''
    # fit LR with L1 penalty - matching into training label categories (logistic regression)
    lr_l1 = LogisticRegression(penalty='l1')
    lr_l1.fit(train_data, train_cat_labels)

    #store coefficients
    coef_lr_l1 = lr_l1.coef_

    # have a weight for all 206 features for each category
    num_cats = len(coef_lr_l1)           
    num_features = len(coef_lr_l1[0])   #length for 1 class

    #find learned weights not-equal to zero
    nz_tokens = find_nonzero(coef_lr_l1, final_features, num_features)

    #build a new training_subset, with nonzero features
    hv_feature_model = SelectFromModel(lr_l1, prefit=True)
    train_subset = hv_feature_model.transform(train_data)
    dev_subset = hv_feature_model.transform(dev_data)
   
    return(train_subset, dev_subset, nz_tokens)

def run_pca(t_subset, d_subset, components=10):  
    '''
    feature reduction and colinearity removal via PCA
    returns new features for both train and subset
    '''
    pca = PCA(n_components=components)
    pca.fit(t_subset)
    
    # transform training and dev for testing results
    train_subset_pca = pca.transform(t_subset)
    dev_subset_pca = pca.transform(d_subset)
    
    return (train_subset_pca, dev_subset_pca)

#L2 logistics Regression to place  into categories
def run_logr_l2(train_data, train_labels, dev_data, c=1):
    '''
    L2 logistic regression to place into categories
    '''
    
    # run Logistic Regression with L2 penalty, on smaller feature set to place into categories
    lr_l2 = LogisticRegression(penalty='l2', C=c)
    lr_l2.fit(train_data, train_labels)

    #predict categories for train and dev
    train_categories = lr_l2.predict(train_data)
    dev_categories = lr_l2.predict(dev_data)
    
    return(train_categories, dev_categories)

def run_rfc(train_data, train_cat_labels, dev_data, estimators=1000):
    '''
    Random Forest Classifier to Place into Categories
    '''
    rfc = RandomForestClassifier(n_estimators=estimators)
    rfc.fit(train_data, train_cat_labels)
        
    #predict categories for train and dev
    train_categories = rfc.predict(train_data)
    dev_categories = rfc.predict(dev_data)
    
    return(train_categories, dev_categories)

# AdaBoost Classifier to place into Categories
def run_abc(train_data, train_labels, dev_data, estimators=1000, lrn_rate=0.1):
    '''
    AdaBoost Classifier to place into Categories
    '''
    abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=estimators, learning_rate=lrn_rate)
    abc.fit(train_data, train_labels)
    
    #predict categories for train and dev
    train_categories = abc.predict(train_data)
    dev_categories = abc.predict(dev_data)
    
    return(train_categories, dev_categories)

def run_lr(train_categories, dev_categories, train_data, train_labels, dev_data, dev_labels):
    '''
    for each category, pull training and dev prediction
    consolidate them into the lists below
    '''    
    dev_predictions = []
    dev_actuals = []
    
    for cat in list(unique(train_categories)):
        #get index of train and dev obs in this category, and separate data
        idx = list(np.array(np.where(train_categories == cat))[0])
        idx_dev = list(np.array(np.where(dev_categories == cat))[0])
        train = train_data.take(idx, axis=0) # axis = 0 provides the reduced components for positive
        trn_labels = train_labels.take(idx, axis=0) 
        dev = dev_data.take(idx_dev, axis=0)  
        dv_labels = dev_labels.take(idx_dev, axis=0)

        #fit linear regression to each subset of data and store predictions
        lr = LinearRegression(fit_intercept=True)
        lr.fit(train, trn_labels)
        
        if len(dev) > 0:
            # in case one category has no records, added the if statement
            predictions = lr.predict(dev)
            # convert any negative predictions to zero (enables log error scoring)
            predictions[predictions < 0] = 0
            #get_errors(lr, train, trn_labels, dev, dv_labels)
            dev_actuals.append(dv_labels)
            dev_predictions.append(predictions)

    #flatten list of lists
    dev_actuals = [val for sublist in dev_actuals for val in sublist]
    dev_predictions = [val for sublist in dev_predictions for val in sublist]
    
    return(rmsle(dev_actuals, dev_predictions))

def run_rfr(train_categories, dev_categories, train_data, train_labels, dev_data, dev_labels, estimators=100):
    # for each category, pull training and dev prediction
    # consolidate them into the lists below
    dev_predictions = []
    dev_actuals = []

    for cat in list(unique(train_categories)):
        #get index of train and dev obs in this category, and separate data
        idx = list(np.array(np.where(train_categories == cat))[0])
        idx_dev = list(np.array(np.where(dev_categories == cat))[0])
        train = train_data.take(idx, axis=0) # axis = 0 provides the reduced components for positive
        trn_labels = train_labels.take(idx, axis=0) 
        dev = dev_data.take(idx_dev, axis=0)  
        dv_labels = dev_labels.take(idx_dev, axis=0)
        #print(len(dev_labels))

        if len(dev) > 0:
            #fit RF  Regression to each subset of data and store predictions
            rfr = RandomForestRegressor(n_estimators=estimators, random_state=0, n_jobs=-1)
            rfr.fit(train, trn_labels)

            # in case one category has no records, added the if statement
            predictions = rfr.predict(dev)
            
            # convert any negative predictions to zero (enables log error scoring)
            predictions[predictions < 0] = 0
            dev_actuals.append(dv_labels)
            dev_predictions.append(predictions)

    #flatten list of lists
    dev_actuals = [val for sublist in dev_actuals for val in sublist]
    dev_predictions = [val for sublist in dev_predictions for val in sublist]
    
    return(rmsle(dev_actuals, dev_predictions))

def build_cat_data(category, model, best_params, train_data, dev_data, train_labels, dev_labels, 
                   train_categories, dev_categories):
    
    if model not in ['mlp', 'knn']:
        # reduce # f dimensions
        pca = PCA(n_components=best_params['features__pca__n_components'])
        
        # Select high value original features
        selection = SelectKBest(k=best_params['features__univ_select__k'])
        
        combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
        
        # Use combined features to transform dataset:
        sub_features = combined_features.fit(train_data, train_labels)
        train_reduced = sub_features.transform(train_data)
        dev_reduced = sub_features.transform(dev_data)
    else:
        train_reduced = train_data
        dev_reduced = dev_data
    
    #now, subset out the correct set of data based on category
    idx = list(np.array(np.where(train_categories == category))[0])
    idx_dev = list(np.array(np.where(dev_categories == category))[0])
    cat_train_data = train_reduced.take(idx, axis=0)
    cat_train_labels = train_labels.take(idx, axis=0)  
    cat_dev_data = dev_reduced.take(idx_dev, axis=0)  
    cat_dev_labels = dev_labels.take(idx_dev, axis=0)
    
    return cat_train_data, cat_train_labels, cat_dev_data, cat_dev_labels

# supporting function
def build_test_cats(category, train, test, train_labels, test_ids, train_categories, test_categories):
    
    #now, subset out the correct set of data based on category
    idx = list(np.array(np.where(train_categories == category))[0])
    idx_test = list(np.array(np.where(test_categories == category))[0])
    cat_train_data = train.take(idx, axis=0)
    cat_train_labels = train_labels.take(idx, axis=0)  
    cat_test_data = test.take(idx_test, axis=0)
    cat_test_ids = test_ids.take(idx_test, axis=0)
    
    return cat_train_data, cat_train_labels, cat_test_data, cat_test_ids
