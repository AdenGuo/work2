# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 08:10:19 2015

@author: adenguo
"""

import pandas
import datetime
import random
import numpy


origin_file_path = '../turnstile_data_master_with_weather.csv'


def load_data():
    df = pandas.read_csv(origin_file_path)
    return df

def add_datetime(dataframe):
    dataframe_string = dataframe['DATEn'] + ' ' + dataframe['TIMEn']
    dataframe['datetime'] = pandas.Series([datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in dataframe_string])
    return dataframe

def create_new_dateframe_0(dataframe):
    new_df = add_datetime(dataframe)
    new_df['unit'] = dataframe['UNIT']
    new_df['hourly_entries'] = dataframe['ENTRIESn_hourly']
    del new_df['UNIT']
    del new_df['ENTRIESn_hourly']
    return new_df

def add_datetime_str(dataframe):
    dataframe['datetime_str'] = pandas.Series([x.strftime('%Y-%m-%d %H:%M:%S') \
                                for x in dataframe['datetime']])
    return dataframe
    
def convert_to_float_time(datetime_str):
    time = datetime_str.split(' ')[1]
    return float(time.split(':')[0]) + float(time.split(':')[1])/60

def add_dec_time(dataframe):
    dataframe['dec_time'] = pandas.Series([convert_to_float_time(x)  for x in dataframe.datetime_str])
    return dataframe

def convert_to_int_weeksday(date, date_format):
    return int(datetime.datetime.strptime(date, date_format).strftime('%w'))
    
def add_weeksday(dataframe):
    dataframe['weeksday'] = pandas.Series([convert_to_int_weeksday(x,'%Y-%m-%d %H:%M:%S') for x in dataframe.datetime_str])
    return dataframe

def is_weekend(weekdsay):
    return int(weekdsay in set([0,6]))

def add_is_weekend(dataframe):
    dataframe['is_weekend'] = pandas.Series([is_weekend(weekdsay) for weekdsay in dataframe['weeksday']])
    return dataframe

def add_dummy_unit(dataframe):
    dummy_units = pandas.get_dummies(dataframe['unit'], prefix='unit')
    df = dataframe.join(dummy_units)
    dummy_unit_list = list(dummy_units.columns.values)
    return df,dummy_unit_list

def write_features_with_weight_to_file(features_list_weight, filename):
    with open(filename, 'w') as f:
        for t in features_list_weight:
            for s in t:            
                f.write(str(s) + ',')
            f.write('\n')

def read_features_with_weight_from_file(filename):
    with open(filename, 'r') as f:
        features_list_weight = [(line.rstrip('\n').split(',')[0], float(line.rstrip('\n').split(',')[1]))  for line in f]
    return features_list_weight

def write_features_to_file(features_list, filename):
    with open(filename, 'w') as f:
        for s in features_list:
            f.write(s + '\n')

def read_features_from_file(filename):
    with open(filename, 'r') as f:
        features_list = [line.rstrip('\n') for line in f]
    return features_list
    
def calculate_distance(record_one, record_two, numeric_features_list_with_weight):
    distance = 0.0
    for feature_weight in numeric_features_list_with_weight:
        distance = distance + (record_one[feature_weight[0]] - record_two[feature_weight[0]])**2\
                              *feature_weight[1]
    return distance
    
def create_categorical_features_dict(record, categorical_features):
    result = {}    
    for index in range(len(categorical_features)):
        result[categorical_features[index]] = record[categorical_features[index]]
    return result
        
def select_by_categorical_features(dataframe, categorical_features_dict):
    df = dataframe
    for key in categorical_features_dict:
        df = df[df[key] == categorical_features_dict[key]]
    return df

def calculate_distance_dataframe(record, dataframe, numeric_features_list_with_weight):
    distances = pandas.Series()
    for index, a_record in dataframe.iterrows():
        distances.loc[index] = calculate_distance(record, a_record, numeric_features_list_with_weight)
    dataframe['distance'] = distances
    return dataframe

def get_k_smallest_distance(dataframe, k):
    if len(dataframe) >= k:    
        return dataframe.sort('distance')[:k]
    else:
        return dataframe

#delta = 1.5, k = 10
#0.68

#delta = 2, k = 10
#0.70

def prediction_a_record(record, dataframe, categorical_features, \
                        numeric_features_list_with_weight, k, delta):
    categorical_features_dict = create_categorical_features_dict(record, categorical_features)
    new_df = select_by_categorical_features(dataframe, categorical_features_dict)
    new_df = calculate_distance_dataframe(record, new_df, numeric_features_list_with_weight)
    new_df = get_k_smallest_distance(new_df, k)
    new_df = new_df[numpy.abs(new_df.hourly_entries-new_df.hourly_entries.mean()) <= (delta*new_df.hourly_entries.std())]
    return new_df['hourly_entries'].mean()

def prediction_dataframe(test_dataframe, dataframe, categorical_features, \
                         numeric_features_list_with_weight, k, delta):
    result = pandas.Series()
    number = 0    
    for index, a_record in test_dataframe.iterrows():
#        print number
#        number = number + 1
        result.loc[index] =  prediction_a_record(a_record, dataframe, \
                                           categorical_features,\
                                           numeric_features_list_with_weight,\
                                           k, delta)
    return result
    
def compute_sum_square(values, predictions):
    return numpy.sum((values - predictions)**2)

def compute_r_squared(values, predictions):
    r_squared = 1 - numpy.sum((values - predictions)**2)\
                    /numpy.sum((values - numpy.mean(values))**2)
    return r_squared
"""
dataframe = load_data()
#new_dataframe = create_new_dateframe(dataframe)
new_dataframe = create_new_dateframe_0(dataframe)
new_dataframe = add_datetime_str(new_dataframe)
new_dataframe = add_dec_time(new_dataframe)
new_dataframe = add_weeksday(new_dataframe)
new_dataframe = add_is_weekend(new_dataframe)
categorical_features = ['unit', 'rain', 'is_weekend']
write_features_with_weight_to_file(numeric_features_list_with_weight, \
                                   'numeric_features_list_weight')
write_features_to_file(categorical_features, 'categorical_features')
new_dataframe.to_csv('new_weather_turnstile.csv')

dataframe = pandas.read_csv('new_weather_turnstile.csv')
from sklearn.cross_validation import train_test_split
test_propotion = 0.1
train_new_weather_turnstile, test_new_weather_turnstile = \
train_test_split(dataframe, test_size = test_propotion)
train_new_weather_turnstile.to_csv('train_new_weather_turnstile.csv')
test_new_weather_turnstile.to_csv('test_new_weather_turnstile.csv')

numeric_features_list_with_weight = read_features_with_weight_from_file \
                                    ('numeric_features_list_weight')
categorical_features = read_features_from_file('categorical_features')
dataframe = pandas.read_csv('train_new_weather_turnstile.csv')
test_dataframe = pandas.read_csv('test_new_weather_turnstile.csv')

predictions = prediction_dataframe(test_dataframe, dataframe,\
                                   categorical_features, \
                                   numeric_features_list_with_weight,\
                                   k, delta)
sum_square_complex = compute_sum_square(test_dataframe['hourly_entries'], predictions)
r_square_complex = compute_r_squared(test_dataframe['hourly_entries'], predictions)
"""        
