# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:29:12 2015

@author: AdenGuo
"""
import pandas
import datetime
import random

origin_file_path = '../turnstile_data_master_with_weather.csv'


def load_data():
    df = pandas.read_csv(origin_file_path)
    return df
    
def load_data_new():
    df = pandas.read_csv('new_weather_turnstile.csv')
    return df

def convert_to_int_weeksday(date, date_format):
    return int(datetime.datetime.strptime(date, date_format).strftime('%w'))

def add_weeksday(dataframe):
    dataframe['weeksday'] = pandas.Series([convert_to_int_weeksday(x,'%Y-%m-%d %H:%M:%S') for x in dataframe.datetime_str])
    return dataframe

def convert_to_float_time(datetime_str):
    time = datetime_str.split(' ')[1]
    return float(time.split(':')[0]) + float(time.split(':')[1])/60

def add_dec_time(dataframe):
    dataframe['dec_time'] = pandas.Series([convert_to_float_time(x)  for x in dataframe.datetime_str])
    return dataframe

def subset_by_time(lower_time, upper_time, dataframe):
    temp = dataframe[dataframe['dec_time'] >= lower_time]  
    return  temp[temp['dec_time'] < upper_time]

def add_datetime(dataframe):
    dataframe_string = dataframe['DATEn'] + ' ' + dataframe['TIMEn']
    dataframe['datetime'] = pandas.Series([datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in dataframe_string])
    return dataframe
    
def add_dummy_unit(dataframe):
    dummy_units = pandas.get_dummies(dataframe['unit'], prefix='unit')
    df = dataframe.join(dummy_units)
    dummy_unit_list = list(dummy_units.columns.values)
    return df,dummy_unit_list

def is_weekend(weekdsay):
    return int(weekdsay in set([0,6]))

def add_is_weekend(dataframe):
    dataframe['is_weekend'] = pandas.Series([is_weekend(weekdsay) for weekdsay in dataframe['weeksday']])
    return dataframe

def write_features_to_file(features_list, filename):
    with open(filename, 'w') as f:
        for s in features_list:
            f.write(s + '\n')

def read_features_from_file(filename):
    with open(filename, 'r') as f:
        features_list = [line.rstrip('\n') for line in f]
    return features_list

def compute_from_two_records(records_before, records_after):
    datetime_before = records_before['datetime'] 
    datetime_after = records_after['datetime']
    duration = datetime_after - datetime_before
    data_list = [records_before['UNIT'], datetime_before +  duration/2, \
                records_after['ENTRIESn_hourly']/(duration.total_seconds()/3600), \
                (records_before['maxpressurei'] + records_after['maxpressurei'])/2, \
                (records_before['maxdewpti'] + records_after['maxdewpti'])/2, \
                (records_before['mindewpti'] + records_after['mindewpti'])/2, \
                (records_before['minpressurei'] + records_after['minpressurei'])/2, \
                (records_before['meandewpti'] + records_after['meandewpti'])/2, \
                (records_before['meanpressurei'] + records_after['meanpressurei'])/2, \
                [records_before['fog'], records_after['fog']][random.randint(0,1)],\
                [records_before['rain'], records_after['rain']][random.randint(0,1)],\
                (records_before['meanwindspdi'] + records_after['meanwindspdi'])/2, \
                (records_before['mintempi'] + records_after['mintempi'])/2, \
                (records_before['meantempi'] + records_after['meantempi'])/2, \
                (records_before['maxtempi'] + records_after['maxtempi'])/2, \
                (records_before['precipi'] + records_after['precipi'])/2, \
                [records_before['thunder'], records_after['thunder']][random.randint(0,1)]]
    return data_list

def create_new_dateframe(dataframe):
    new_df = add_datetime(dataframe)
    column_name_list = ['unit', 'datetime','hourly_entries', 'maxpressurei', \
                  'maxdewpti', 'mindewpti', 'minpressurei', 'meandewpti', \
                  'meanpressurei', 'fog', 'rain', 'meanwindspdi', 'mintempi', \
                  'meantempi', 'maxtempi', 'precipi', 'thunder']      
    df = pandas.DataFrame(columns=column_name_list)
    unit_list = list(set(new_df['UNIT']))
    for unit in unit_list:
        print(unit)        
        temp_df_1 = pandas.DataFrame(columns=column_name_list)
        temp_df_0 = new_df[new_df['UNIT'] == unit]
        temp_df_0 = temp_df_0.sort('datetime')
        for i in range(1,len(temp_df_0)):
            temp_df_1.loc[i-1] = compute_from_two_records(temp_df_0.iloc[i-1], temp_df_0.iloc[i])
        df = df.append(temp_df_1, ignore_index = True)
    return df

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

def select_by_weeksday_time(dataframe, time_lower, time_upper, weeksday_list):
    df = dataframe[dataframe['dec_time'] >= time_lower]
    df = df[df['dec_time'] < time_upper]
    result = pandas.DataFrame()
    for day in weeksday_list:    
        temp = df[df['weeksday'] == day]
        result = result.append(temp)
    return result

works_day = set([1,2,3,4,5])
weekend = set([6,0])
weeks_day = set([0,1,2,3,4,5,6])

#time_intervals = [(0,5,works_day),(5,9,works_day),(9,12,works_day),(12,16,works_day),(16,20,works_day),(20,23.999,works_day),\
#                  (0,5,weekend),(5,9,weekend),(9,12,weekend),(12,16,weekend),(16,20,weekend),(20,23.999,weekend)]

#time_intervals = [(0,23.999,weeks_day)]
#0.46
#time_intervals = [(0,23.999,works_day), (0,23.999,weekend)]
#0.50
#time_intervals = [(0,12,works_day), (12,23.999,works_day),\
#                  (0,12,weekend), (12,23.999,weekend)]
#0.62
#time_intervals = [(0,8,works_day), (8,16,works_day), (16,23.999,works_day),\
#                  (0,8,weekend), (8,16,weekend), (16,23.999,weekend)]
#0.69031921372663829                  
#time_intervals = [(0,6,works_day), (6,12,works_day), (12,18,works_day), \
#                  (18,23.999,works_day), \
#                  (0,6,weekend), (6,12,weekend), (12,18,weekend), \
#                  (18,23.999,weekend)]
#0.74574564770581975
time_intervals = [(0,4,works_day), (4,8,works_day), (8,12,works_day), \
                 (12,16,works_day), (16,20,works_day), (20,23.999,works_day),\
                 (0,4,weekend), (4,8,weekend), (8,12,weekend), \
                 (12,16,weekend), (16,20,weekend), (20,23.999,weekend)]
#0.82182085018382367
#time_intervals = [(0,5,works_day), (5,10,works_day), (10,16,works_day), \
#                 (16,20,works_day), (20,23.999,works_day),\
#                 (0,5,weekend), (5,10,weekend), (10,16,weekend), \
#                 (16,20,weekend), (20,23.999,weekend)]
#0.77536500283829524

#time_intervals = [(0,3,works_day), (3,6,works_day), (6,9,works_day), \
#                 (9,12,works_day), (12,15,works_day),(15,18,works_day),\
#                 (18,21,works_day),(21,23.999,works_day),
#                 (0,3,weekend), (3,6,weekend), (6,9,weekend), \
#                 (9,12,weekend), (12,15,weekend),(15,18,weekend),\
#                 (18,21,weekend),(21,23.999,weekend)]
#0.83774175839176723
import statsmodels.api as sm                  

def add_constant(features):
    const_series = pandas.DataFrame({'const':pandas.Series([1 for x in range(len(features))], index = features.index)})
    new_features = pandas.concat([const_series,features],axis=1)    
    return new_features

def linear_regression(features, values):

    features_new = add_constant(features)
    model = sm.OLS(values,features_new)
    results = model.fit()
    #intercept = results.params[0]
    #params = results.params[1:]    
    return results.params

def create_coeff_matrix(time_intervals, dataframe):
    global features_list
    result_matrix_columns_list = [x for x in features_list]
    result_matrix_columns_list.insert(0,'const')
    result_matrix = pandas.DataFrame(columns=result_matrix_columns_list)    
    for index in range(len(time_intervals)):
        print(index)
        sub_df = select_by_weeksday_time(dataframe, \
                   time_intervals[index][0], time_intervals[index][1],\
                   time_intervals[index][2])
        features = sub_df[features_list]
        values = sub_df['hourly_entries']
        coeffs = linear_regression(features, values)
        result_matrix.loc[index] = coeffs
    return result_matrix

def get_coeffs_index(a_record, time_intervals):
    for index in range(len(time_intervals)):
        if a_record['dec_time'] >= time_intervals[index][0] and \
           a_record['dec_time'] < time_intervals[index][1] and \
           a_record['weeksday'] in time_intervals[index][2]:
            return index

def make_prediction(a_record, coeffs, features_list):
    result = 0    
    for feature in features_list:
        result = result + a_record[feature]*coeffs[feature]
        #print result
    #print result    
    result = result + coeffs['const']
    if result < 0:
        result = 0.0
    return result

def make_predictions_dataframe(dataframe, time_intervals, features_list, \
    coeff_matrix):
    result = pandas.Series()
    for index, a_record in dataframe.iterrows():
        coeff_index = get_coeffs_index(a_record, time_intervals)
        coeffs = coeff_matrix.loc[coeff_index]
        result.loc[index] = make_prediction(a_record, coeffs, features_list)
        #print index
    return result
 
import numpy
       
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
new_dataframe, dummy_unit_list = add_dummy_unit(new_dataframe)
features_list = ['rain','meantempi', 'dec_time','meanwindspdi','precipi', 'is_weekend']
features_list.extend(dummy_unit_list)
write_features_to_file(features_list, 'features_list')
new_dataframe.to_csv('new_weather_turnstile.csv')

dataframe = pandas.read_csv('new_weather_turnstile.csv')
from sklearn.cross_validation import train_test_split
test_propotion = 0.1
train_new_weather_turnstile, test_new_weather_turnstile = \
train_test_split(dataframe, test_size = test_propotion)
train_new_weather_turnstile.to_csv('train_new_weather_turnstile.csv')
test_new_weather_turnstile.to_csv('test_new_weather_turnstile.csv')

features_list = read_features_from_file('features_list')
dataframe = pandas.read_csv('train_new_weather_turnstile.csv')
#dataframe = pandas.read_csv('new_weather_turnstile.csv')
coeff_matrix = create_coeff_matrix(time_intervals, dataframe)
coeff_matrix.to_csv('coeff_matrix.csv')

features_list = read_features_from_file('features_list')
coeff_matrix = pandas.read_csv('coeff_matrix.csv')
test_dataframe = pandas.read_csv('test_new_weather_turnstile.csv')
#test_dataframe = pandas.read_csv('new_weather_turnstile.csv')
predictions = make_predictions_dataframe(test_dataframe, time_intervals,\
                                         features_list, coeff_matrix)
sum_square_complex = compute_sum_square(test_dataframe['hourly_entries'], predictions)
r_square_complex = compute_r_squared(test_dataframe['hourly_entries'], predictions)
"""