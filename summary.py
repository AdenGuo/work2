# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:30:26 2015

@author: adenguo
"""

import linear_models
import nonparametric_model
import pandas

def run_for(n):
    index_list = []
    SSE_linear = []
    SSE_two_level = []
    SSE_nonparametric = []
    R2_linear = []
    R2_two_level = []
    R2_nonparametric = []
    dataframe = linear_models.load_data()    
    new_dataframe = linear_models.create_new_dateframe_0(dataframe)
    new_dataframe = linear_models.add_datetime_str(new_dataframe)
    new_dataframe = linear_models.add_dec_time(new_dataframe)
    new_dataframe = linear_models.add_weeksday(new_dataframe)
    new_dataframe = linear_models.add_is_weekend(new_dataframe)
    new_dataframe, dummy_unit_list = linear_models.add_dummy_unit(new_dataframe)
    features_list = ['rain','meantempi', 'dec_time','meanwindspdi','precipi', 'is_weekend']
    features_list.extend(dummy_unit_list)
    numeric_features_list_with_weight = [('meantempi',1), ('dec_time',2), \
                                     ('meanwindspdi',1),('precipi',1)]
    categorical_features = ['unit', 'rain', 'is_weekend']
    linear_models.write_features_to_file(features_list, 'features_list')
    nonparametric_model.write_features_with_weight_to_file(numeric_features_list_with_weight, \
                                       'numeric_features_list_weight')
    nonparametric_model.write_features_to_file(categorical_features, 'categorical_features')    
    new_dataframe.to_csv('new_weather_turnstile.csv')    
    from sklearn.cross_validation import train_test_split
    test_propotion = 0.1
    delta = 2
    k = 10
    for i in range(n):
        index_list.append('Split' + ' ' + str(i + 1))
        print "Testing " + str(i + 1) + " begin!"
        print "Data spliting..."            
        dataframe = pandas.read_csv('new_weather_turnstile.csv')
        train_new_weather_turnstile, test_new_weather_turnstile = \
        train_test_split(dataframe, test_size = test_propotion)
        train_new_weather_turnstile.to_csv('train_new_weather_turnstile.csv')
        test_new_weather_turnstile.to_csv('test_new_weather_turnstile.csv')
        
        print "Test simple linear model..." 
        time_intervals = [(0,23.999,linear_models.weeks_day)]
        features_list = linear_models.read_features_from_file('features_list')
        dataframe = pandas.read_csv('train_new_weather_turnstile.csv')
        coeff_matrix = linear_models.create_coeff_matrix(features_list, time_intervals, dataframe)
        coeff_matrix.to_csv('coeff_matrix.csv')        
        features_list = linear_models.read_features_from_file('features_list')
        coeff_matrix = pandas.read_csv('coeff_matrix.csv')
        test_dataframe = pandas.read_csv('test_new_weather_turnstile.csv')
        predictions = linear_models.make_predictions_dataframe(test_dataframe, time_intervals,\
                                                 features_list, coeff_matrix)
        sum_square_complex = linear_models.compute_sum_square(test_dataframe['hourly_entries'], predictions)
        r_square_complex = linear_models.compute_r_squared(test_dataframe['hourly_entries'], predictions)
        SSE_linear.append(sum_square_complex)
        R2_linear.append(r_square_complex)
        
        print "Test level two regression model..."
        time_intervals = [(0,4,linear_models.works_day), (4,8,linear_models.works_day),\
                          (8,12,linear_models.works_day), (12,16,linear_models.works_day), \
                          (16,20,linear_models.works_day), (20,23.999,linear_models.works_day),\
                          (0,4,linear_models.weekend), (4,8,linear_models.weekend),\
                          (8,12,linear_models.weekend), (12,16,linear_models.weekend), \
                          (16,20,linear_models.weekend), (20,23.999,linear_models.weekend)]
        features_list = linear_models.read_features_from_file('features_list')
        features_list.remove('is_weekend')
        dataframe = pandas.read_csv('train_new_weather_turnstile.csv')
        coeff_matrix = linear_models.create_coeff_matrix(features_list, time_intervals, dataframe)
        coeff_matrix.to_csv('coeff_matrix.csv')        
        features_list = linear_models.read_features_from_file('features_list')
        features_list.remove('is_weekend')
        coeff_matrix = pandas.read_csv('coeff_matrix.csv')
        test_dataframe = pandas.read_csv('test_new_weather_turnstile.csv')
        predictions = linear_models.make_predictions_dataframe(test_dataframe, time_intervals,\
                                                 features_list, coeff_matrix)
        sum_square_complex = linear_models.compute_sum_square(test_dataframe['hourly_entries'], predictions)
        r_square_complex = linear_models.compute_r_squared(test_dataframe['hourly_entries'], predictions)
        SSE_two_level.append(sum_square_complex)
        R2_two_level.append(r_square_complex)

        print "Test non-parametric model..."                    
        numeric_features_list_with_weight = nonparametric_model.read_features_with_weight_from_file \
                                    ('numeric_features_list_weight')
        categorical_features = nonparametric_model.read_features_from_file('categorical_features')
        dataframe = pandas.read_csv('train_new_weather_turnstile.csv')
        test_dataframe = pandas.read_csv('test_new_weather_turnstile.csv')
        
        predictions = nonparametric_model.prediction_dataframe(test_dataframe, dataframe,\
                                           categorical_features, \
                                           numeric_features_list_with_weight,\
                                           k, delta)
        sum_square_complex = nonparametric_model.compute_sum_square(test_dataframe['hourly_entries'], predictions)
        r_square_complex = nonparametric_model.compute_r_squared(test_dataframe['hourly_entries'], predictions) 
        SSE_nonparametric.append(sum_square_complex)
        R2_nonparametric.append(r_square_complex)
    result = pandas.DataFrame({'SSE linear':pandas.Series(SSE_linear, index_list), \
                               'SSE two level': pandas.Series(SSE_two_level, index_list), \
                               'SSE nonparametric': pandas.Series(SSE_nonparametric, index_list),\
                               'R2 linear':pandas.Series(R2_linear, index_list),\
                               'R2 two level':pandas.Series(R2_two_level, index_list),\
                               'R2 nonparametric':pandas.Series(R2_nonparametric, index_list)}) 
    return result
        
def load_result():
    return pandas.read_csv("ten_splits_result.csv", index_col = 0)            