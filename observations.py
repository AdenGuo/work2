# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 18:34:54 2015

@author: adenguo
"""
import pandas
import linear_models
import matplotlib.pyplot as plt

from linear_models import load_data
from linear_models import convert_to_int_weeksday
from linear_models import add_weeksday
from linear_models import is_weekend
from linear_models import add_is_weekend
from linear_models import add_datetime
from linear_models import add_datetime_str

def rain_correlated_time(unit_number, time_number):
    dataframe = load_data()
    unit_name = dataframe['UNIT'].iloc[unit_number]
    sub_df = dataframe[dataframe['UNIT'] == unit_name]
    time_ser = sub_df.TIMEn.value_counts()
    time_ser.sort(axis = 0, ascending = False)
    average_rain_list =[]
    average_no_rain_list = []
    leading_index = time_ser.index[:time_number]
    for time in leading_index:
        time_df = sub_df[sub_df['TIMEn'] == time]                
        average_rain_list.append(time_df[time_df['rain'] == 1].EXITSn_hourly. mean())
        average_no_rain_list.append(time_df[time_df['rain'] == 0].EXITSn_hourly. mean())
    time_count = pandas.DataFrame({'rain' : pandas.Series(average_rain_list, leading_index), \
                                   'no_rain': pandas.Series(average_no_rain_list, leading_index)})
    return time_count


def plot_daily_data(unit_number):
    dataframe = load_data()
    dataframe = add_datetime(dataframe)
    dataframe = add_datetime_str(dataframe)
    dataframe = add_weeksday(dataframe)
    dataframe = add_is_weekend(dataframe)
    one_unit_df = dataframe[dataframe['UNIT'] == dataframe.iloc[unit_number]['UNIT']]
    one_unit_df = one_unit_df[one_unit_df['is_weekend'] == 0 ]
    groups = one_unit_df.groupby('DATEn')
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.Hour, group.ENTRIESn_hourly, marker='o', linestyle='-', ms=12, label=name)
    #ax.legend(("One color a day"))
    plt.xlabel("Time(hour)", fontsize = 20)
    plt.ylabel("Number of Entries", fontsize = 20)
    plt.title("Entries Versus Time In One Day", fontsize = 20)
    plt.text(2,20000,"One color for a day",fontsize = 16)
    plt.show()
    return plt                             