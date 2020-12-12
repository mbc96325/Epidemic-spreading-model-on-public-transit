import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pickle
import time
colors = sns.color_palette("Paired")

def plot_impact(save_fig):
    results = pd.read_csv('../data/impact_cutting_bus_region.csv')
    bus_regions  = pd.read_csv('../data/bus_region.txt', sep = ',')
    bus_regions = bus_regions.dropna()
    # drop the last few str records
    bus_regions = bus_regions.loc[bus_regions['BusStopCod'].apply(lambda x: 'N' not in x)]
    #
    bus_stop_id_info = pd.read_csv('../data/bus_stop_id_lookup.csv')
    bus_regions['BusStopCod'] = bus_regions['BusStopCod'].astype('int')
    bus_regions = bus_regions.merge(bus_stop_id_info, left_on = ['BusStopCod'], right_on =['Old_stop_id'])
    region_id = list(pd.unique(bus_regions['OBJECTID']))
    date_list = list(range(start_date, end_date + 1, 1))  #
    num_people_region = []
    for area_id in region_id:
        print('===============CURRENT region ' + str(area_id) + '=============')
        with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(sample_seed)
                  + '_close_bus_' + 'region_' + str(int(area_id)) +  'random.pickle', 'rb') as handle:
            data = pickle.load(handle)
        num_people = len(pd.unique(data['P_id']))
        num_people_region.append(num_people)
    with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(
            sample_size) + '_seed_' + str(sample_seed) + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        original_people = len(pd.unique(data['P_id']))
    results['percentage_people_reduction'] = num_people_region
    results['percentage_people_reduction'] = (original_people - results['percentage_people_reduction']) / original_people
    R0_old = 1.76299
    results['R0_reduction'] =(R0_old - results['R0']) / R0_old
    results.to_csv('../data/impact_cutting_bus_region.csv',index=False)
    a=1
if __name__ == '__main__':
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31
    sample_size = 100000
    sample_seed = 0

    plot_impact(save_fig = 0)