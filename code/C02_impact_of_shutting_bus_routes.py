import pandas as pd
import numpy as np
import time
import pickle
import random
import matplotlib.pyplot as plt
import _constants
import _simulation_engine as sim_eng
from sklearn.utils import shuffle


import warnings
warnings.filterwarnings('ignore')

def get_new_pax_data(data, route_stop_id, route_demand, bus_id_info, percentage_close):
    total_close = int(round(len(route_demand) * percentage_close))
    used_route = route_demand.iloc[total_close:, :]
    closed_routes = route_demand.iloc[:total_close, :]


    idx_closed = data['Srvc_Number'].isin(list(closed_routes['Srvc_Number']))
    data_closed = data.loc[idx_closed]
    data_unclose = data.loc[~idx_closed]
    route_stop_id_used = route_stop_id.loc[route_stop_id['route_id'].isin(list(used_route['Srvc_Number']))]
    data_closed = data_closed.merge(route_stop_id_used, left_on = ['boarding_stop'],right_on = ['bus_stop'])
    data_closed = data_closed.merge(route_stop_id_used, left_on = ['alighting_stop'],right_on = ['bus_stop'])
    data_closed = data_closed.loc[data_closed['route_id_x'] == data_closed['route_id_y']]
    data_closed = data_closed.drop_duplicates(['P_id','trip_id'])
    ### find the bus id
    bus_id_info = bus_id_info.rename(columns = {'start_time':'bus_time'})
    bus_id_info['time_int'] = bus_id_info['bus_time'] // 1800 # focus on 30 min interval
    data_closed['time_int'] = data['start_time'] // 1800
    data_closed = data_closed.merge(bus_id_info, left_on = ['date','time_int','route_id_x','boarding_stop'],
                                    right_on = ['date','time_int','Srvc_Number','boarding_stop'])
    data_closed['time_diff'] = np.abs(data_closed['start_time'] - data_closed['bus_time'])
    data_closed = data_closed.sort_values(['time_diff'])
    data_closed = data_closed.drop_duplicates(['P_id','trip_id'])
    data_closed['bus_id'] = data_closed['bus_id_y']
    data_closed['start_time'] = data_closed['bus_time']

    data_new = data_closed.loc[:, ['P_id','bus_id','date','start_time','ride_duration','boarding_stop','alighting_stop']]
    data_new = pd.concat([data_new, data_unclose])
    return data_new

def get_new_pax_data_random(data, route_stop_id, route_demand, bus_id_info, percentage_close):
    total_close = int(round(len(route_demand) * percentage_close))

    route_demand = shuffle(route_demand, random_state=int(total_close))

    used_route = route_demand.iloc[total_close:, :]
    closed_routes = route_demand.iloc[:total_close, :]


    idx_closed = data['Srvc_Number'].isin(list(closed_routes['Srvc_Number']))
    data_closed = data.loc[idx_closed]
    data_unclose = data.loc[~idx_closed]
    route_stop_id_used = route_stop_id.loc[route_stop_id['route_id'].isin(list(used_route['Srvc_Number']))]
    data_closed = data_closed.merge(route_stop_id_used, left_on = ['boarding_stop'],right_on = ['bus_stop'])
    data_closed = data_closed.merge(route_stop_id_used, left_on = ['alighting_stop'],right_on = ['bus_stop'])
    data_closed = data_closed.loc[data_closed['route_id_x'] == data_closed['route_id_y']]
    data_closed = data_closed.drop_duplicates(['P_id','trip_id'])
    ### find the bus id
    bus_id_info = bus_id_info.rename(columns = {'start_time':'bus_time'})
    bus_id_info['time_int'] = bus_id_info['bus_time'] // 1800 # focus on 30 min interval
    data_closed['time_int'] = data['start_time'] // 1800
    data_closed = data_closed.merge(bus_id_info, left_on = ['date','time_int','route_id_x','boarding_stop'],
                                    right_on = ['date','time_int','Srvc_Number','boarding_stop'])
    data_closed['time_diff'] = np.abs(data_closed['start_time'] - data_closed['bus_time'])
    data_closed = data_closed.sort_values(['time_diff'])
    data_closed = data_closed.drop_duplicates(['P_id','trip_id'])
    data_closed['bus_id'] = data_closed['bus_id_y']
    data_closed['start_time'] = data_closed['bus_time']

    data_new = data_closed.loc[:, ['P_id','bus_id','date','start_time','ride_duration','boarding_stop','alighting_stop']]
    data_new = pd.concat([data_new, data_unclose])
    return data_new



def get_new_pax_data_region(data, route_stop_id, route_demand, bus_id_info, banned_bus_stop):
    banned_bus_stop = banned_bus_stop.merge(route_stop_id, left_on = ['bus_stop'], right_on=['bus_stop'])
    drop_routes = list(pd.unique(banned_bus_stop['route_id']))
    drop_routes_idx = route_demand['Srvc_Number'].isin(drop_routes)
    used_route = route_demand.loc[~drop_routes_idx]
    closed_routes = route_demand.loc[drop_routes_idx, :]


    idx_closed = data['Srvc_Number'].isin(list(closed_routes['Srvc_Number']))
    data_closed = data.loc[idx_closed]
    data_unclose = data.loc[~idx_closed]
    route_stop_id_used = route_stop_id.loc[route_stop_id['route_id'].isin(list(used_route['Srvc_Number']))]
    data_closed = data_closed.merge(route_stop_id_used, left_on = ['boarding_stop'],right_on = ['bus_stop'])
    data_closed = data_closed.merge(route_stop_id_used, left_on = ['alighting_stop'],right_on = ['bus_stop'])
    data_closed = data_closed.loc[data_closed['route_id_x'] == data_closed['route_id_y']]
    data_closed = data_closed.drop_duplicates(['P_id','trip_id'])
    ### find the bus id
    bus_id_info = bus_id_info.rename(columns = {'start_time':'bus_time'})
    bus_id_info['time_int'] = bus_id_info['bus_time'] // 1800 # focus on 30 min interval
    data_closed['time_int'] = data['start_time'] // 1800
    data_closed = data_closed.merge(bus_id_info, left_on = ['date','time_int','route_id_x','boarding_stop'],
                                    right_on = ['date','time_int','Srvc_Number','boarding_stop'])
    data_closed['time_diff'] = np.abs(data_closed['start_time'] - data_closed['bus_time'])
    data_closed = data_closed.sort_values(['time_diff'])
    data_closed = data_closed.drop_duplicates(['P_id','trip_id'])
    data_closed['bus_id'] = data_closed['bus_id_y']
    data_closed['start_time'] = data_closed['bus_time']

    data_new = data_closed.loc[:, ['P_id','bus_id','date','start_time','ride_duration','boarding_stop','alighting_stop']]
    data_new = pd.concat([data_new, data_unclose])
    return data_new

def get_route_stop_id(trip_sg, stop_times_sg,bus_stop_id_info):

    route_stop = stop_times_sg.merge(trip_sg, on =['trip_id'])
    route_stop = route_stop.merge(bus_stop_id_info,left_on = ['stop_id'], right_on = ['Old_stop_id'])
    # print(route_stop['arrival_time'].dtype)
    # route_stop['time_stamp'] = route_stop['arrival_time'].str.split(':')
    # route_stop['time_stamp'] = route_stop['time_stamp'].apply(lambda x: float(x[0]) * 3600 + float(x[1]) * 60 + float(x[2]))
    # route_stop['time_int'] = route_stop['time_stamp'] // 3600
    route_stop_list = route_stop.drop_duplicates(['bus_stop','route_id'])
    return route_stop_list[['bus_stop','route_id']]



def construct_PT_encounter_net(data,time_int_s, time_int_e):
    #['P_id', 'bus_id', 'date', 'start_time', 'ride_duration', 'boarding_stop', 'alighting_stop']
    data = data.drop(columns = ['start_time'])
    data = data.rename(columns={'start_time_day': 'start_time', 'end_time_day': 'end_time'})
    data_net = data[['P_id','bus_id','start_time',
                     'end_time']].merge(data[['P_id','bus_id','start_time','end_time']],on = ['bus_id'])
    # WLOG, assume y is the late people
    data_net = data_net.loc[data_net['start_time_y']>=data_net['start_time_x']]
    # filter people without contact
    data_net = data_net.loc[data_net['end_time_x'] > data_net['start_time_y']]
    #
    data_net['time_int_s'] = time_int_s
    data_net['time_int_e'] = time_int_e
    data_net['left_bound_x'] = data_net[['start_time_x', 'time_int_s']].max(axis=1)
    data_net['right_bound_x'] = data_net[['end_time_x', 'time_int_e']].min(axis=1)
    data_net['left_bound_y'] = data_net[['start_time_y', 'time_int_s']].max(axis=1)
    data_net['right_bound_y'] = data_net[['end_time_y', 'time_int_e']].min(axis=1)
    data_net['contact_duration'] = data_net[['right_bound_x','right_bound_y']].min(axis=1) - data_net['left_bound_y']
    data_net = data_net.loc[data_net['contact_duration']>0]
    data_net = data_net.loc[data_net['P_id_x']!=data_net['P_id_y']]
    # data_net['w_ij'] = data_net['contact_duration'] / (time_int_e - time_int_s)
    # a=1
    return data_net[['P_id_x','P_id_y','contact_duration']]

def PT_connection(data,date_list,time_interval):
    data['start_time_day'] = data['start_time'] + data['date'] * 3600*24
    data['end_time_day'] = data['start_time_day'] + data['ride_duration']
    time_int_id = 0
    PT_net = {}
    for date in date_list:
        data_date = data.loc[data['date'] == date]
        for time_int in range(0, 86400, time_interval):
            # print('current time interval id:', time_int_id)
            time_int_id += 1
            # give data for one time interval
            time_int_s = date* 3600*24 + time_int
            time_int_e = date* 3600*24 + time_int + time_interval
            data_time_int = data_date.loc[~((data_date['start_time_day'] > time_int_e)|
                                          (data_date['end_time_day']< time_int_s))]
            if len(data_time_int) > 0:
                data_net = construct_PT_encounter_net(data_time_int, time_int_s, time_int_e)
                data_net = data_net.groupby(['P_id_x', 'P_id_y'])['contact_duration'].sum().reset_index()
                data_net.loc[data_net['contact_duration'] > time_interval,
                             'contact_duration'] = time_interval  # should not happen, just for conservative consideration.

                data_net['time_int_id'] = time_int_id
                PT_net[time_int_id] = data_net

            else:
                PT_net[time_int_id] = pd.DataFrame({'P_id_x':[],'P_id_y':[],'contact_duration':[]})
    # print('Finish PT encounter network')
    return PT_net


def cal_pt_cont_time(PT_net_new):
    total_contact_time_list_pt = []
    total_contact_time_square_list_pt = []
    time_int_list = []
    for time_int in PT_net_new:
        time_int_list.append(time_int)
        pt_net_int = PT_net_new[time_int]
        total_contact_time_list_pt.append(pt_net_int['contact_duration'].sum())
        total_contact_time_square_list_pt.append(np.sum(np.square(pt_net_int['contact_duration'])))

    contact_time = pd.DataFrame({'time_interval': time_int_list, 'total_contact_time': total_contact_time_list_pt,
                                 'total_contact_time_square': total_contact_time_square_list_pt})
    return contact_time


if __name__ == '__main__':
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31
    sample_size = 100000
    sample_seed = 0
    GENERATE_SAMPLE = True
    GENERATE_CONTACT_TIME = False
    bus_id_info = pd.read_csv('../data/bus_id_lookup.csv')
    bus_id_info['Srvc_Number'] = bus_id_info['Srvc_Number'].str.replace(' ','')
    bus_stop_id_info = pd.read_csv('../data/bus_stop_id_lookup.csv')
    with open('../data/data_bus_arrival_info' + str(start_date) + '_' + str(end_date) + '.pickle',
              'rb') as handle:
        bus_info_dict = pickle.load(handle)

    bus_arr_info_df = []
    for day in bus_info_dict:
        bus_info = bus_info_dict[day]
        bus_info['date'] = day
        bus_arr_info_df.append(bus_info)
    bus_arr_info_df = pd.concat(bus_arr_info_df)
    bus_arr_info_df = bus_arr_info_df.merge(bus_id_info, on =['bus_id'])

    if GENERATE_SAMPLE:
        tic = time.time()
        percentage_close_list = [0.1, 0.2, 0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        trip_sg = pd.read_csv('../data/gtfs/trips.txt', sep = ',')
        stop_times_sg = pd.read_csv('../data/gtfs/stop_times.txt', sep=',')
        route_stop_id = get_route_stop_id(trip_sg, stop_times_sg,bus_stop_id_info)
        print('process data time', time.time() - tic)
        with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(
                sample_size) + '_seed_' + str(sample_seed) + '.pickle', 'rb') as handle:
            data = pickle.load(handle)

        data = data.merge(bus_id_info, on=['bus_id'])
        data['trip_id'] = list(range(0, len(data)))
        route_demand = data.groupby(['Srvc_Number'], sort=False)['P_id'].count().reset_index()
        route_demand = route_demand.rename(columns={'P_id': 'demand'})
        # ######################### from high to low
        # route_demand = route_demand.sort_values(['demand'], ascending=False)
        # for percentage_close in percentage_close_list:
        #     print('===============CURRENT percentage ' + str(percentage_close) + '=============')
        #     data_new = get_new_pax_data(data, route_stop_id, route_demand, bus_arr_info_df, percentage_close)
        #     with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(
        #             sample_seed) + '_close_bus_' + str(int(percentage_close * 100)) + 'high_low.pickle', 'wb') as handle:
        #         pickle.dump(data_new, handle)
        # ######################## from low to high
        # route_demand = route_demand.sort_values(['demand'], ascending=True)
        # for percentage_close in percentage_close_list:
        #     print('===============CURRENT percentage ' + str(percentage_close) + '=============')
        #     data_new = get_new_pax_data(data, route_stop_id, route_demand, bus_arr_info_df, percentage_close)
        #     with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(
        #             sample_seed) + '_close_bus_' + str(int(percentage_close * 100)) + 'low_high.pickle', 'wb') as handle:
        #         pickle.dump(data_new, handle)
        #
        ######################### random
        # for percentage_close in percentage_close_list:
        #     print('===============CURRENT percentage ' + str(percentage_close) + '=============')
        #     data_new = get_new_pax_data_random(data, route_stop_id, route_demand, bus_arr_info_df, percentage_close)
        #     with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(
        #             sample_seed) + '_close_bus_' + str(int(percentage_close * 100)) + 'random.pickle', 'wb') as handle:
        #         pickle.dump(data_new, handle)

        ######################### shutting by areas
        bus_regions  = pd.read_csv('../data/bus_region.txt', sep = ',')
        bus_regions = bus_regions.dropna()
        # drop the last few str records
        bus_regions = bus_regions.loc[bus_regions['BusStopCod'].apply(lambda x: 'N' not in x)]
        #
        bus_regions['BusStopCod'] = bus_regions['BusStopCod'].astype('int')
        bus_regions = bus_regions.merge(bus_stop_id_info, left_on = ['BusStopCod'], right_on =['Old_stop_id'])
        region_id = list(pd.unique(bus_regions['OBJECTID']))
        temp = pd.DataFrame({'region_id':region_id,'R0':[-1]*len(region_id)})
        temp.to_csv('../data/impact_cutting_bus_region.csv',index=False)
        for area_id in region_id:
            banned_bus_stop = bus_regions.loc[bus_regions['OBJECTID'] == area_id]
            banned_bus_stop = banned_bus_stop.loc[:,['OBJECTID','bus_stop']]
            print('===============CURRENT region ' + str(area_id) + '=============')
            data_new = get_new_pax_data_region(data, route_stop_id, route_demand, bus_arr_info_df, banned_bus_stop)
            with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(
                    sample_seed) + '_close_bus_' + 'region_' + str(int(area_id)) + 'random.pickle', 'wb') as handle:
                pickle.dump(data_new, handle)


    if GENERATE_CONTACT_TIME:
        ###################PERCENTAGE===============
        date_list = list(range(start_date, end_date + 1, 1))  #
        percentage_close_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for percentage_close in percentage_close_list:
            print('===============CURRENT percentage ' + str(percentage_close) + '=============')
            tic = time.time()
            scenario_name = 'low_high'
            with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size)
                      + '_seed_' + str(sample_seed) + '_close_bus_' + str(int(percentage_close * 100)) +  scenario_name + '.pickle', 'rb') as handle:
                data = pickle.load(handle)
            save_file_name = '_close_bus_' + str(int(percentage_close * 100)) + scenario_name

            print('load file time', time.time() - tic)
            PT_net_new = PT_connection(data, date_list, time_interval)
            contact_time_pt = cal_pt_cont_time(PT_net_new)
            print('finish_PT')
            contact_time_pt.to_csv('output/pt_total_contact_time_' + save_file_name + '.csv',index=False)
            #===============

        ###################REGION===============
        # bus_regions  = pd.read_csv('../data/bus_region.txt', sep = ',')
        # bus_regions = bus_regions.dropna()
        # # drop the last few str records
        # bus_regions = bus_regions.loc[bus_regions['BusStopCod'].apply(lambda x: 'N' not in x)]
        # #
        # bus_regions['BusStopCod'] = bus_regions['BusStopCod'].astype('int')
        # bus_regions = bus_regions.merge(bus_stop_id_info, left_on = ['BusStopCod'], right_on =['Old_stop_id'])
        # region_id = list(pd.unique(bus_regions['OBJECTID']))
        # date_list = list(range(start_date, end_date + 1, 1))  #
        # for area_id in region_id:
        #     print('===============CURRENT region ' + str(area_id) + '=============')
        #     tic = time.time()
        #     with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(sample_seed)
        #               + '_close_bus_' + 'region_' + str(int(area_id)) +  'random.pickle', 'rb') as handle:
        #         data = pickle.load(handle)
        #     save_file_name = '_close_bus_' + 'region_' + str(int(area_id))
        #     print('load file time', time.time() - tic)
        #     PT_net_new = PT_connection(data, date_list, time_interval)
        #     contact_time_pt = cal_pt_cont_time(PT_net_new)
        #     print('finish_PT')
        #     contact_time_pt.to_csv('output/pt_total_contact_time_' + save_file_name + '.csv',index=False)
            #===============






