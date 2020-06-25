import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import time
import pickle
import random
import matplotlib.pyplot as plt
import _constants
import _simulation_engine as sim_eng

# def get_new_departure_time(data, unit_move, date_range):
#     data_new = []
#     data['start_time_day'] = data['start_time'] + data['date'] * 86400
#     for day in range(date_range[0],date_range[1]+1):
#         data_date = data.loc[data['date']==day]
#         data_group = data_date.groupby(['boarding_stop'],sort=False)
#         count = 0
#         print('day',day)
#         # if day == 5:
#         #     break
#         for idx, info in data_group:
#             # count += 1
#             # print('current bus stop id', count, 'total 4500+')
#             busid_count = info.groupby(['bus_id'],sort=False)['P_id','start_time_day'].\
#                 agg({'P_id':'count','start_time_day':'mean'}).reset_index(drop = False)
#             busid_count = busid_count.rename(columns = {'P_id': 'num_people','start_time_day': 'bus_time'})
#             busid_count['key'] = 1
#             busid_count = busid_count.merge(busid_count, on = ['key'])
#             busid_count['time_diff'] = np.abs(busid_count['bus_time_x'] - busid_count['bus_time_y'])
#             busid_count = busid_count.loc[busid_count['time_diff']<unit_move] # get nearby bus
#             busid_count_new = busid_count.groupby(['bus_id_x','bus_time_x','num_people_x'])['num_people_y'].mean().reset_index(drop=False)
#             busid_count_new['change'] = busid_count_new['num_people_y'] - busid_count_new['num_people_x']
#             # avoid round not equal
#             busid_count_new['change'] = np.round(busid_count_new['change'])
#             busid_count_new['num_people_y'] = busid_count_new['num_people_x'] + busid_count_new['change']
#             if len(info) > sum(busid_count_new['num_people_y']):
#                 idx = random.sample(list(busid_count_new.index),int(len(info) - sum(busid_count_new['num_people_y'])))
#                 busid_count_new.loc[idx,'num_people_y'] += 1
#             elif len(info) < sum(busid_count_new['num_people_y']):
#                 idx = random.sample(list(busid_count_new.index),int(sum(busid_count_new['num_people_y']) - len(info) ))
#                 busid_count_new.loc[idx,'num_people_y'] -= 1
#             busid_count_new = busid_count_new.loc[:, ['bus_id_x', 'bus_time_x','num_people_x', 'num_people_y']]
#             busid_count_new = busid_count_new.rename(columns = {'bus_id_x':'bus_id','bus_time_x':'bus_time','num_people_y':'new_num_people'})
#             busid_count_new = busid_count_new.sort_values(['bus_time'])
#             info_new = info.loc[:,['P_id','start_time_day','ride_duration','boarding_stop','alighting_stop']]
#             info_new = info_new.sort_values(['start_time_day'])
#             bus_id_list =np.repeat(busid_count_new['bus_id'].values, busid_count_new['new_num_people'].astype('int').values)
#             bus_time_list = np.repeat(busid_count_new['bus_time'].values,
#                                     busid_count_new['new_num_people'].astype('int').values)
#             info_new['bus_id'] = bus_id_list
#             info_new['start_time_day'] = np.round(bus_time_list)
#             data_new.append(info_new)
#     data_new = pd.concat(data_new)
#     data_new = data_new.reset_index(drop=True)
#     data_new['date'] = data_new['start_time_day'] // 86400
#     data_new['start_time'] = data_new['start_time_day'] % 86400
#     data_new = data_new.drop(columns = ['start_time_day'])
#     return data_new
#
#
# def get_new_departure_time_new(data, unit_move, time_interval, date_range,bus_info_dict, K):
#     data_new = []
#     data['start_time_day'] = data['start_time'] + data['date'] * 86400
#     for day in range(date_range[0],date_range[1]+1):
#         if day == 5:
#             break
#         data_date = data.loc[data['date']==day]
#         data_date['time_int'] = data_date['start_time'] // time_interval
#         data_date['trip_id'] = range(len(data_date))
#         data_date_pax = pd.DataFrame({'time_int':list(range(0, 86400//time_interval)),'P_id':[0] * (86400//time_interval)})
#         temp = data_date.groupby(['time_int'])['P_id'].count()
#         data_date_pax = data_date_pax.set_index(['time_int'])
#         data_date_pax['P_id'] = temp
#         data_date_pax = data_date_pax.reset_index(drop=False).fillna(0)
#         # plt.plot(data_date_pax['time_int'], data_date_pax['P_id'], marker='o', markersize=6)
#         # plt.show()
#         first_zero = data_date_pax.loc[data_date_pax['P_id'] == 0].head(1).index[0]
#         last_zero = data_date_pax.loc[data_date_pax['P_id'] == 0].tail(1).index[0]
#
#         def move_avg(data_date_pax):
#             N = int(unit_move / time_interval)*2 + 1
#             if N > len(data_date_pax):
#                 N = len(data_date_pax)
#             new_pax_num = np.convolve(data_date_pax['P_id'], np.ones((N,)) / N, mode='same')
#             k = 0
#             while k < K:
#                 new_pax_num = np.convolve(new_pax_num, np.ones((N,)) / N, mode='same')
#                 k += 1
#             data_date_pax['new_pax_num'] = np.round(new_pax_num)
#             return data_date_pax
#
#         def revise_num(data_date_pax):
#             if sum(data_date_pax['P_id']) > sum(data_date_pax['new_pax_num']):
#                 uni_num = (sum(data_date_pax['P_id']) - sum(data_date_pax['new_pax_num'])) // len(data_date_pax)
#                 data_date_pax.loc[:, 'new_pax_num'] += uni_num
#                 left_num = (sum(data_date_pax['P_id']) - sum(data_date_pax['new_pax_num'])) % len(data_date_pax)
#                 idx = random.sample(list(data_date_pax.index), int(left_num))
#                 data_date_pax.loc[idx, 'new_pax_num'] += 1
#
#             elif sum(data_date_pax['P_id']) < sum(data_date_pax['new_pax_num']):
#                 uni_num = (sum(data_date_pax['P_id']) - sum(data_date_pax['new_pax_num'])) // len(data_date_pax)
#                 data_date_pax.loc[:, 'new_pax_num'] -= uni_num
#                 left_num = (sum(data_date_pax['P_id']) - sum(data_date_pax['new_pax_num'])) % len(data_date_pax)
#                 idx = random.sample(list(data_date_pax.index), int(left_num))
#                 data_date_pax.loc[idx, 'new_pax_num'] -= 1
#             assert sum(data_date_pax['P_id']) == sum(data_date_pax['new_pax_num'])
#             return data_date_pax
#         data_date_pax1 = move_avg(data_date_pax.iloc[:first_zero,:])
#         data_date_pax2 = move_avg(data_date_pax.iloc[last_zero:, :])
#         data_date_pax1 = revise_num(data_date_pax1)
#         data_date_pax2 = revise_num(data_date_pax2)
#         data_date_pax3 = data_date_pax.iloc[first_zero: last_zero, :]
#         data_date_pax = pd.concat([data_date_pax1, data_date_pax2,data_date_pax3])
#         data_date_pax = data_date_pax.sort_values(['time_int']).fillna(0)
#         data_date_pax = revise_num(data_date_pax)
#         assert sum(data_date_pax['P_id']) == sum(data_date_pax['new_pax_num'])
#         # plt.plot(data_date_pax['time_int'], data_date_pax['new_pax_num'], marker='*', markersize=6)
#         # plt.show()
#
#         data_date_new = data_date.loc[:,['P_id','trip_id','start_time','date','ride_duration','boarding_stop','alighting_stop']]
#         data_date_new = data_date_new.sort_values(['start_time'])
#         new_pax_num = data_date_pax['new_pax_num'].values
#         new_time_list = []
#         for time_int in range(0, 86400 // time_interval):
#             new_time_list += list(np.linspace(time_int * time_interval + 1, (time_int + 1) * time_interval, num=int(new_pax_num[time_int])))
#         assert len(data_date) == len(new_time_list)
#         data_date_new['start_time'] = new_time_list
#         assert len(data_date_new) == len(new_time_list)
#         #===========get bus info
#         bus_info = bus_info_dict[day]
#         bus_info = bus_info.rename(columns = {'start_time':'bus_time'})
#         max_headway = 30*60
#         bus_info['time_int'] = bus_info['bus_time'] // max_headway
#         data_date_new['time_int'] = data_date_new['start_time'] // max_headway
#         data_date_new = data_date_new.merge(bus_info, on = ['boarding_stop','time_int'], how ='left')
#         data_date_new_nona = data_date_new.loc[~data_date_new['bus_time'].isna()]
#         data_date_new_na = data_date_new.loc[data_date_new['bus_time'].isna()]
#         data_date_new_na = data_date_new_na.drop(columns = ['bus_time','bus_id','time_int'])
#         data_date_new_na = data_date_new_na.merge(bus_info, on = ['boarding_stop'], how ='left')
#         data_date_new = pd.concat([data_date_new_nona, data_date_new_na])
#         data_date_new['time_diff'] = np.abs(data_date_new['bus_time'] - data_date_new['start_time'])
#         data_date_new = data_date_new.sort_values(['time_diff'])
#         data_date_new = data_date_new.drop_duplicates(['P_id','trip_id'], keep = 'first')
#         data_date_new['start_time'] = data_date_new['bus_time']
#         assert len(data_date_new) == len(data_date)
#         data_date_new = data_date_new.drop(columns = ['bus_time','time_int','trip_id','time_diff'])
#         data_new.append(data_date_new)
#
#     return pd.concat(data_new)



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



def construct_contact_net(data, time_int_s, time_int_e, start_time_name,end_time_name, od_name):
    #['P_id', 'bus_id', 'date', 'start_time', 'ride_duration', 'boarding_stop', 'alighting_stop']
    data = data.rename(columns = {start_time_name: 'start_time', end_time_name: 'end_time'})
    data_net = data[['P_id',od_name,'start_time',
                     'end_time']].merge(data[['P_id',od_name,'start_time','end_time']],on = [od_name])
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
    data_net['contact_duration'] = data_net['contact_duration']*0.5 # assume only stay half of the time at an O or D
    data_net['contact_duration'] = data_net['contact_duration'].apply(round)
    data_net['contact_duration'] = data_net['contact_duration'].astype('int')
    data_net = data_net.loc[data_net['P_id_x']!=data_net['P_id_y']]
    # data_net['w_ij'] = data_net['contact_duration'] / (time_int_e - time_int_s)
    # a=1
    return data_net[['P_id_x','P_id_y','contact_duration']]

def local_contact(data, date_list, time_interval):
    # process OD
    data['start_time_day'] = data['start_time'] + data['date'] * 3600*24
    data['end_time_day'] = data['start_time_day'] + data['ride_duration']
    data = data.drop(columns = ['start_time'])
    tic = time.time()
    data = data.sort_values(['P_id','start_time_day'])
    # print('sort time', time.time() - tic)
    tic = time.time()
    data['trip_seq'] = data.groupby(['P_id']).cumcount()
    # print('group time', time.time() - tic)
    data['trip_seq_next'] = data['trip_seq'] - 1
    data['trip_seq_last'] = data['trip_seq'] + 1
    data['o_end_time'] = data['start_time_day']
    data['d_start_time'] = data['end_time_day']
    tic = time.time()
    data = data.merge(data[['P_id', 'trip_seq_next', 'start_time_day']], left_on = ['P_id', 'trip_seq'], right_on = ['P_id', 'trip_seq_next'])
    data = data.rename(columns = {'start_time_day_y':'d_end_time'})
    # print('merge 1 time', time.time() - tic)
    tic = time.time()
    data = data.merge(data[['P_id', 'trip_seq_last', 'end_time_day']], left_on = ['P_id', 'trip_seq'], right_on = ['P_id', 'trip_seq_last'])
    data = data.rename(columns = {'end_time_day_y':'o_start_time'})
    # print('merge 2 time', time.time() - tic)
    data = data.loc[:,['P_id','date','o_start_time','o_end_time','d_start_time','d_end_time','boarding_stop','alighting_stop']]
    # filter unreasonable data (due to unaccurate record or cross days)
    old_len = len(data)
    data = data.loc[(data['o_start_time']<data['o_end_time']) & (data['d_start_time']<data['d_end_time'])] # due to the records errors in the AFC data
    # print('Num time records errors', len(data) - old_len)
    #==== trip sequence define as consecutive trips within 24 hours
    # print('Total trip seq records', len(data))
    data = data.loc[(data['o_end_time'] - data['o_start_time']< 24*3600)]
    data = data.loc[(data['d_end_time'] - data['d_start_time']< 24*3600)]
    # print('After 24 h threshold trip seq records', len(data))
    #=======
    time_int_id = 0
    local_contact_net = {}
    for date in date_list:
        data_date = data.loc[data['date'] == date]
        for time_int in range(0, 86400, time_interval):
            time_int_id += 1
            print('current time interval id:', time_int_id)
            # give data for one time interval
            time_int_s = date*3600*24 + time_int
            time_int_e = date*3600*24  + time_int + time_interval

            ########### o connection
            data_time_int_1 = data_date.loc[~((data_date['o_start_time'] > time_int_e)|
                                          (data_date['o_end_time']< time_int_s))]
            data_time_int_1 = data_time_int_1.rename(columns = {'o_start_time':'station_start_time','o_end_time':'station_end_time',
                                                                'boarding_stop':'station_id'})
            data_time_int_2 = data_date.loc[~((data_date['d_start_time'] > time_int_e) |
                                            (data_date['d_end_time'] < time_int_s))]
            data_time_int_2 = data_time_int_2.rename(
                columns={'d_start_time': 'station_start_time', 'd_end_time': 'station_end_time',
                         'alighting_stop':'station_id'})
            data_time_int = pd.concat([data_time_int_1,data_time_int_2],sort = False)
            data_time_int = data_time_int.drop(columns = ['o_start_time','o_end_time','d_start_time','d_end_time'])
            if len(data_time_int) > 0:
                data_net = construct_contact_net(data_time_int, time_int_s, time_int_e,
                                                   start_time_name = 'station_start_time',end_time_name ='station_end_time',
                                                   od_name ='station_id')
            else:
                data_net = pd.DataFrame({'P_id_x':[],'P_id_y':[],'contact_duration':[]})
            ########## merge to together
            data_net = data_net.groupby(['P_id_x','P_id_y'])['contact_duration'].sum().reset_index()
            data_net.loc[data_net['contact_duration']>time_interval,'contact_duration'] = time_interval # should not happen, just for conservative consideration.
            data_net['time_int_id'] = time_int_id
            local_contact_net[time_int_id] = data_net
    # print('Finish local contact network')
    return local_contact_net


def get_new_departure_time_new3(data, unit_move, time_interval, date_range,bus_info_dict, K):
    data_new = []
    data['start_time_day'] = data['start_time'] + data['date'] * 86400
    for day in range(date_range[0],date_range[1]+1):
        # if day == 5:
        #     break
        data_date = data.loc[data['date']==day]
        data_date['time_int'] = data_date['start_time'] // time_interval
        data_date['trip_id'] = range(len(data_date))
        data_date_pax = pd.DataFrame({'time_int':list(range(0, 86400//time_interval)),'P_id':[0] * (86400//time_interval)})
        temp = data_date.groupby(['time_int'])['P_id'].count()
        data_date_pax = data_date_pax.set_index(['time_int'])
        data_date_pax['P_id'] = temp
        data_date_pax = data_date_pax.reset_index(drop=False).fillna(0)
        # plt.plot(data_date_pax['time_int'], data_date_pax['P_id'], marker='o', markersize=6)
        # plt.show()

        def move_avg(data_date_pax):
            N = int(unit_move / time_interval)*2 + 1
            if N > len(data_date_pax):
                N = len(data_date_pax)
            new_pax_num = np.convolve(data_date_pax['P_id'], np.ones((N,)) / N, mode='same')
            k = 0
            while k < K:
                new_pax_num = np.convolve(new_pax_num, np.ones((N,)) / N, mode='same')
                k += 1
            data_date_pax['new_pax_num'] = np.round(new_pax_num)
            return data_date_pax

        def revise_num(data_date_pax):
            if sum(data_date_pax['P_id']) > sum(data_date_pax['new_pax_num']):
                uni_num = (sum(data_date_pax['P_id']) - sum(data_date_pax['new_pax_num'])) // len(data_date_pax)
                data_date_pax.loc[:, 'new_pax_num'] += uni_num
                left_num = (sum(data_date_pax['P_id']) - sum(data_date_pax['new_pax_num'])) % len(data_date_pax)
                idx = random.sample(list(data_date_pax.index), int(left_num))
                data_date_pax.loc[idx, 'new_pax_num'] += 1

            elif sum(data_date_pax['P_id']) < sum(data_date_pax['new_pax_num']):
                uni_num = (sum(data_date_pax['P_id']) - sum(data_date_pax['new_pax_num'])) // len(data_date_pax)
                data_date_pax.loc[:, 'new_pax_num'] -= uni_num
                left_num = (sum(data_date_pax['P_id']) - sum(data_date_pax['new_pax_num'])) % len(data_date_pax)
                idx = random.sample(list(data_date_pax.index), int(left_num))
                data_date_pax.loc[idx, 'new_pax_num'] -= 1
            assert sum(data_date_pax['P_id']) == sum(data_date_pax['new_pax_num'])
            return data_date_pax
        start_value = 35 # to avoid moving non-trip time (0:00 - 6:00)
        data_date_pax1 = move_avg(data_date_pax.iloc[start_value:,:])
        data_date_pax2 = data_date_pax.iloc[:start_value, :]
        data_date_pax1 = revise_num(data_date_pax1)
        data_date_pax = pd.concat([data_date_pax1, data_date_pax2])
        data_date_pax = data_date_pax.sort_values(['time_int']).fillna(0)
        data_date_pax = revise_num(data_date_pax)
        assert sum(data_date_pax['P_id']) == sum(data_date_pax['new_pax_num'])
        # plt.plot(data_date_pax['time_int'], data_date_pax['new_pax_num'], marker='*', markersize=6)
        # plt.show()

        data_date_new = data_date.loc[:,['P_id','trip_id','start_time','date','ride_duration','boarding_stop','alighting_stop']]
        data_date_new = data_date_new.sort_values(['start_time'])
        new_pax_num = data_date_pax['new_pax_num'].values
        new_time_list = []
        for time_int in range(0, 86400 // time_interval):
            new_time_list += list(np.linspace(time_int * time_interval + 1, (time_int + 1) * time_interval, num=int(new_pax_num[time_int])))
        assert len(data_date) == len(new_time_list)
        data_date_new['start_time'] = new_time_list
        assert len(data_date_new) == len(new_time_list)
        #===========get bus info
        bus_info = bus_info_dict[day]
        bus_info = bus_info.rename(columns = {'start_time':'bus_time'})
        max_headway = 30*60
        bus_info['time_int'] = bus_info['bus_time'] // max_headway
        data_date_new['time_int'] = data_date_new['start_time'] // max_headway
        data_date_new = data_date_new.merge(bus_info, on = ['boarding_stop','time_int'], how ='left')
        data_date_new_nona = data_date_new.loc[~data_date_new['bus_time'].isna()]
        data_date_new_na = data_date_new.loc[data_date_new['bus_time'].isna()]
        data_date_new_na = data_date_new_na.drop(columns = ['bus_time','bus_id','time_int'])
        data_date_new_na = data_date_new_na.merge(bus_info, on = ['boarding_stop'], how ='left')
        data_date_new = pd.concat([data_date_new_nona, data_date_new_na])
        data_date_new['time_diff'] = np.abs(data_date_new['bus_time'] - data_date_new['start_time'])
        data_date_new = data_date_new.sort_values(['time_diff'])
        data_date_new = data_date_new.drop_duplicates(['P_id','trip_id'], keep = 'first')
        data_date_new['start_time'] = data_date_new['bus_time']
        assert len(data_date_new) == len(data_date)
        data_date_new = data_date_new.drop(columns = ['bus_time','time_int','trip_id','time_diff'])
        data_new.append(data_date_new)

    return pd.concat(data_new)

#
# def get_new_departure_time_new2(data, unit_move, time_interval, date_range,bus_info_dict, K):
#     data_new = []
#     data['start_time_day'] = data['start_time'] + data['date'] * 86400
#     for day in range(date_range[0],date_range[1]+1):
#         # if day == 5:
#         #     break
#         data_date = data.loc[data['date']==day]
#         data_date['trip_id'] = range(len(data_date))
#         data_date_new = data_date.loc[:,['P_id','trip_id','start_time','date','ride_duration','boarding_stop','alighting_stop']]
#         data_date_new['time_int'] = data_date_new['start_time'] // unit_move
#         bus_info = bus_info_dict[day]
#         bus_info = bus_info.rename(columns={'start_time': 'bus_time'})
#         bus_info['time_int'] = bus_info['bus_time'] // unit_move
#         bus_pax1 = data_date_new.merge(bus_info, on = ['boarding_stop','time_int'])
#         bus_pax1['time_diff'] = np.abs(bus_pax1['start_time'] - bus_pax1['bus_time'])
#         bus_pax1 = bus_pax1.loc[bus_pax1['time_diff']<unit_move]
#         bus_info['time_int'] -= 1
#         bus_pax2 = data_date_new.merge(bus_info, on = ['boarding_stop','time_int'])
#         bus_pax2['time_diff'] = np.abs(bus_pax2['start_time'] - bus_pax2['bus_time'])
#         bus_pax2 = bus_pax2.loc[bus_pax2['time_diff']<unit_move]
#         bus_info['time_int'] += 2
#         bus_pax3 = data_date_new.merge(bus_info, on = ['boarding_stop','time_int'])
#         bus_pax3['time_diff'] = np.abs(bus_pax3['start_time'] - bus_pax2['bus_time'])
#         bus_pax3 = bus_pax3.loc[bus_pax3['time_diff']<unit_move]
#         bus_pax = pd.concat([bus_pax1,bus_pax2,bus_pax3])
#         bus_pax = bus_pax.drop_duplicates(['P_id','trip_id','bus_id'])
#         bus_pax['num_avai_bus'] = bus_pax.groupby(['P_id','trip_id'], sort=False)['bus_id'].transform('count')
#         bus_pax['pax_contribution'] = 1/bus_pax['num_avai_bus']
#         busid_count_new = bus_pax.groupby(['boarding_stop', 'bus_id','bus_time'])['pax_contribution'].sum().reset_index()
#         # print(sum(busid_count_new['pax_contribution']))
#         # print(len(data_date))
#         busid_count_new['new_num_pax'] = np.round(busid_count_new['pax_contribution'])
#         busid_count_new_no_pax = busid_count_new.loc[busid_count_new['new_num_pax']==0]
#         busid_count_new_with_pax = busid_count_new.loc[busid_count_new['new_num_pax'] > 0]
#         busid_count_new_no_pax = busid_count_new_no_pax.sort_values(['pax_contribution'],ascending=False)
#         lost_pax = len(data_date) - sum(busid_count_new['new_num_pax'])
#         busid_count_new_no_pax = busid_count_new_no_pax.iloc[:int(lost_pax)]
#         busid_count_new_no_pax['new_num_pax'] = 1
#         busid_count_new = pd.concat([busid_count_new_with_pax, busid_count_new_no_pax])
#         assert sum(busid_count_new['new_num_pax']) == len(data_date)
#         busid_count_new = busid_count_new.sort_values(['boarding_stop','bus_time'])
#         data_date_new = data_date_new.sort_values(['boarding_stop','start_time'])
#
#         bus_id_list =np.repeat(busid_count_new['bus_id'].values, busid_count_new['new_num_pax'].astype('int').values)
#         bus_time_list = np.repeat(busid_count_new['bus_time'].values,
#                                 busid_count_new['new_num_pax'].astype('int').values)
#         data_date_new['bus_id'] = bus_id_list
#         data_date_new['start_time'] = np.round(bus_time_list)
#         data_date_new = data_date_new.drop(columns=['time_int', 'trip_id'])
#         data_new.append(data_date_new)
#
#     return pd.concat(data_new)

def test_distribution_demand(data, data_old):
    time_interval = 3600
    data['time_interval'] = data['start_time'] // time_interval
    data = data.sort_values(['start_time'])
    print(len(data))
    print(len(data_old))
    data_demand = data.groupby(['date', 'time_interval'])['P_id'].count().reset_index(drop=False)
    data_old['time_interval'] = data_old['start_time'] // time_interval
    data_demand2 = data_old.groupby(['date', 'time_interval'])['P_id'].count().reset_index(drop=False)
    plt.plot(data_demand['time_interval'],data_demand['P_id'], marker = 'o', markersize = 6,color = 'r',label = 'new')
    plt.plot(data_demand2['time_interval'], data_demand2['P_id'], marker='o', markersize=6,color = 'b',label = 'old')
    plt.legend()
    plt.show()

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


def cal_global_cont_time(total_time_int, time_interval):
    num_global = int(round(_constants.theta_g * sample_size * sample_size))
    np.random.seed(3)
    global_net_int = pd.DataFrame({'P_id_x': np.random.choice(range(0, sample_size), num_global),
                                   'P_id_y': np.random.choice(range(0, sample_size), num_global),
                                   'w_ij': 1})
    global_net_int = global_net_int.loc[global_net_int['P_id_x'] != global_net_int['P_id_y']]


    total_contact_time_list_global = [len(global_net_int) * time_interval] * total_time_int
    total_contact_time_square_list_global = [len(global_net_int) * time_interval * time_interval]  * total_time_int
    time_int_list = list(range(1, total_time_int + 1 , 1))

    contact_time = pd.DataFrame({'time_interval': time_int_list, 'total_contact_time': total_contact_time_list_global,
                                 'total_contact_time_square': total_contact_time_square_list_global})
    return contact_time

def cal_local_cont_time(local_net):
    total_contact_time_list_local = []
    total_contact_time_square_list_local = []
    time_int_list = []
    for time_int in local_net:
        time_int_list.append(time_int)
        local_net_int_original = local_net[time_int]
        # =================== randomly assign local connection
        num_local = int(round(_constants.theta_l * len(local_net_int_original)))
        seed = 0
        np.random.seed(seed)
        local_net_idx = np.random.choice(local_net_int_original.index, num_local)
        local_net_int = local_net_int_original.loc[local_net_idx, :]
        total_contact_time_list_local.append(local_net_int['contact_duration'].sum())
        total_contact_time_square_list_local.append(np.sum(np.square(local_net_int['contact_duration'])))
    # ============================
    contact_time = pd.DataFrame({'time_interval': time_int_list, 'total_contact_time': total_contact_time_list_local,
                                 'total_contact_time_square': total_contact_time_square_list_local})
    return contact_time

def test_distribution_pt_time(data, data_old):
    date_list = [4]
    time_interval = 3600
    PT_net_new = PT_connection(data, date_list, time_interval)
    contact_time_new = cal_pt_cont_time(PT_net_new)
    PT_net_old = PT_connection(data_old, date_list, time_interval)
    contact_time_old = cal_pt_cont_time(PT_net_old)

    local_net_new = local_contact(data, date_list, time_interval)
    contact_time_new_local = cal_local_cont_time(local_net_new)
    local_net_old = local_contact(data_old, date_list, time_interval)
    contact_time_old_local = cal_local_cont_time(local_net_old)

    plt.plot(contact_time_new['time_interval'].iloc[:24], contact_time_new['total_contact_time'].iloc[:24], label='new')
    plt.plot(contact_time_old['time_interval'].iloc[:24], contact_time_old['total_contact_time'].iloc[:24], label='old pt')
    print('total_cont_time_new',sum(contact_time_new['total_contact_time']))
    print('total_cont_time_old', sum(contact_time_old['total_contact_time']))
    plt.legend()
    plt.show()


    plt.plot(contact_time_new_local['time_interval'].iloc[:24], contact_time_new_local['total_contact_time'].iloc[:24], label='new')
    plt.plot(contact_time_old_local['time_interval'].iloc[:24], contact_time_old_local['total_contact_time'].iloc[:24], label='old local')
    print('total_cont_time_new',sum(contact_time_new_local['total_contact_time']))
    print('total_cont_time_old', sum(contact_time_old_local['total_contact_time']))
    plt.legend()
    plt.show()

def single_testing(pt_net, local_net, sample_pax, save_file_name, time_interval, sample_size, seed):
    start_time_int = 0
    sim_period = round(1*len(pt_net) - start_time_int)
    mu_r = _constants.mu_r
    mu_d = _constants.mu_d
    beta_I = _constants.beta_I
    beta_E = _constants.beta_E
    theta_l = _constants.theta_l
    theta_g = _constants.theta_g
    gamma = _constants.gamma

    para_dict = {'mu_r': mu_r, 'mu_d':mu_d, 'beta_I': beta_I,'beta_E': beta_E,
                 'theta_l':theta_l,'theta_g':theta_g,'gamma':gamma,'initial_I':_constants.initial_I}
    simulation_seed = 0
    sim = sim_eng.simulation(pt_net, sim_period, local_net, para_dict, sample_size,
                             time_interval, sample_pax, start_time_int, save_file_name,
                             verbal = False, random_seed = simulation_seed)
    tic = time.time()
    sim.run()
    running_time = round(time.time() - tic,1)
    df_test_info = {'test_name': [save_file_name], 'sample_size':[sample_size],'time_interval':[time_interval],
                    'sample_seed':[seed],'sim_seed':[simulation_seed],'start_time_int':[start_time_int],
                    'total_sim_period':[sim_period]}
    for para in para_dict:
        df_test_info[para] = [para_dict[para]]
    df_test_info['running_time'] = [running_time]
    df_test_info = pd.DataFrame(df_test_info)
    df_test_info.to_csv('output/simulation_para_' +  save_file_name + '.csv',index=False)

def simulation():
    time_interval = 1*3600 # 1 hour
    sample_size = 100000
    seed = 0
    unit_move_list = [10,30,50,70,90,110]
    # unit_move_list = [110]
    for unit_move in unit_move_list:
        print('current spreading', unit_move)
        unit_move = unit_move * 60 # min
        tic = time.time()
        with open('../data/PT_eco_net_' + str(sample_size) +'_seed_' + str(seed) + '_spreading_' + str(int(unit_move/60)) + '.pickle', 'rb') as handle:
            pt_net = pickle.load(handle)
        with open('../data/Local_contact_net_'+ str(sample_size) + '_seed_' + str(seed) + '_spreading_' + str(int(unit_move/60)) + '.pickle', 'rb') as handle:
            local_net = pickle.load(handle)
        with open('../data/sample_pax_'+ str(sample_size) + '_seed_' + str(seed) + '.pickle', 'rb') as handle:
            sample_pax = pickle.load(handle)
        print('load data time', time.time() - tic)
        save_file_name = 'spreading' + str(int(unit_move/60))
        single_testing(pt_net, local_net, sample_pax, save_file_name, time_interval, sample_size, seed)



if __name__ == '__main__':
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31
    sample_size = 100000
    sample_seed = 0
    TEST = True
    GENERATE_SAMPLE = True
    GENERATE_CONTACT_TIME = True
    unit_move_list = [110]
    with open('../data/data_bus_arrival_info' + str(start_date) + '_' + str(end_date) + '.pickle',
              'rb') as handle:
        bus_info_dict = pickle.load(handle)

    ####################
    # unit_move = 90 * 60
    # with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(
    #         sample_seed) + '_spreading_' + str(int(unit_move / 60)) + '.pickle', 'rb') as handle:
    #     data_new = pickle.load(handle)
    # with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(
    #         sample_seed) + '.pickle', 'rb') as handle:
    #     data = pickle.load(handle)
    # test_distribution(data_new.loc[data_new['date'] == 4], data.loc[data['date'] == 4])
    ######################

    if GENERATE_SAMPLE:
        for unit_move in unit_move_list:
            unit_move = unit_move * 60 # min
            print('current unit', unit_move)
            if TEST:
                tic = time.time()
                with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(sample_seed) +'.pickle', 'rb') as handle:
                    data = pickle.load(handle)
                print('load time', time.time() - tic)
                time_interval = 10
                data_new = get_new_departure_time_new3(data, unit_move,time_interval * 60, [start_date,end_date],bus_info_dict, K = 6) # k num of moving avg
                # test_distribution_demand(data_new, data.loc[data['date'] == 4])
                # test_distribution_pt_time(data_new, data.loc[data['date'] == 4])
                with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(sample_seed) +'_spreading_' + str(int(unit_move/60)) + '.pickle', 'wb') as handle:
                    pickle.dump(data_new, handle)
            print('==============================')
        # print('Please run 02_to generate net first before running simulation')

    if GENERATE_CONTACT_TIME:
        unit_move_list = [110]
        date_list = list(range(start_date, end_date + 1, 1))  #
        # date_list = [4]
        time_interval = 3600
        for unit_move in unit_move_list:
            save_file_name = 'spreading_' + str(int(unit_move))

            print('===============CURRENT UNIT ' + str(unit_move) + '=============')
            unit_move = unit_move * 60  # min
            tic = time.time()
            with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(sample_seed) +'_spreading_' + str(int(unit_move/60)) + '.pickle', 'rb') as handle:
                data = pickle.load(handle)
            print('load file time', time.time() - tic)
            PT_net_new = PT_connection(data, date_list, time_interval)
            contact_time_pt = cal_pt_cont_time(PT_net_new)
            print('finish_PT')
            local_net_new = local_contact(data, date_list, time_interval)
            contact_time_local = cal_local_cont_time(local_net_new)
            print('finish_local')
            total_time_int = len(contact_time_local)
            contact_time_global = cal_global_cont_time(total_time_int,time_interval)
            print('finish_global')
            contact_time_pt.to_csv('output/pt_total_contact_time_' + save_file_name + '.csv',index=False)
            #===============
            contact_time_local.to_csv('output/local_total_contact_time_' + save_file_name + '.csv',index=False)
            # ===============
            contact_time_global.to_csv('output/global_total_local_contact_time_' + save_file_name + '.csv',index=False)








