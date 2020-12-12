import pandas as pd
import numpy as np
import time
import pickle

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

def PT_connection(data,date_list,time_interval,sample_size,seed, save_file_name):
    data['start_time_day'] = data['start_time'] + data['date'] * 3600*24
    data['end_time_day'] = data['start_time_day'] + data['ride_duration']
    time_int_id = 0
    PT_net = {}
    for date in date_list:
        data_date = data.loc[data['date'] == date]
        for time_int in range(0, 86400, time_interval):
            print('current time interval id:', time_int_id)
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


    with open(save_file_name, 'wb') as handle:
        pickle.dump(PT_net, handle)
    print('Finish PT encounter network')

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

def local_contact(data, date_list, time_interval,sample_size,seed, save_file_name):
    # process OD
    data['start_time_day'] = data['start_time'] + data['date'] * 3600*24
    data['end_time_day'] = data['start_time_day'] + data['ride_duration']
    data = data.drop(columns = ['start_time'])
    tic = time.time()
    data = data.sort_values(['P_id','start_time_day'])
    print('sort time', time.time() - tic)
    tic = time.time()
    data['trip_seq'] = data.groupby(['P_id']).cumcount()
    print('group time', time.time() - tic)
    data['trip_seq_next'] = data['trip_seq'] - 1
    data['trip_seq_last'] = data['trip_seq'] + 1
    data['o_end_time'] = data['start_time_day']
    data['d_start_time'] = data['end_time_day']
    tic = time.time()
    data = data.merge(data[['P_id', 'trip_seq_next', 'start_time_day']], left_on = ['P_id', 'trip_seq'], right_on = ['P_id', 'trip_seq_next'])
    data = data.rename(columns = {'start_time_day_y':'d_end_time'})
    print('merge 1 time', time.time() - tic)
    tic = time.time()
    data = data.merge(data[['P_id', 'trip_seq_last', 'end_time_day']], left_on = ['P_id', 'trip_seq'], right_on = ['P_id', 'trip_seq_last'])
    data = data.rename(columns = {'end_time_day_y':'o_start_time'})
    print('merge 2 time', time.time() - tic)
    data = data.loc[:,['P_id','date','o_start_time','o_end_time','d_start_time','d_end_time','boarding_stop','alighting_stop']]
    # filter unreasonable data (due to unaccurate record or cross days)
    old_len = len(data)
    data = data.loc[(data['o_start_time']<data['o_end_time']) & (data['d_start_time']<data['d_end_time'])] # due to the records errors in the AFC data
    print('Num time records errors', len(data) - old_len)
    #==== trip sequence define as consecutive trips within 24 hours
    print('Total trip seq records', len(data))
    data = data.loc[(data['o_end_time'] - data['o_start_time']< 24*3600)]
    data = data.loc[(data['d_end_time'] - data['d_start_time']< 24*3600)]
    print('After 24 h threshold trip seq records', len(data))
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

    with open(save_file_name, 'wb') as handle:
        pickle.dump(local_contact_net, handle)
    print('Finish local contact network')

def generate_sample_data(data_all, start_date, end_date, sample_size, seed):
    data = data_all.loc[(data_all['date'] >= start_date) & (data_all['date'] <= end_date)]
    passenger_list = pd.unique(data['P_id'])
    np.random.seed(seed)
    used_p = np.random.choice(passenger_list, sample_size)
    data = data.set_index(['P_id'])
    data = data.loc[used_p,:]
    data = data.reset_index(drop=False)
    old_len = len(data)
    data = data.drop_duplicates()
    print('Duplicate records',old_len - len(data))
    with open('../data/sample_pax_' + str(sample_size) + '_seed_' + str(seed) + '.pickle', 'wb') as handle:
        pickle.dump(used_p, handle)
    with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(seed) + '.pickle', 'wb') as handle:
        pickle.dump(data, handle)

if __name__ == '__main__':
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31
    sample_size = 100000 #100000
    sample_seed = 0
    GENERATE_DATA = False
    if GENERATE_DATA:
        tic = time.time()
        data_all = pd.read_csv('../data/data_Aug_compressed.csv')
        print('load data time', time.time() - tic)
        tic = time.time()
        generate_sample_data(data_all, start_date, end_date, sample_size, sample_seed)
        print('generate sample data time', time.time() - tic)
    else:
        tic = time.time()
        ##################
        with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(sample_seed) +'.pickle', 'rb') as handle:
            data = pickle.load(handle)
        print('load time', time.time() - tic)
        date_list = list(range(start_date,end_date+1,1)) #
        save_file_name = '../data/PT_eco_net_' + str(sample_size) + '_seed_' + str(seed) + '.pickle'
        PT_connection(data, date_list, time_interval, sample_size, sample_seed,save_file_name)
        print('==================================')
        save_file_name = '../data/Local_contact_net_' + str(sample_size) + '_seed_' + str(seed) + '.pickle'
        local_contact(data, date_list, time_interval, sample_size, sample_seed,save_file_name)


        # ################## TEST FOR spreading
        # unit_move_list = [10,30,50,70,90,110]
        # for unit_move in unit_move_list:
        #     print('===============CURRENT UNIT ' + str(unit_move) + '=============')
        #     unit_move = unit_move * 60  # min
        #     with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(sample_seed) +'_spreading_' + str(int(unit_move/60)) + '.pickle', 'rb') as handle:
        #         data = pickle.load(handle)
        #     print('load time', time.time() - tic)
        #     date_list = list(range(start_date,end_date+1,1)) #
        #     save_file_name = '../data/PT_eco_net_' + str(sample_size) + '_seed_' + str(sample_seed) + '_spreading_' + str(int(unit_move/60)) + '.pickle'
        #     PT_connection(data, date_list, time_interval, sample_size, sample_seed, save_file_name)
        #     print('==================================')
        #     save_file_name = '../data/Local_contact_net_' + str(sample_size) + '_seed_' + str(sample_seed) +'_spreading_' + str(int(unit_move/60)) +  '.pickle'
        #     local_contact(data, date_list, time_interval, sample_size, sample_seed, save_file_name)
        #     print('==================================')




