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

import networkx as nx


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


def get_core_info(pt_net_day):
    pt_net_day = pd.concat(pt_net_day)
    pt_net_day = pt_net_day.drop_duplicates(['P_id_x','P_id_y'])
    G = nx.from_pandas_edgelist(pt_net_day, source='P_id_x', target = 'P_id_y')
    core_number = nx.core_number(G)
    core_num_df = pd.DataFrame.from_dict(core_number, 'index')
    core_num_df = core_num_df.reset_index()
    core_num_df = core_num_df.rename(columns = {'index':'P_id',0:'core_numbers'})
    core_num_df = core_num_df.sort_values(['core_numbers'],ascending=False)
    core_num_group = core_num_df.groupby(['core_numbers'])['P_id'].count()

def get_delete_pax_id(pt_net_day, k_num):
    pt_net_day = pd.concat(pt_net_day)
    pt_net_day = pt_net_day.drop_duplicates(['P_id_x','P_id_y'])
    G = nx.from_pandas_edgelist(pt_net_day, source='P_id_x', target = 'P_id_y')
    core_number = nx.core_number(G)
    core_num_df = pd.DataFrame.from_dict(core_number, 'index')
    core_num_df = core_num_df.reset_index()
    core_num_df = core_num_df.rename(columns = {'index':'P_id',0:'core_numbers'})
    core_num_df = core_num_df.sort_values(['core_numbers'],ascending=False)
    # core_num_group = core_num_df.groupby(['core_numbers'])['P_id'].count()
    # print('old pax', len(core_num_df))
    delete_pax = core_num_df.loc[core_num_df['core_numbers'] >= k_num]
    # print('new pax', len(used_pax))
    return delete_pax

def get_net_sample_pax(delete_pax, data_date):
    delete_pax = list(pd.unique(delete_pax['P_id']))
    num_delete = len(delete_pax)
    all_pax = list(pd.unique(data_date['P_id']))
    random.seed(num_delete)
    random_delete = random.sample(all_pax, num_delete)
    data_date_process = data_date.copy()
    data_date_process = data_date_process.set_index(['P_id'])
    remain_pax = list(set(all_pax).difference(delete_pax))
    remain_pax_random = list(set(all_pax).difference(random_delete))
    # print(len(remain_pax))
    data_date_remain = data_date_process.loc[remain_pax]
    data_date_remain_random = data_date_process.loc[remain_pax_random]
    data_date_remain = data_date_remain.reset_index(drop=False)
    data_date_remain_random = data_date_remain_random.reset_index(drop=False)
    # print(len(data_date_remain))
    return data_date_remain, data_date_remain_random

if __name__ == '__main__':
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31
    sample_size = 100000
    sample_seed = 0
    TEST = True
    GENERATE_SAMPLE = False
    GENERATE_CONTACT_TIME = True

    if GENERATE_SAMPLE:
        # load net
        tic = time.time()
        with open('../data/PT_eco_net_' + str(sample_size) + '_seed_' + str(sample_seed) + '.pickle', 'rb') as handle:
            pt_net = pickle.load(handle)
        with open('../data/Local_contact_net_' + str(sample_size) + '_seed_' + str(sample_seed) + '.pickle', 'rb') as handle:
            local_net = pickle.load(handle)
        with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(sample_seed) +'.pickle', 'rb') as handle:
            data = pickle.load(handle)
        print('load data time', time.time() - tic)
        # stack net and get k-core number
        date_list = list(range(start_date, end_date + 1, 1))  #
        day_time_int_dict = {}
        count = 0
        for day in date_list:
            day_time_int_dict[day] = list(np.arange(1,24+1, 1) + count*24)
            count += 1

        k_core_test = [8,7,6,5,4,3] # need
        # date_list = [4]
        for k_num_filter in k_core_test:
            new_sample_pax = []
            new_sample_pax_random = []
            for day in date_list:
                pt_net_day=[]
                for time_int in day_time_int_dict[day]:
                    pt_net_day.append(pt_net[time_int])
                # get_core_info(pt_net_day)
                delete_pax = get_delete_pax_id(pt_net_day, k_num_filter)
                new_sample, new_sample_random = get_net_sample_pax(delete_pax, data.loc[data['date']==day])
                new_sample_pax.append(new_sample)
                new_sample_pax_random.append(new_sample_random)
            new_sample_pax = pd.concat(new_sample_pax)
            new_sample_pax_random = pd.concat(new_sample_pax_random)
            with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(sample_seed) +'_k_core_' + str(int(k_num_filter)) + '.pickle', 'wb') as handle:
                pickle.dump(new_sample_pax, handle)
            with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(
                    sample_size) + '_seed_' + str(sample_seed) + '_k_core_random' + str(int(k_num_filter)) + '.pickle',
                      'wb') as handle:
                pickle.dump(new_sample_pax_random, handle)
            print('Finishe generate k core ' + str(k_num_filter) + '==============================')
        # print('Please run 02_to generate net first before running simulation')

    if GENERATE_CONTACT_TIME:
        k_core_test = [8,7,6,5,4,3] # need

        date_list = list(range(start_date, end_date + 1, 1))  #

        for k_num_filter in k_core_test:


            print('===============CURRENT k_core' + str(k_num_filter) + '=============')

            tic = time.time()
            # save_file_name = '_k_core_' + str(int(k_num_filter))
            # with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' +
            #           str(sample_seed) +'_k_core_' + str(int(k_num_filter)) + '.pickle', 'rb') as handle:
            #     data = pickle.load(handle)
            save_file_name = '_k_core_random' + str(int(k_num_filter))
            with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' +
                      str(sample_seed) +'_k_core_random' + str(int(k_num_filter)) + '.pickle', 'rb') as handle:
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








