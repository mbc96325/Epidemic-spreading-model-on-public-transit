import pandas as pd
import numpy as np
import time
import pickle
import _constants
import matplotlib.pyplot as plt
import random

class caculation:
    def __init__(self,pt_net, sim_period, local_net, para_dict, sample_size, time_interval, sample_pax,
                 start_time_int, random_seed = 0):
        self.pt_net = pt_net
        self.local_net = local_net
        self.beta_I = para_dict['beta_I']
        self.beta_E = para_dict['beta_E']
        self.mu_r = para_dict['mu_r']
        self.mu_d = para_dict['mu_d']
        self.theta_l = para_dict['theta_l']
        self.theta_g = para_dict['theta_g']
        self.gamma = para_dict['gamma']
        self.sim_period = sim_period
        self.period_length = len(self.pt_net)
        self.time_interval = time_interval
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.pax_index = sample_pax

    def calculate_contact_time(self, save_file_name):
        total_contact_time_list_pt = []
        total_contact_time_square_list_pt = []
        total_contact_time_list_local = []
        total_contact_time_square_list_local = []
        total_contact_time_list_global = []
        total_contact_time_square_list_global = []
        time_int_list = []
        for time_int in range(1,self.sim_period+1):
            time_int_list.append(time_int)
            pt_net_int = self.pt_net[time_int]
            local_net_int_original = self.local_net[time_int]
            # =================== randomly assign local connection
            seed = self.random_seed + time_int * 2 + 1
            num_local = int(round(self.theta_l * len(local_net_int_original)))
            np.random.seed(seed)
            local_net_idx = np.random.choice(local_net_int_original.index, num_local)
            local_net_int = local_net_int_original.loc[local_net_idx, :]
            #============================
            total_contact_time_list_pt.append(pt_net_int['contact_duration'].sum())
            total_contact_time_square_list_pt.append(np.sum(np.square(pt_net_int['contact_duration'])))
            total_contact_time_list_local.append(local_net_int['contact_duration'].sum())
            total_contact_time_square_list_local.append(np.sum(np.square(local_net_int['contact_duration'])))
            #==================
            num_global = int(round(self.theta_g * self.sample_size * self.sample_size))
            np.random.seed(self.random_seed + time_int * 100 + 2)
            global_net_int = pd.DataFrame({'P_id_x': np.random.choice(self.pax_index, num_global),
                                           'P_id_y': np.random.choice(self.pax_index, num_global),
                                           'w_ij': 1})
            global_net_int = global_net_int.loc[global_net_int['P_id_x'] != global_net_int['P_id_y']]
            #=================
            total_contact_time_list_global.append(len(global_net_int) * self.time_interval)
            total_contact_time_square_list_global.append(len(global_net_int) * self.time_interval * self.time_interval)

        # ===============
        contact_time = pd.DataFrame({'time_interval':time_int_list,'total_contact_time':total_contact_time_list_pt,
                                     'total_contact_time_square':total_contact_time_square_list_pt})

        contact_time.to_csv('output/pt_total_contact_time_' + save_file_name + '.csv',index=False)
        #===============
        contact_time = pd.DataFrame({'time_interval':time_int_list,'total_contact_time':total_contact_time_list_local,
                                     'total_contact_time_square':total_contact_time_square_list_local})

        contact_time.to_csv('output/local_total_contact_time_' + save_file_name + '.csv',index=False)
        # ===============
        contact_time = pd.DataFrame({'time_interval':time_int_list,'total_contact_time':total_contact_time_list_global,
                                     'total_contact_time_square':total_contact_time_square_list_global})

        contact_time.to_csv('output/global_total_local_contact_time_' + save_file_name + '.csv',index=False)

def single_caculation(pt_net, local_net, sample_pax, save_file_name,sample_size):
    info_file = pd.read_csv('output/simulation_para_' + save_file_name + '.csv')
    start_time_int = info_file['start_time_int'].iloc[0]
    sim_period = info_file['total_sim_period'].iloc[0]
    mu_r = info_file['mu_r'].iloc[0]
    mu_d = info_file['mu_d'].iloc[0]
    beta_I = info_file['beta_I'].iloc[0]
    beta_E = info_file['beta_E'].iloc[0]
    theta_l = info_file['theta_l'].iloc[0]
    theta_g = info_file['theta_g'].iloc[0]
    gamma = info_file['gamma'].iloc[0]
    sim_seed = info_file['sim_seed'].iloc[0]
    time_interval = info_file['time_interval'].iloc[0]

    para_dict = {'mu_r': mu_r, 'mu_d':mu_d, 'beta_I': beta_I,'beta_E': beta_E,
                 'theta_l':theta_l,'theta_g':theta_g,'gamma':gamma}

    cal = caculation(pt_net, sim_period, local_net, para_dict, sample_size,
                         time_interval, sample_pax, start_time_int, random_seed = sim_seed)
    cal.calculate_contact_time(save_file_name)

def caculate_for_sample_data():
    sample_size = 100000
    seed = 0
    with open('../data/PT_eco_net_' + str(sample_size) + '_seed_' + str(seed) + '.pickle', 'rb') as handle:
        pt_net = pickle.load(handle)
    with open('../data/Local_contact_net_'+ str(sample_size) + '_seed_' + str(seed) + '.pickle', 'rb') as handle:
        local_net = pickle.load(handle)
    with open('../data/sample_pax_'+ str(sample_size) + '_seed_' + str(seed) + '.pickle', 'rb') as handle:
        sample_pax = pickle.load(handle)

    #############=========================
    # save_file_name = 'test1'
    # single_caculation(pt_net, local_net, sample_pax, save_file_name, sample_size)
    #############=========================
    # save_file_name_list = ['system_test_' + str(id) for id in range(312, 320+1, 1)]
    # for save_file_name in save_file_name_list:
    #     print('===============')
    #     print('Current process ID:', save_file_name)
    #     single_caculation(pt_net, local_net, sample_pax, save_file_name,sample_size)

def caculate_for_spreading_data():

    time_interval = 1*3600 # 1 hour
    sample_size = 100000
    seed = 0
    unit_move_list = [10,30,50,70,90,110]
    # unit_move_list = [30]
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
        print('===============')
        save_file_name = 'spreading' + str(int(unit_move / 60))
        print('Current process ID:', save_file_name)
        single_caculation(pt_net, local_net, sample_pax, save_file_name,sample_size)




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
    PT_contact_time = []
    for date in date_list:
        data_date = data.loc[data['date'] == date]
        for time_int in range(0, 86400, time_interval):
            time_int_id += 1
            print('current time interval id:', time_int_id)
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

            else:
                data_net = pd.DataFrame({'P_id_x':[],'P_id_y':[],'contact_duration':[],'time_int_id':[]})

            total_cont_time = data_net['contact_duration'].sum()
            total_cont_time_sq = np.sum(np.square(data_net['contact_duration']))
            PT_contact_time.append(pd.DataFrame({'time_interval':[time_int_id], 'total_contact_time':[total_cont_time], 'total_contact_time_square':[total_cont_time_sq]}))

    return pd.concat(PT_contact_time)

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

def local_contact(data, date_list, time_interval, theta_l):
    # process OD
    data['start_time_day'] = data['start_time'] + data['date'] * 3600*24
    data['end_time_day'] = data['start_time_day'] + data['ride_duration']
    data = data.drop(columns = ['start_time'])
    tic = time.time()
    data = data.sort_values(['start_time_day'])
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
    local_contact_time = []
    #############
    bus_stop = pd.read_csv('../data/bus_stop_id_lookup.csv')
    bus_stop_list = list(bus_stop['bus_stop'])
    ##############
    for date in date_list:
        data_date = data.loc[data['date'] == date]
        count_local_per_bus = []
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
            data_net = []

            count_local_per_bus_timeint = []
            for bus_stop_id in bus_stop_list:
                data_time_int_busstop = data_time_int.loc[data_time_int['station_id'] == bus_stop_id]
                count_local_per_bus_timeint.append(len(data_time_int_busstop['P_id']))
                # for a specific bus stop, they are all connected, can directly sample node
                passenger_list = pd.unique(data_time_int_busstop['P_id'])
                sample_size = int(round(len(passenger_list) * (np.sqrt(theta_l))))
                random.seed(0)
                used_p = random.sample(list(passenger_list), sample_size)
                # print('num_sample_used', len(used_p))
                data_time_int_busstop = data_time_int_busstop.set_index(['P_id'])
                data_time_int_busstop_sample = data_time_int_busstop.loc[used_p, :]
                data_time_int_busstop_sample = data_time_int_busstop_sample.reset_index(drop=False)
                if len(data_time_int_busstop_sample) > 0:
                    data_net_bus = construct_contact_net(data_time_int_busstop_sample, time_int_s, time_int_e,
                                                       start_time_name = 'station_start_time',end_time_name ='station_end_time',
                                                       od_name ='station_id')
                else:
                    data_net_bus = pd.DataFrame({'P_id_x':[],'P_id_y':[],'contact_duration':[]})
                data_net.append(data_net_bus)
            data_net = pd.concat(data_net,sort=False)
            avg_local_per_bus = sum(count_local_per_bus_timeint)/len(count_local_per_bus_timeint)
            print('avg_local_per_bus_stop', avg_local_per_bus)
            count_local_per_bus.append(avg_local_per_bus)
            ########## merge to together
            data_net = data_net.groupby(['P_id_x','P_id_y'])['contact_duration'].sum().reset_index()
            data_net.loc[data_net['contact_duration']>time_interval,'contact_duration'] = time_interval # should not happen, just for conservative consideration.
            data_net['time_int_id'] = time_int_id

            total_cont_time = data_net['contact_duration'].sum()
            total_cont_time_sq = np.sum(np.square(data_net['contact_duration']))
            local_contact_time.append(pd.DataFrame({'time_interval':[time_int_id], 'total_contact_time':[total_cont_time], 'total_contact_time_square':[total_cont_time_sq]}))
        print('avg local people per day', sum(count_local_per_bus)/len(count_local_per_bus))
        count_local_per_bus = []
    return pd.concat(local_contact_time)

def global_contact(data, date_list, time_interval, theta_g):
    passenger_list = pd.unique(data['P_id'])
    total_pax = len(passenger_list)
    time_int_id = 0
    global_contact_time = []
    single_edge_time = 3600
    print('total pax', len(passenger_list))
    for date in date_list:
        for time_int in range(0, 86400, time_interval):
            time_int_id += 1
            total_edges_used = total_pax * (total_pax-1) * theta_g
            total_cont_time = total_edges_used * single_edge_time
            total_cont_time_sq = total_edges_used * single_edge_time * single_edge_time
            global_contact_time.append(pd.DataFrame(
                {'time_interval': [time_int_id], 'total_contact_time': [total_cont_time],
                 'total_contact_time_square': [total_cont_time_sq]}))
    return pd.concat(global_contact_time)


def caculate_for_all_data():
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31
    tic = time.time()
    TEST = False
    if TEST:
        sample_size = 100000
        sample_seed = 0
        with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(sample_seed) +'.pickle', 'rb') as handle:
            data = pickle.load(handle)
        date_list = list(range(4, 5, 1))  #
    else:
        data = pd.read_csv('../data/data_Aug_compressed.csv')
        date_list = list(range(start_date, end_date + 1, 1))  #
    print('total trips', len(data))
    print('load file time', time.time() - tic)

    #================================
    # pt_cont_time = PT_connection(data, date_list, time_interval)
    # pt_cont_time.to_csv('output/PT_total_contact_time_all_city.csv',index=False)
    #================================
    theta_l = _constants.theta_l

    local_contact_time = local_contact(data, date_list, time_interval, theta_l)
    local_contact_time.to_csv('output/local_total_contact_time_all_city.csv',index=False)
    #================================
    # theta_g = _constants.theta_g
    # global_contact_time = global_contact(data, date_list, time_interval, theta_g)
    # global_contact_time.to_csv('output/global_total_contact_time_all_city.csv',index=False)

if __name__ == '__main__':
    #============================
    caculate_for_sample_data()

    #=============================
    # caculate_for_all_data()

    #============================
    caculate_for_spreading_data()
