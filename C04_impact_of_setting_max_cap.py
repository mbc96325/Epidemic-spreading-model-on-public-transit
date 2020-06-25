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

def construct_PT_encounter_net(data,time_int_s, time_int_e,cap, delete_pax_list):
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
    #==============SET MAX CAP==============
    data_pax1 = data_net.loc[:,['bus_id','P_id_x']].rename(columns = {'P_id_x':'P_id'})
    data_pax2 = data_net.loc[:, ['bus_id', 'P_id_y']].rename(columns = {'P_id_y':'P_id'})
    data_pax = pd.concat([data_pax1,data_pax2])
    data_pax = data_pax.drop_duplicates()
    data_net_num = data_pax.groupby(['bus_id'])['P_id'].count().reset_index()
    bus_need_to_proces = data_net_num.loc[data_net_num['P_id']>cap]
    bus_need_to_proces = bus_need_to_proces.rename(columns = {'P_id':'num_pax'})
    if len(bus_need_to_proces)>0:
        bus_id_list = list(bus_need_to_proces['bus_id'])
        data_pax_new = data_pax.merge(bus_need_to_proces, on = ['bus_id'])
        old_pax = set(list(data_pax_new['P_id']))
        data_pax_new = data_pax_new.sample(frac=1).reset_index(drop=True) # shuffle
        data_pax_new = data_pax_new.groupby(['bus_id']).head(cap)
        data_pax_new = data_pax_new.reset_index()
        data_pax_remained = data_pax.loc[~data_pax['bus_id'].isin(bus_id_list)]
        new_pax = set(list(data_pax_new['P_id']))
        delete_pax = list(old_pax.difference(new_pax))
        new_pax = list(new_pax)
        delete_pax_list += delete_pax
        data_pax_new = pd.concat([data_pax_remained,data_pax_new])
        data_net = data_net.merge(data_pax_new[['P_id','bus_id']], left_on = ['P_id_x','bus_id'],right_on = ['P_id','bus_id'])
        data_net = data_net.drop(columns=['P_id'])
        data_net = data_net.merge(data_pax_new[['P_id','bus_id']], left_on=['P_id_y', 'bus_id'], right_on=['P_id', 'bus_id'])
        data_net = data_net.drop(columns=['P_id'])
        # if len(data_net)>0:
        #     a=1
        # data_net = data_net.loc[data_net['P_id_x'].isin(new_pax)]
        # data_net = data_net.loc[data_net['P_id_y'].isin(new_pax)]
    # data_net = data_net.sort_values(['bus_id'])
    return data_net[['P_id_x','P_id_y','contact_duration']], delete_pax_list

def PT_connection(data,date_list,time_interval, cap, delete_pax_list):
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
                data_net, delete_pax_list = construct_PT_encounter_net(data_time_int, time_int_s, time_int_e, cap, delete_pax_list)
                if len(data_net) == 0:
                    PT_net[time_int_id] = pd.DataFrame({'P_id_x': [], 'P_id_y': [], 'contact_duration': []})
                else:
                    data_net = data_net.groupby(['P_id_x', 'P_id_y'])['contact_duration'].sum().reset_index()
                    data_net.loc[data_net['contact_duration'] > time_interval,
                                 'contact_duration'] = time_interval  # should not happen, just for conservative consideration.

                    data_net['time_int_id'] = time_int_id
                    PT_net[time_int_id] = data_net

            else:
                PT_net[time_int_id] = pd.DataFrame({'P_id_x':[],'P_id_y':[],'contact_duration':[]})
    # print('Finish PT encounter network')
    return PT_net, delete_pax_list


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
    seed = 0
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31
    sample_size = 100000
    sample_seed = 0
    GENERATE_SAMPLE_AND_NET = True



    if GENERATE_SAMPLE_AND_NET:
        ###################PERCENTAGE===============
        date_list = list(range(start_date, end_date + 1, 1))  #
        # cap_list = list(range(10,1,-1))
        cap_list = [10,9,8,7,6,5,4,3,2]
        num_delete_pax_dict = {'max_cap':[],'num_affected_pax':[]}
        for cap in  cap_list:
            delete_pax_list = []
            print('===============CURRENT percentage ' + str(cap) + '=============')
            tic = time.time()
            with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(
                    sample_size) + '_seed_' + str(sample_seed) + '.pickle', 'rb') as handle:
                data = pickle.load(handle)
            save_file_name = 'max_cap_' + str(int(cap))

            print('load file time', time.time() - tic)
            PT_net_new, delete_pax_list = PT_connection(data, date_list, time_interval, cap, delete_pax_list)
            delete_pax_list = list(set(delete_pax_list))
            num_delete_pax = len(delete_pax_list)
            contact_time_pt = cal_pt_cont_time(PT_net_new)
            print('Total contact time', sum(contact_time_pt['total_contact_time']))
            num_delete_pax_dict['max_cap'].append(cap)
            num_delete_pax_dict['num_affected_pax'].append(num_delete_pax)
            print('finish_PT')
            contact_time_pt.to_csv('output/pt_total_contact_time_' + save_file_name + '.csv',index=False)
            #===============

        num_delete_pax_dict = pd.DataFrame(num_delete_pax_dict)
        num_delete_pax_dict.to_csv('../data/number_of_affected_pax_max_cap.csv',index=False)





