import pandas as pd
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
from matplotlib.collections import LineCollection
from fa2 import ForceAtlas2
from _curved_edges import curved_edges


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

def construct_pt_net(data,time_used, time_interval):
    data['start_time_day'] = data['start_time']
    data['end_time_day'] = data['start_time_day'] + data['ride_duration']
    time_int_id = 1
    PT_net = []
    for time_int in range(time_used[0],time_used[1], time_interval):
        time_int_s = time_int
        time_int_e = time_int + time_interval
        data_time_int = data.loc[~((data['start_time_day'] > time_int_e) |
                                   (data['end_time_day'] < time_int_s))]
        print('number of people in time interval', time_int, len(data_time_int))
        if len(data_time_int) > 0:
            data_net = construct_PT_encounter_net(data_time_int, time_int_s, time_int_e)
            data_net = data_net.groupby(['P_id_x', 'P_id_y'])['contact_duration'].sum().reset_index()
            data_net.loc[data_net['contact_duration'] > time_interval,
                         'contact_duration'] = time_interval  # should not happen, just for conservative consideration.

            data_net['time_int_id'] = time_int_id
            PT_net.append(data_net)
        else:
            PT_net.append(pd.DataFrame({'P_id_x': [], 'P_id_y': [], 'contact_duration': []}))
        time_int_id += 1
    return pd.concat(PT_net)


def construct_local_net(data,time_used, time_interval):
    # process OD
    data['start_time_day'] = data['start_time']
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
    ## assume o start 6 hours ago, d ends 6 hours later (typical work time)
    data['o_start_time'] = data['o_end_time'] - 6*3600
    data['d_end_time'] = data['d_start_time'] + 6*3600
    data = data.loc[(data['o_start_time']<data['o_end_time']) & (data['d_start_time']<data['d_end_time'])] # due to the records errors in the AFC data
    #==== trip sequence define as consecutive trips within 24 hours
    print('Total trip seq records', len(data))
    data = data.loc[(data['o_end_time'] - data['o_start_time']< 24*3600)]
    data = data.loc[(data['d_end_time'] - data['d_start_time']< 24*3600)]
    print('After 24 h threshold trip seq records', len(data))
    #=======
    local_contact_net = []
    time_int_id = 0
    for time_int in range(time_used[0],time_used[1], time_interval):
        time_int_id += 1
        print('current time interval id:', time_int_id)
        # give data for one time interval
        time_int_s = time_int
        time_int_e = time_int + time_interval

        ########### o connection
        data_time_int_1 = data.loc[~((data['o_start_time'] > time_int_e)|
                                      (data['o_end_time']< time_int_s))]
        data_time_int_1 = data_time_int_1.rename(columns = {'o_start_time':'station_start_time','o_end_time':'station_end_time',
                                                            'boarding_stop':'station_id'})
        data_time_int_2 = data.loc[~((data['d_start_time'] > time_int_e) |
                                        (data['d_end_time'] < time_int_s))]
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
        local_contact_net.append(data_net)
    return pd.concat(local_contact_net)

def draw_network(net_used, used_individual, pos, save_fig,tail_name):
    time_int_list = pd.unique(net_used['time_int_id'])
    time_int_list = sorted(time_int_list)
    count = 0
    # forceatlas2 = ForceAtlas2()
    # pos = forceatlas2.forceatlas2_networkx_layout(G_all, pos=None, iterations=30)
    for time_ind in time_int_list:
        count += 1
        net_ind =net_used.loc[net_used['time_int_id'] == time_ind]
        G = nx.from_pandas_edgelist(net_ind,'P_id_x','P_id_y')
        print('num of node before', G.number_of_nodes())
        G.add_nodes_from(used_individual)
        print('num of node after',G.number_of_nodes())
        # Produce the curves
        curves = curved_edges(G, pos)
        #lc = LineCollection(curves, color='w', alpha=0.05)

        # Plot
        plt.figure(figsize=(6, 6))
        # degree_list = np.sqrt(np.array([key[1] + 1 for key in G.degree()]))*4
        degree_list = np.array([key[1] + 1 for key in G.degree()])
        for curve in curves:
            plt.plot(curve[:,0],curve[:,1],color='k', lw = 0.3, alpha=0.4)
        nx.draw_networkx_nodes(G, pos, node_size=  degree_list, node_color='r', alpha=1)
        # nx.draw_networkx_edges(G, pos, width=0.2, edge_color='k', style='solid', alpha=0.4)
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.tight_layout()
        if save_fig == 1:
            plt.savefig('img/' + str(sample_net) + '_sample_' + tail_name + '_time_' + str(time_ind) +  '.png', dpi=200)
        else:
            plt.show()
            plt.close()

def network_viz(net_used_pt, net_used_local, sample_net, save_fig):
    ################PT FIRST
    all_individual = set(pd.unique(net_used_pt['P_id_x'])).union(set(pd.unique(net_used_pt['P_id_y'])))
    random.seed(3)
    used_individual = list(set((random.sample(list(all_individual), sample_net))))
    print('Number of used individual', len(used_individual))
    net_used_pt = net_used_pt.loc[(net_used_pt['P_id_x'].isin(used_individual)) & (net_used_pt['P_id_y'].isin(used_individual))]
    net_used_local = net_used_local.loc[
        (net_used_local['P_id_x'].isin(used_individual)) & (net_used_local['P_id_y'].isin(used_individual))]
    G_all = nx.from_pandas_edgelist(net_used_pt, 'P_id_x', 'P_id_y')
    print('num of node before', G_all.number_of_nodes())
    G_all.add_nodes_from(used_individual)
    print('num of node all', G_all.number_of_nodes())
    pos = nx.spring_layout(G_all, k=1, seed=6)

    draw_network(net_used_pt, used_individual, pos, save_fig, tail_name = 'PT_cont_net')
    draw_network(net_used_local, used_individual, pos, save_fig, tail_name = 'Local_cont_net')

def select_data(data, date_used, time_used, time_interval):
    total_int = (time_used[1] - time_used[0]) // 3600
    data_used = data.loc[data['date'] == date_used]
    data_used = data_used.loc[(data_used['start_time'] >= time_used[0]) & (data_used['start_time'] < time_used[1])]
    data_used ['time_interval'] =data_used['start_time'] // time_interval
    # select most congested bus:
    data_used_bus_count = data_used.groupby(['bus_id','time_interval'])['P_id'].count().reset_index(drop=False)
    data_used_bus_count = data_used_bus_count.loc[data_used_bus_count['P_id']>20]
    data_used_bus_count = data_used_bus_count.groupby(['bus_id'])['time_interval','P_id'].agg({'time_interval': 'count', 'P_id': 'sum'}).reset_index(drop=False)
    data_used_bus_count = data_used_bus_count.loc[data_used_bus_count['time_interval'] == total_int] # every int with data
    data_used_bus_count = data_used_bus_count.sort_values(['P_id'],ascending = False)
    max_bus_id = data_used_bus_count['bus_id'].iloc[0]
    data_used =  data_used.loc[data_used['bus_id'] == max_bus_id]
    return data_used



if __name__ == '__main__':
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31
    GENERATE = False
    if GENERATE:
        tic = time.time()
        data = pd.read_csv('../data/data_Aug_compressed.csv')
        print('load file time',time.time() - tic)
        date_used = 21  # aug 21
        time_used = [7 * 3600, 10 * 3600]
        data_used = select_data(data, date_used, time_used, time_interval)
        with open('../data/data_for_PT_net_plotting', 'wb') as handle:
            pickle.dump(data_used, handle)
        print('finish generate data')
    else:
        with open('../data/data_for_PT_net_plotting', 'rb') as handle:
            data_used = pickle.load(handle)
        #====================pt contact net
        time_used = [7 * 3600, 10 * 3600]
        sample_net = 100
        pt_net_used = construct_pt_net(data_used, time_used, time_interval)
        local_net_used = construct_local_net(data_used, time_used, time_interval)
        network_viz(pt_net_used,local_net_used, sample_net, save_fig=1)





    # with open('../data/PT_eco_net_' + str(sample_size) + '_seed_' + str(sample_seed) + '.pickle', 'rb') as handle:
    #     pt_net = pickle.load(handle)
    # with open('../data/Local_contact_net_' + str(sample_size) + '_seed_' + str(sample_seed) + '.pickle', 'rb') as handle:
    #     local_net = pickle.load(handle)
