import pandas as pd
import numpy as np
import time
import pickle
import networkx as nx

def generate_pt_mat(sample_size, pt_net, seed, net_whole):

    pt_net_mat = {}
    print('start')

    for time_int in pt_net:
        print('PT time', time_int)
        pt_net_int = pt_net[time_int]
        if len(pt_net_int) == 0:
            mat = np.zeros(shape=(sample_size, sample_size), dtype=float)
            pt_net_mat[time_int] = mat
            continue
        pt_net_int = pt_net_int.merge(net_whole,left_on = ['P_id_x'],right_on = ['P_id'])
        pt_net_int = pt_net_int.rename(columns= {'new_id':'P_id_x_new'})
        pt_net_int = pt_net_int.merge(net_whole, left_on=['P_id_y'], right_on=['P_id'])
        pt_net_int = pt_net_int.rename(columns={'new_id': 'P_id_y_new'})
        pt_net_int['w_ij'] = pt_net_int['contact_duration']/3600
        val = pt_net_int[['P_id_x_new','P_id_y_new','w_ij']].values
        i = val[:, 0].ravel().astype(int)
        j = val[:, 1].ravel().astype(int)
        v = val[:, 2].ravel()
        mat = np.zeros(shape=(sample_size, sample_size), dtype=float)
        mat[i, j] = v
        pt_net_mat[time_int] = mat

    # with open('../data/PT_eco_mat_' + str(sample_size) +'_seed_' + str(seed) +  '.pickle', 'wb') as handle:
    #     pickle.dump(pt_net_mat,handle)


def generate_local_mat(sample_size, local_net, seed, net_whole):

    local_net_mat = {}
    print('start')

    for time_int in local_net:
        print('local net time', time_int)
        local_net_int = local_net[time_int]
        if len(local_net_int) == 0:
            mat = np.zeros(shape=(sample_size, sample_size), dtype=float)
            local_net_mat[time_int] = mat
            continue
        local_net_int = local_net_int.merge(net_whole,left_on = ['P_id_x'],right_on = ['P_id'])
        local_net_int = local_net_int.rename(columns= {'new_id':'P_id_x_new'})
        local_net_int = local_net_int.merge(net_whole, left_on=['P_id_y'], right_on=['P_id'])
        local_net_int = local_net_int.rename(columns={'new_id': 'P_id_y_new'})
        local_net_int['w_ij'] = local_net_int['contact_duration']/3600
        val = local_net_int[['P_id_x_new','P_id_y_new','w_ij']].values
        i = val[:, 0].ravel().astype(int)
        j = val[:, 1].ravel().astype(int)
        v = val[:, 2].ravel()
        mat = np.zeros(shape=(sample_size, sample_size), dtype=float)
        mat[i, j] = v
        local_net_mat[time_int] = mat

    # with open('../data/local_mat_' + str(sample_size) +'_seed_' + str(seed) +  '.pickle', 'wb') as handle:
    #     pickle.dump(local_net_mat,handle)

if __name__ == '__main__':
    time_interval = 1*3600 # 1 hour
    sample_size = 5000
    seed = 0
    with open('../data/sample_pax_' + str(sample_size) + '_seed_' + str(seed) + '.pickle', 'rb') as handle:
        passenger_list = pickle.load(handle)
    with open('../data/PT_eco_net_' + str(sample_size) +'_seed_' + str(seed) +  '.pickle', 'rb') as handle:
        pt_net = pickle.load(handle)
    with open('../data/Local_contact_net_'+ str(sample_size) + '_seed_' + str(seed) +  '.pickle', 'rb') as handle:
        local_net = pickle.load(handle)

    net_whole = pd.DataFrame({'P_id': passenger_list})
    net_whole['new_id'] = range(len(passenger_list))


    # generate_pt_mat(sample_size, pt_net, seed, net_whole)
    # generate_local_mat(sample_size, local_net, seed, net_whole)