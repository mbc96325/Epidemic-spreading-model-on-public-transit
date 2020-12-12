import pickle
import _constants
import matplotlib.pyplot as plt
import _simulation_engine as sim_eng
import pandas as pd
import time
import itertools
from pprint import pprint
import random


def single_testing(save_file_name, time_interval, sample_size, seed):
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
                             verbal = True, random_seed = simulation_seed)
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


def system_test(para_hyperspace, num_of_test, sample_case_seed, iterate_all, time_interval, sample_size, seed):
    # get all combination
    inputdata = []
    for key in para_hyperspace:
        inputdata.append(para_hyperspace[key])
    result = list(itertools.product(*inputdata))
    if iterate_all:
        test_samples = result
    else:
        random.seed(sample_case_seed)
        test_samples = random.sample(result, num_of_test)
    print('Total sim cases', len(test_samples))
    count = 0
    for test_case in test_samples:
        count += 1
        save_file_name = 'system_test_' + str(count)
        print('==============')
        print('Current test number', count)
        start_time_int = 0
        sim_period = round(1 * len(pt_net) - start_time_int)
        mu_r = test_case[0]
        mu_d = test_case[1]
        beta_I = test_case[2]
        beta_E = 0.01*test_case[2]
        theta_l = test_case[3]
        theta_g = test_case[4]
        gamma = test_case[5]
        initial_I = test_case[6]

        para_dict = {'mu_r': mu_r, 'mu_d': mu_d, 'beta_I': beta_I, 'beta_E': beta_E,
                     'theta_l': theta_l, 'theta_g': theta_g, 'gamma': gamma, 'initial_I': initial_I}

        print(para_dict)

        simulation_seed = 0
        sim = sim_eng.simulation(pt_net, sim_period, local_net, para_dict, sample_size,
                                 time_interval, sample_pax, start_time_int, save_file_name,
                                 verbal=False, random_seed=simulation_seed)
        tic = time.time()
        sim.run()
        running_time = round(time.time() - tic, 1)
        df_test_info = {'test_name': [save_file_name], 'sample_size': [sample_size], 'time_interval': [time_interval],
                        'sample_seed': [seed], 'sim_seed': [simulation_seed], 'start_time_int': [start_time_int],
                        'total_sim_period': [sim_period]}
        for para in para_dict:
            df_test_info[para] = [para_dict[para]]
        df_test_info['running_time'] = [running_time]
        df_test_info = pd.DataFrame(df_test_info)
        df_test_info.to_csv('output/simulation_para_' + save_file_name + '.csv', index=False)

if __name__ == '__main__':
    time_interval = 1*3600 # 1 hour
    sample_size = 100000
    seed = 0
    tic = time.time()
    with open('../data/PT_eco_net_' + str(sample_size) + '_seed_' + str(seed) + '.pickle', 'rb') as handle:
        pt_net = pickle.load(handle)
    with open('../data/Local_contact_net_'+ str(sample_size) + '_seed_' + str(seed) + '.pickle', 'rb') as handle:
        local_net = pickle.load(handle)
    with open('../data/sample_pax_'+ str(sample_size) + '_seed_' + str(seed) + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
    print('load data time', time.time() - tic)

    #=================Single testing
    # single_testing('test1',time_interval, sample_size, seed)

    # =================System testing
    para_hyperspace = {'mu_r': [0.00042,0.0008,0.001,0.005], 'mu_d':[0.00042], 'beta_I': [0.00082, 0.0001,0.00001,0.0015,1e-6],
                       'theta_l':[1e-3,1e-2,1e-4,0],'theta_g':[1e-8,1e-7,1e-9,0],'gamma':[0.0104],'initial_I':[30]}
    # note beta_E is fixed as 0.01 beta_I
    iterate_all = True # if true, num_of_test and sample_case_seed are not effective
    num_of_test = 1
    sample_case_seed = 1
    system_test(para_hyperspace, num_of_test,sample_case_seed,iterate_all, time_interval, sample_size, seed)


