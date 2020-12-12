import pandas as pd
import numpy as np
import time

class simulation:
    def __init__(self,pt_net, sim_period, local_net, para_dict, sample_size, time_interval, sample_pax,
                 start_time_int, save_file_name, verbal, random_seed = 0):
        self.pt_net = pt_net
        self.local_net = local_net
        self.beta_I = para_dict['beta_I']
        self.beta_E = para_dict['beta_E']
        self.mu_r = para_dict['mu_r']
        self.mu_d = para_dict['mu_d']
        self.theta_l = para_dict['theta_l']
        self.theta_g = para_dict['theta_g']
        self.gamma = para_dict['gamma']
        self.initial_I = para_dict['initial_I']
        if sim_period > len(pt_net):
            print('Sim period large than all smart card data period')
            print('Will assume periodic travel behavior \n')
        self.period_length = len(self.pt_net)
        self.pax_state = np.zeros((sample_size,sim_period))
        self.time_int = 0
        self.state_list = {'S':1,'E':2,'I':3,'R':4}
        self.random_seed = random_seed
        self.time_interval = time_interval
        self.pax_index = sample_pax
        self.start_time_int = start_time_int
        self.save_file_name = save_file_name
        self.verbal = verbal

    def initialization(self):
        self.time_int = 0
        self.pax_state[:,0] = self.state_list['S']
        # randomly assign initial I
        np.random.seed(self.random_seed)
        idx = np.random.choice(range(0,self.pax_state[:,0].size + 1), size = self.initial_I)
        self.pax_state[idx, 0] = self.state_list['I']

    def From_S_to_E(self, Net_df, time_int, prob_S2E, seed):
        Net_df['prob_to_be_infected'] = Net_df['w_ij'] * prob_S2E
        np.random.seed(seed)
        Net_df['generator'] = np.random.rand(len(Net_df))
        Net_df_infect = Net_df.loc[Net_df['generator']<= Net_df['prob_to_be_infected']]
        if len(Net_df_infect)>0:
            idx = Net_df_infect.loc[:,'P_id_to_cal'].values
            pax_idx = np.isin(self.pax_index, idx)
            self.pax_state[pax_idx,time_int] = self.state_list['E']

    def From_E_to_I(self, E_pax, time_int, prob_E2I, seed):
        np.random.seed(seed)
        generator = np.random.rand(len(E_pax))
        idx = E_pax[np.where(generator <= prob_E2I)].reshape(-1)
        idx_not_change = E_pax[np.where(generator > prob_E2I)].reshape(-1)
        if len(idx)>0:
            self.pax_state[idx, time_int] = self.state_list['I']
        if len(idx_not_change)>0:
            self.pax_state[idx_not_change, time_int] = self.state_list['E']


    def From_I_to_R(self, I_pax, time_int, prob_E2I, seed):
        np.random.seed(seed)
        generator = np.random.rand(len(I_pax))
        idx = I_pax[np.where(generator <= prob_E2I)].reshape(-1)
        idx_not_change = I_pax[np.where(generator > prob_E2I)].reshape(-1)
        if len(idx)>0:
            self.pax_state[idx, time_int] = self.state_list['R']
        if len(idx_not_change)>0:
            self.pax_state[idx_not_change, time_int] = self.state_list['I']

    def calculate_infect_by_net(self, net_int, time_int, seed):
        I_pax = self.pax_index[np.where(self.pax_state[:,time_int - 1] == self.state_list['I'])]
        # using 1 and 2 is because we only record one edge
        if len(I_pax) > 0:
            Infect_by_I_1 = net_int.set_index(['P_id_x']).join(pd.DataFrame(I_pax)).drop(columns = [0]).rename(columns = {'P_id_y':'P_id_to_cal'})
            Infect_by_I_2 = net_int.set_index(['P_id_y']).join(pd.DataFrame(I_pax)).drop(columns=[0]).rename(columns = {'P_id_x':'P_id_to_cal'})
            Infect_by_I = pd.concat([Infect_by_I_1,Infect_by_I_2])
            self.From_S_to_E(Infect_by_I,time_int, self.beta_I, seed)

        E_pax = self.pax_index[np.where(self.pax_state[:, time_int - 1] == self.state_list['E'])]
        if len(E_pax) > 0:
            Infect_by_E_1 = net_int.set_index(['P_id_x']).join(pd.DataFrame(E_pax)).drop(columns = [0]).rename(columns = {'P_id_y':'P_id_to_cal'})
            Infect_by_E_2 = net_int.set_index(['P_id_y']).join(pd.DataFrame(E_pax)).drop(columns=[0]).rename(columns = {'P_id_x':'P_id_to_cal'})
            Infect_by_E = pd.concat([Infect_by_E_1,Infect_by_E_2])
            self.From_S_to_E(Infect_by_E,time_int, self.beta_E, seed*2)

    def calculate_E_to_I(self, time_int, seed):
        E_pax = np.argwhere(self.pax_state[:, time_int - 1] == self.state_list['E'])
        if len(E_pax) > 0:
            self.From_E_to_I(E_pax,time_int, self.gamma, seed)

    def calculate_I_to_R(self, time_int, seed):
        I_pax = np.argwhere(self.pax_state[:, time_int - 1] == self.state_list['I'])
        if len(I_pax) > 0:
            self.From_I_to_R(I_pax, time_int, self.mu_r + self.mu_d, seed)

    def state_transition(self,time_int):
        time_int_net = time_int % self.period_length + self.start_time_int
        if time_int_net == 0:
            time_int_net = self.period_length + self.start_time_int
        pt_net_int = self.pt_net[time_int_net]
        local_net_int_original = self.local_net[time_int_net]
        #=================== randomly assign local connection
        seed = self.random_seed + time_int * 2 + 1
        num_local = int(round(self.theta_l * len(local_net_int_original)))
        np.random.seed(seed)
        local_net_idx = np.random.choice(local_net_int_original.index, num_local)
        local_net_int = local_net_int_original.loc[local_net_idx,:]
        #=====================
        pt_net_int['w_ij'] = pt_net_int['contact_duration'] / self.time_interval
        local_net_int['w_ij'] = local_net_int['contact_duration'] / self.time_interval
        #==================
        seed = self.random_seed + time_int*2 + 1
        self.calculate_infect_by_net(pt_net_int, time_int, seed) # both I infect S or E infect S
        #==================
        seed = self.random_seed + time_int * 3 + 1
        self.calculate_infect_by_net(local_net_int, time_int, seed)  # both I infect S or E infect S
        #==================
        seed = self.random_seed + time_int * 4 + 1
        num_global = int(round(self.theta_g * self.pax_state.shape[0] * self.pax_state.shape[0]))
        np.random.seed(self.random_seed + time_int*100 + 2)
        global_net_int = pd.DataFrame({'P_id_x': np.random.choice(self.pax_index,num_global),
                                       'P_id_y': np.random.choice(self.pax_index,num_global),
                                       'w_ij':1})
        global_net_int = global_net_int.loc[global_net_int['P_id_x'] != global_net_int['P_id_y']]
        self.calculate_infect_by_net(global_net_int, time_int, seed)  # both I infect S or E infect S
        # ==================
        seed = self.random_seed + time_int * 5 + 1
        self.calculate_E_to_I(time_int, seed)  # both I infect S or E infect S
        # ==================
        seed = self.random_seed + time_int * 6 + 1
        self.calculate_I_to_R(time_int, seed)  # both I infect S or E infect S
        # =================
        # R to R
        self.pax_state[self.pax_state[:, time_int-1] == self.state_list['R'], time_int] = self.state_list['R']
        #==================
        # Assign the rest to S
        self.pax_state[self.pax_state[:,time_int] == 0, time_int] = self.state_list['S']


    def run(self):
        current_state_list = {}
        for state in self.state_list:
            current_state_list[state] = []
        time_int_list = []
        self.time_int = 0
        tic = time.time()
        for time_int in range(0,self.pax_state.shape[1]):
            time_int_list.append(time_int)
            if time_int==0:
                self.initialization()
            else:
                self.state_transition(time_int)

            for state in self.state_list:
                current_state_num = len(self.pax_state[self.pax_state[:, time_int] == self.state_list[state], time_int])
                current_state_list[state].append(current_state_num)

            # verbal
            if self.verbal:
                if time_int % 24 == 0:
                    print('Current time interval', time_int)
                    for state in self.state_list:
                        print('Current ' + state, current_state_list[state][-1])
                    print('=========================================')

            self.time_int += 1
        print('Total simulation time', time.time() - tic)
        ###########Save to file
        output = pd.DataFrame({'time_interval':time_int_list,'S_t':current_state_list['S'],
                               'E_t':current_state_list['E'],'I_t':current_state_list['I'],
                               'R_t':current_state_list['R']})
        output.to_csv('output/sim_results_' + self.save_file_name + '.csv',index=False)
