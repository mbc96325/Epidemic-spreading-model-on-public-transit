import pandas as pd
import numpy as np
import time
import pickle
import _constants
import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("Paired")
def plot_infect_log(name_list, save_fig):
    ###########Plot
    font_size = 16
    state = 'I'
    color_id = 0
    plt.figure(figsize=(10, 8))
    for name in name_list:
        data = pd.read_csv('output/sim_results_'+ name +  '.csv')
        sim_info = pd.read_csv('output/simulation_para_' +  name + '.csv')
        sample_size = sim_info['sample_size'].iloc[0]
        data[state+'_t' + '_prop'] = data[state+'_t']/sample_size
        plt.loglog(data['time_interval'], data[state+'_t' + '_prop'], 'k-',color = colors[color_id], markersize = 4)
        color_id += 1

    plt.xlabel('Log time', fontsize=font_size)
    plt.ylabel('Log proportion ' + state + ' people', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/Loglog of ' + state + '.png', dpi = 200)
    else:
        plt.show()
    #plt.close()


def generate_x_label(total_sim_period):
    day_list = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    Mon_start_time_int = 12
    div = total_sim_period // 7*24
    mod = total_sim_period % 7*24
    if mod>0:
        div += 1
    x_ticks = []
    new_ticks = []
    xvline_x = [1]
    for period in range(div):
        count = 0
        for key in day_list:
            x_ticks_value = Mon_start_time_int + count*24 + 7*24*period
            if x_ticks_value <= total_sim_period:
                x_ticks.append(x_ticks_value)
                xvline_x.append(x_ticks_value + 12)
                new_ticks.append(key)
                count += 1
            else:
                break
    return x_ticks, new_ticks, xvline_x


def plot_all_states(name_list, save_fig):
    state_list = ['S','E','I','R']
    ###########Plot
    font_size = 16
    for name in name_list:
        sim_info = pd.read_csv('output/simulation_para_' +  name + '.csv')
        total_sim_period = sim_info['total_sim_period'].iloc[0]
        data = pd.read_csv('output/sim_results_' + name + '.csv')
        x_ticks, new_ticks, xvline_x = generate_x_label(total_sim_period)
        for state in state_list:
            plt.figure(figsize=(18, 8))
            plt.plot(data['time_interval'], data[state + '_t'], 'k-^', markersize=4)
            plt.xlabel('Time', fontsize=font_size)
            plt.ylabel('Number of ' + state + ' people', fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.xticks(x_ticks, new_ticks, fontsize=font_size)
            [plt.axvline(_x, linewidth=1, color='g') for _x in xvline_x]
            plt.tight_layout()
            if save_fig == 1:
                plt.savefig('img/Profile of ' + state + '_' + self.save_file_name + '.png', dpi=200)
            else:
                plt.show()

if __name__ == '__main__':
    name_list = ['system_test_1']
    #plot_infect_log(name_list, save_fig = 0)
    plot_all_states(name_list, save_fig = 0)