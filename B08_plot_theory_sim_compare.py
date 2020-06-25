import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
colors = sns.color_palette("Paired")
colors2 = sns.color_palette("muted")
def plot_I_compare_scatter(data_thry_I , data_sim_I, save_fig):
    value_thry = data_thry_I.values.ravel()
    value_sim = data_sim_I.values.ravel()
    font_size = 16
    matplotlib.rcParams['font.size'] = font_size-2
    fig,ax = plt.subplots(figsize=(7, 7))
    # the main axes is subplot(111) by default
    plt.scatter(value_sim, value_thry, s=10, marker = '.', color = colors2[0])
    plt.plot([0,4300], [0,4300], 'k--', linewidth = 1)
    plt.xlabel('Simulation model (# Infectious people)', fontsize=font_size)
    plt.ylabel('Theoretical model (# Infectious people)', fontsize=font_size)
    ax.yaxis.major.formatter._useMathText = True
    # x_ticks = list(range(0,trip_freq_lim + 10,10))
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.ylim([0, 4300])
    plt.xlim([0, 4300])
    plt.tight_layout()

    if save_fig == 1:
        plt.savefig('img/compare_theory_sim_I_scatter.png', dpi = 200)
    else:
        plt.show()

def plot_E_compare_scatter(data_thry_I , data_sim_I, save_fig):
    value_thry = data_thry_I.values.ravel()
    value_sim = data_sim_I.values.ravel()
    font_size = 16
    matplotlib.rcParams['font.size'] = font_size-2
    fig,ax = plt.subplots(figsize=(7, 7))
    # the main axes is subplot(111) by default
    plt.scatter(value_sim, value_thry, s=10, marker = '.', color = colors2[1])
    plt.plot([0,1000], [0,1000], 'k--', linewidth = 1)
    plt.xlabel('Simulation model (# Exposed people)', fontsize=font_size)
    plt.ylabel('Theoretical model (# Exposed people)', fontsize=font_size)
    ax.yaxis.major.formatter._useMathText = True
    # x_ticks = list(range(0,trip_freq_lim + 10,10))
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.ylim([0, 1000])
    plt.xlim([0, 1000])
    plt.tight_layout()

    if save_fig == 1:
        plt.savefig('img/compare_theory_sim_E_scatter.png', dpi = 200)
    else:
        plt.show()


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

def plot_E_compare_time_varying(data_thry_E , data_sim_E, case_used, save_fig):

    # case_used = [1,10,30,19,300]
    font_size = 16
    time_all = np.array(range(0,data_thry_E.shape[0])) + 1
    color_id = 0
    case_id_num = 0
    total_sim_period = data_thry_E.shape[0]
    x_ticks, new_ticks, xvline_x = generate_x_label(total_sim_period)
    plt.figure(figsize=(18, 7))
    for case_id in case_used:
        case_id_num += 1
        plt.plot(time_all, data_sim_E.iloc[:, case_id], '-^', markersize=4, color = colors[color_id], label = 'Case ' +str(case_id_num) + ' (Simulation)')
        color_id += 1
        plt.plot(time_all, data_thry_E.iloc[:, case_id], '-^', markersize=4, color = colors[color_id], label = 'Case ' +str(case_id_num) + ' (Theory)')
        color_id += 1
    plt.xlabel('Day ID', fontsize=font_size)
    plt.ylabel('Number of exposed people', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    new_ticks = [i for i in range(1, len(x_ticks)+1)]
    plt.xticks(x_ticks, new_ticks, fontsize=font_size)
    xvline_x_new = [xvline_x[7], xvline_x[14], xvline_x[21]]
    [plt.axvline(_x, linewidth=0.8, color='k',linestyle = '--') for _x in xvline_x_new]

    xvline_x_new = [xvline_x[7], xvline_x[14], xvline_x[21]]

    for i in range(len(xvline_x_new) + 1):
        off_set = 20
        if i == 0:
            x_pos = xvline_x_new[i] / 2 - off_set
        elif i>= len(xvline_x_new):
            x_pos = xvline_x_new[-1] + xvline_x_new[0] / 2 - off_set
        else:
            x_pos = (xvline_x_new[i-1] + xvline_x_new[i])/ 2 - off_set

        y_pos = 800
        plt.text(x_pos, y_pos, 'Week ' + str(i+1), fontsize = font_size, alpha = 0.5)


    plt.legend(fontsize=font_size-1, ncol = 5, loc = 'upper center')
    plt.ylim([0,1000])
    plt.xlim([-1, total_sim_period+1])
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/compare_theory_sim_E_time_varying.png', dpi=200)
    else:
        plt.show()

def plot_I_compare_time_varying(data_thry_I , data_sim_I,case_used, save_fig):
    # last_row = data_thry_I.iloc[-1,:]
    # want_case = [500,1000,2000,2700,3500]
    # case_used = []
    # for want_num in want_case:
    #     idx = np.argmin(np.abs(np.array(last_row) - want_num))
    #     case_used.append(idx)
    # case_used = [1,10,30,19,300]
    font_size = 16
    time_all = np.array(range(0,data_thry_I.shape[0])) + 1
    color_id = 0
    case_id_num = 0
    total_sim_period = data_thry_I.shape[0]
    x_ticks, new_ticks, xvline_x = generate_x_label(total_sim_period)
    plt.figure(figsize=(18, 7))
    for case_id in case_used:
        case_id_num += 1
        plt.plot(time_all, data_sim_I.iloc[:, case_id], '-^', markersize=4, color = colors[color_id], label = 'Case ' +str(case_id_num) + ' (Simulation)')
        color_id += 1
        plt.plot(time_all, data_thry_I.iloc[:, case_id], '-^', markersize=4, color = colors[color_id], label = 'Case ' +str(case_id_num) + ' (Theory)')
        color_id += 1
    plt.xlabel('Day ID', fontsize=font_size)
    plt.ylabel('Number of infectious people', fontsize=font_size)
    plt.yticks(fontsize=font_size)

    new_ticks = [i for i in range(1, len(x_ticks) + 1)]

    plt.xticks(x_ticks, new_ticks, fontsize=font_size)
    xvline_x_new = [xvline_x[7], xvline_x[14], xvline_x[21]]
    [plt.axvline(_x, linewidth=0.8, color='k',linestyle = '--') for _x in xvline_x_new]

    for i in range(len(xvline_x_new) + 1):
        off_set = 20
        if i == 0:
            x_pos = xvline_x_new[i] / 2 - off_set
        elif i>= len(xvline_x_new):
            x_pos = xvline_x_new[-1] + xvline_x_new[0] / 2 - off_set
        else:
            x_pos = (xvline_x_new[i-1] + xvline_x_new[i])/ 2 - off_set

        y_pos = 2900
        plt.text(x_pos, y_pos, 'Week ' + str(i+1), fontsize = font_size, alpha = 0.5)

    plt.legend(fontsize=font_size-1, ncol = 5, loc = 'upper center')
    plt.ylim([0,3600])
    plt.xlim([-1, total_sim_period+1])
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/compare_theory_sim_I_time_varying.png', dpi=200)
    else:
        plt.show()
    return case_used

def plot_E_I_compare_time_varying(data_thry_E , data_sim_E,data_thry_I, data_sim_I, save_fig = 0):
    last_row = data_thry_E.iloc[-1,:]
    want_case = [30,150,300,450,600]
    case_used = []
    for want_num in want_case:
        idx = np.argmin(np.abs(np.array(last_row) - want_num))
        case_used.append(idx)
    plot_E_compare_time_varying(data_thry_E , data_sim_E, case_used, save_fig = save_fig)
    plot_I_compare_time_varying(data_thry_I, data_sim_I, case_used, save_fig=save_fig)
if __name__ == '__main__':
    data_thry_E = pd.read_csv('../data/theoretical_model_E.csv',header = None)
    data_thry_I = pd.read_csv('../data/theoretical_model_I.csv',header = None)
    data_sim_E = pd.read_csv('../data/simulation_model_E.csv',header = None)
    data_sim_I = pd.read_csv('../data/simulation_model_I.csv',header = None)

    # plot_I_compare_scatter(data_thry_I, data_sim_I, save_fig = 0)
    # plot_E_compare_scatter(data_thry_E, data_sim_E, save_fig = 0)

    plot_E_I_compare_time_varying(data_thry_E , data_sim_E,data_thry_I, data_sim_I, save_fig = 1)

