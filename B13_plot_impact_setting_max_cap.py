import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pickle
import random
colors = sns.color_palette("Paired")

def plot_impact(save_fig):
    results = pd.read_csv('../data/impact_of_close_bus.csv')
    # results_1 = pd.read_csv('../data/impact_of_close_bus.csv')
    # results_2 = pd.read_csv('../data/impact_of_close_bus.csv')
    results2 = pd.read_csv('../data/impact_of_max_bus_cap.csv')



    #================
    #get number of people controlled
    percentage_close_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    high_low = [0]
    with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(
            sample_seed) + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
    total_people = len(pd.unique(data['P_id']))

    max_aff_pax_per = 0.8
    for percentage_close in percentage_close_list:
        print('===============CURRENT percentage ' + str(percentage_close) + '=============')

        scenario_name = '' # high low
        with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size)
                  + '_seed_' + str(sample_seed) + '_close_bus_' + str(
            int(percentage_close * 100)) + scenario_name + '.pickle', 'rb') as handle:
            data = pickle.load(handle)
        per = (total_people - len(pd.unique(data['P_id'])))/total_people
        if per < max_aff_pax_per:
            high_low.append(per)

    affected_pax_cap = pd.read_csv('../data/number_of_affected_pax_max_cap.csv')
    affected_pax_cap = pd.concat([pd.DataFrame({'max_cap':[-1], 'num_affected_pax':[0]}), affected_pax_cap])
    affected_pax_cap['per'] = affected_pax_cap['num_affected_pax'] / total_people

    a=1


    font_size = 16
    fig, ax = plt.subplots(figsize=(10, 6))
    l1 =ax.plot(affected_pax_cap['per'], results2['R0'], marker = 's', markersize = 10, linewidth = 1.5,color = colors[1], label = r'$R_0$'+ ' (Max bus load)')
    results = results.iloc[:len(high_low)]
    l2 =ax.plot(high_low, results['R0'], marker='s', markersize=10, linewidth=1.5, color=colors[3],
             label= r'$R_0$'+ ' (H-L)')

    ax.set_xlabel('Percentage of affected passengers', fontsize=font_size)
    ax.set_ylabel('Equivalent ' + r'$R_0$', fontsize=font_size)

    y_ticks = list(np.arange(1.55, 1.77 + 0.03, 0.03))
    ax.set_yticks(y_ticks)
    y_tickslabel =[]
    for y in y_ticks:
        y_ti = str(round(y,2))
        if len(y_ti) <= 3:
            y_ti += '0'
        y_tickslabel.append(y_ti)
    ax.set_yticklabels(y_tickslabel, fontsize=font_size)
    x_ticks = list(np.arange(0, max_aff_pax_per + 0.1 , 0.1))
    new_ticks = [str(int(100*x)) + '%' for x in x_ticks]

    # ax.set_xticks(x_ticks, fontsize=font_size)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(new_ticks, fontsize=font_size)

    ax.set_ylim([1.55, 1.81])
    # plt.xlim([-5, 95])
    #================
    plt.legend(fontsize=font_size - 3, loc = 'upper center', ncol = 3)
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/impact_of_max_bus_load.png', dpi=200)
    else:
        plt.show()
    a=1

if __name__ == '__main__':
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31
    sample_size = 100000
    sample_seed = 0

    plot_impact(save_fig = 1)