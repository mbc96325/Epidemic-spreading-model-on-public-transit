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
    results2 = pd.read_csv('../data/impact_of_close_bus_random.csv')
    results3 = pd.read_csv('../data/impact_of_close_bus_low_high.csv')
    font_size = 16
    fig, ax = plt.subplots(figsize=(10, 6))
    l1 =ax.plot(results['Close_percentage'], results['R0'], marker = 's', markersize = 10, linewidth = 1.5,color = colors[1], label = r'$R_0$'+ ' (H-L)')
    l2 =ax.plot(results3['Close_percentage'], results3['R0'], marker='s', markersize=10, linewidth=1.5, color=colors[3],
             label= r'$R_0$'+ ' (L-H)')
    l3 =ax.plot(results2['Close_percentage'], results2['R0'], marker='s', markersize=10, linewidth=1.5, color=colors[5],
             label= r'$R_0$'+ ' (Random)')
    #================
    #get number of people controlled
    percentage_close_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    high_low = [0]
    low_high = [0]
    random_ = [0]
    with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(
            sample_seed) + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
    total_people = len(pd.unique(data['P_id']))
    for percentage_close in percentage_close_list:
        print('===============CURRENT percentage ' + str(percentage_close) + '=============')

        scenario_name = 'low_high'
        with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size)
                  + '_seed_' + str(sample_seed) + '_close_bus_' + str(
            int(percentage_close * 100)) + scenario_name + '.pickle', 'rb') as handle:
            data = pickle.load(handle)
        low_high.append( (total_people - len(pd.unique(data['P_id'])))/total_people )

    for percentage_close in percentage_close_list:
        print('===============CURRENT percentage ' + str(percentage_close) + '=============')

        scenario_name = '' # high low
        with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size)
                  + '_seed_' + str(sample_seed) + '_close_bus_' + str(
            int(percentage_close * 100)) + scenario_name + '.pickle', 'rb') as handle:
            data = pickle.load(handle)
        high_low.append( (total_people - len(pd.unique(data['P_id'])))/total_people )

    for percentage_close in percentage_close_list:
        print('===============CURRENT percentage ' + str(percentage_close) + '=============')

        scenario_name = 'random'
        with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size)
                  + '_seed_' + str(sample_seed) + '_close_bus_' + str(
            int(percentage_close * 100)) + scenario_name + '.pickle', 'rb') as handle:
            data = pickle.load(handle)
        random_.append( (total_people - len(pd.unique(data['P_id'])))/total_people )
    ax.set_xlabel('Close percentage (%)', fontsize=font_size)
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
    x_ticks = list(range(0,100,10))
    new_ticks = [str(x) + '%' for x in x_ticks]

    # ax.set_xticks(x_ticks, fontsize=font_size)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(new_ticks, fontsize=font_size)

    ax.set_ylim([1.55, 1.81])
    plt.xlim([-5, 95])
    #================
    ax2 = ax.twinx()  #
    l4 = ax2.plot(results['Close_percentage'], high_low, linestyle = '--', marker = '^', markersize = 8, linewidth = 1.5,color = colors[0], label = 'Affected pax (H-L)')
    l5 = ax2.plot(results3['Close_percentage'], low_high, linestyle = '--',marker = '^', markersize=8, linewidth=1.5, color=colors[2],
             label='Affected pax (L-H)')
    l6 = ax2.plot(results2['Close_percentage'], random_, linestyle = '--',marker = '^', markersize=8, linewidth=1.5, color=colors[4],
             label='Affected pax (Random)')
    ax.set_xlabel('Close percentage (%)', fontsize=font_size)
    ax2.set_ylabel('Percentage of affected passengers', fontsize=font_size)
    #######
    y_ticks2 = list(np.arange(0, 1.2, 0.2))
    ax2.set_yticks(y_ticks2)
    new_ticks2 = [str(int(round(x*100))) + '%' for x in y_ticks2]
    ax2.set_yticklabels(new_ticks2, fontsize=font_size)
    ax2.set_ylim([-0.03,1.2])
    # plt.plot([-1,102],[1,1], 'k--')
    lns = l1 +   l4 + l2 +l5 + l3 + l6
    labs = [l.get_label() for l in lns]

    ax.legend(lns, labs, fontsize=font_size - 3, loc = 'upper center', ncol = 3)
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/impact_of_close_bus.png', dpi=200)
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