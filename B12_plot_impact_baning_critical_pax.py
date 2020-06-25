import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pickle
import time
colors = sns.color_palette("Paired")

def plot_impact(save_fig):
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31
    sample_size = 100000
    sample_seed = 0

    k_core_test = [8, 7, 6, 5, 4, 3]  # need

    date_list = list(range(start_date, end_date + 1, 1))  #
    with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(
            sample_seed) + '.pickle', 'rb') as handle:
        data_old = pickle.load(handle)

    data_new = {}
    for k_num_filter in k_core_test:
        save_file_name = '_k_core_' + str(int(k_num_filter))

        print('===============CURRENT k_core' + str(k_num_filter) + '=============')

        tic = time.time()
        with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(
                sample_seed) + '_k_core_' + str(int(k_num_filter)) + '.pickle', 'rb') as handle:
            data_new[k_num_filter] = pickle.load(handle)

    # ger num of decrease pey day
    num_decrease = {}
    total_pax = []
    count = 0
    for k_num_filter in k_core_test:
        count += 1
        col_name = 'core_' + str(k_num_filter)
        num_decrease[col_name] = []
        data_new_kcore = data_new[k_num_filter]
        for day in date_list:
            day_new_kcore_day = data_new_kcore.loc[data_new_kcore['date']==day]
            day_old_day = data_old.loc[data_old['date']==day]
            old_num = len(pd.unique(day_old_day['P_id']))
            new_num = len(pd.unique(day_new_kcore_day['P_id']))
            if count == 1:
                total_pax.append(old_num)
            control_num = old_num - new_num
            num_decrease[col_name].append(control_num)
    num_decrease['total_pax'] = total_pax
    num_decrease_df = pd.DataFrame(num_decrease)
    control_percentage_list = [0]
    for k_num_filter in k_core_test:
        count += 1
        col_name = 'core_' + str(k_num_filter)
        num_decrease_df['avg_dcrease'] = num_decrease_df[col_name] / num_decrease_df['total_pax']
        control_percentage = num_decrease_df['avg_dcrease'].mean()
        # print('total control k=', k_num_filter, 'is', num_decrease_df['avg_dcrease'].mean())
        control_percentage_list.append(control_percentage)

    k_core_results = pd.read_csv('../data/impact_of_k_core.csv')
    k_core_results['people_reduction'] = control_percentage_list
    k_core_results['people_reduction'] *= 100
    k_core_results['index'] = list(range(len(k_core_results)))
    #=======================
    font_size = 16
    fig, ax = plt.subplots(figsize=(10, 6))
    l1 = ax.plot(k_core_results['index'], k_core_results['R0'], marker = 's', markersize = 10, linewidth = 1.5,color = colors[1], label = r'$R_0$ ' + '(k-core)')
    l2 = ax.plot(k_core_results['index'], k_core_results['R0_rand'], marker='s', markersize=10, linewidth=1.5,
                 color=colors[5], label=r'$R_0$ ' + '(Random)')
    y_ticks = list(np.arange(1.55, 1.77 + 0.03, 0.03))
    ax.set_yticks(y_ticks)
    y_tickslabel =[]
    for y in y_ticks:
        y_ti = str(round(y,2))
        if len(y_ti) <= 3:
            y_ti += '0'
        y_tickslabel.append(y_ti)
    ax.set_yticklabels(y_tickslabel, fontsize=font_size)
    x_ticks = list(k_core_results['index'])
    new_ticks = list(k_core_results['k_core'])

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(new_ticks, fontsize=font_size)
    ax.set_ylim([1.53, 1.78])

    ax.set_xlabel('Controlled k-core', fontsize=font_size)
    ax.set_ylabel('Equivalent ' + r'$R_0$', fontsize=font_size)
    #================
    ax2 = ax.twinx()  #

    # plt.xlim([-5, 95])
    l3 = ax2.plot(k_core_results['index'], k_core_results['people_reduction'], linestyle = '--', marker='^', markersize=8, linewidth=1.5, color=colors[3],
             label= 'Isolated pax')


    y_ticks = list(np.arange(0,120,20))
    ax2.set_yticks(y_ticks)
    y_tickslabel = [str(int(round(x))) + '%' for x in y_ticks]

    ax2.set_yticklabels(y_tickslabel, fontsize=font_size)
    ax2.set_ylim([-5,100])
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(new_ticks, fontsize=font_size)
    ax2.set_ylabel('Percentage of isolated passengers', fontsize=font_size)

    lns = l1 +   l2 + l3
    labs = [l.get_label() for l in lns]

    ax.legend(lns, labs, fontsize=font_size , loc = 'center left')
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/impact_k_core.png', dpi=200)
    else:
        plt.show()

if __name__ == '__main__':
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31
    sample_size = 100000
    sample_seed = 0

    ##======================GENERATE NETWORK==========
    # date_used = 21
    # with open('../data/PT_eco_net_' + str(sample_size) + '_seed_' + str(sample_seed) + '.pickle', 'rb') as handle:
    #     pt_net = pickle.load(handle)
    # generate_edge(start_date, end_date, pt_net, date_used)

    plot_impact(save_fig = 1)
