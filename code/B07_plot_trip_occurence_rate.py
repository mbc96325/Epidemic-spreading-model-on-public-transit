import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
colors = sns.color_palette("Set1")


def plot_compare(data, save_fig):
    color_id = -1
    font_size = 16
    plt.figure(figsize=(10, 6))
    marker_list = ['s','^','o','x','+']
    for case in range(data.shape[0]):
        color_id += 1
        case_info = data.iloc[case,0:3]
        name = ''
        if case_info['control_pt'] == 1:
            name += 'PT'
        if case_info['control_local'] == 1:
            if len(name)!=0:
                name += '+Local'
            else:
                name += 'Local'
        if case_info['control_global'] == 1:
            if len(name)!=0:
                name += '+Global'
            else:
                name += 'Global'
        data_plot = data.iloc[case,3:]
        percengtae = range(1,101,1)
        plt.plot(percengtae, data_plot[:], marker = marker_list[color_id], markersize = 5, linewidth = 1.5, label = name,color = colors[color_id])
        plt.plot([-1,102],[1,1], 'k--')

    plt.xlabel('Control percentage', fontsize=font_size)
    plt.ylabel('Equivalent ' + r'$R_0$', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    x_ticks = list(range(0,120,20))
    new_ticks = [str(x) + '%' for x in x_ticks]
    plt.xticks(x_ticks, new_ticks, fontsize=font_size)
    plt.legend(fontsize=font_size-1)
    plt.ylim([0.5,2.75])
    plt.xlim([-1, 100+1])
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/control_strategy.eps', dpi=200)
    else:
        plt.show()


if __name__ == '__main__':
    data = pd.read_csv('../data/imapct_of_trip_rate.csv')
    plot_compare(data, save_fig = 1)