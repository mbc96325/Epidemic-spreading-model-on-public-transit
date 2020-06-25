import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

colors = sns.color_palette("muted")



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




def plot_E_I(save_fig):
    results = pd.read_excel('../data/status_quo.xlsx', header= None)
    N_I = results.iloc[0, 1:]
    N_E = results.iloc[1, 1:]


    total_sim_period = len(N_I)
    font_size = 16
    time_all = np.array(range(0,total_sim_period)) + 1
    x_ticks, new_ticks, xvline_x = generate_x_label(total_sim_period)
    fig,ax = plt.subplots(figsize=(15, 6))

    plt.plot(time_all, N_I, '-', markersize=4,linewidth = 2, color = colors[1], label = 'Infectious')

    plt.plot(time_all, N_E, '-', markersize=4,linewidth = 2, color = colors[2], label = 'Exposed')

    plt.xlabel('Day ID', fontsize=font_size)
    plt.ylabel('Number of people', fontsize=font_size)
    plt.yticks(fontsize=font_size)

    def process_date(date):
        if date<10:
            return '0' + str(date)
        else:
            return str(date)

    x_ticks_new = [x_ticks[i] for i in range(0,31-4+1,2)]
    new_ticks_2 = ['Aug ' + process_date(date) for date in range(4,31+1,2)]
    new_ticks_3 = [i for i in range(1, len(x_ticks) + 1)]
    plt.xticks(x_ticks, new_ticks_3, fontsize=font_size)
    xvline_x_new = [xvline_x[7], xvline_x[14], xvline_x[21]]
    [plt.axvline(_x, linewidth=0.8, color='k',linestyle = '--',alpha = 0.5) for _x in xvline_x_new]
    plt.legend(fontsize=font_size-1, loc = 'center right')
    # plt.ylim([0,900])
    plt.xlim([-1, total_sim_period+1])
    plt.tight_layout()

    for i in range(len(xvline_x_new) + 1):
        off_set = 20
        if i == 0:
            x_pos = xvline_x_new[i] / 2 - off_set
        elif i>= len(xvline_x_new):
            x_pos = xvline_x_new[-1] + xvline_x_new[0] / 2 - off_set
        else:
            x_pos = (xvline_x_new[i-1] + xvline_x_new[i])/ 2 - off_set

        y_pos = 3200
        plt.text(x_pos, y_pos, 'Week ' + str(i+1), fontsize = font_size, alpha = 0.5)


    #===========================add inset======================
    last_time = 24*7
    time_all_inset = time_all[-last_time:]
    inset_axes(ax, width=4.5, height=2.5, loc=3,bbox_to_anchor=(0.07,0.33,.3,.3), bbox_transform=ax.transAxes)
    plt.plot(time_all_inset,N_E[-last_time:], marker = 'o', markersize = 3,
             color = colors[2],linewidth = 1)

    # plt.title('Probability')
    # x_ticks_new_inset = [x_ticks[-7],x_ticks[-5],x_ticks[-3],x_ticks[-1]]
    x_ticks_new_inset = x_ticks[-7:]
    new_ticks_2_inset = ['Aug ' + process_date(date) for date in range(25,31+1,2)]
    new_ticks_3_inset = new_ticks_3[-7:]
    plt.xlabel('Day ID (Week 4)', fontsize=font_size-3)
    plt.ylabel('Number of exposed people', fontsize=font_size -3)
    plt.yticks(fontsize=font_size-3)
    plt.xticks(x_ticks_new_inset, new_ticks_3_inset, fontsize=font_size-3)
    xvline_x_2 = xvline_x[-8:]
    [plt.axvline(_x, linewidth=0.8, color='k', linestyle='--', alpha = 0.5) for _x in xvline_x_2]
    #======================


    if save_fig == 1:
        plt.savefig('img/status_quo_E_I.png', dpi=200)
    else:
        plt.show()
    # a=1

if __name__ == '__main__':
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31
    sample_size = 100000
    sample_seed = 0

    plot_E_I(save_fig = 1)