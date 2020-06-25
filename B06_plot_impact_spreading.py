import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
colors = sns.color_palette("Paired")
def plot_I_compare(unit_move_list):
    color_id = 0
    max_I_list = []
    sim_result = pd.read_csv('output/sim_results_test1.csv')
    max_I = sim_result['I_t'].iloc[-1]
    max_I_list.append(max_I)
    for unit_move in unit_move_list:
        sim_result = pd.read_csv('output/sim_results_spreading'+ str(unit_move) +'.csv')
        max_I = sim_result['I_t'].iloc[-1]
        max_I_list.append(max_I)
        plt.plot(sim_result['time_interval'],sim_result['I_t'], label = str(unit_move), color = colors[color_id] )
        color_id += 1
    plt.legend()
    plt.show()

    plt.plot([0] + unit_move_list, max_I_list,'-^')
    plt.show()

def plot_time_compare(unit_move_list):
    old_pt = pd.read_csv('output/pt_total_contact_time_test1.csv')
    plt.plot(old_pt['time_interval'].iloc[:24], old_pt['total_contact_time'].iloc[:24], label='old')
    for unit_move in unit_move_list:
        new_pt = pd.read_csv('output/pt_total_contact_time_spreading'+ str(unit_move) + '.csv')
        plt.plot(new_pt['time_interval'].iloc[:24], new_pt['total_contact_time'].iloc[:24], label=str(unit_move))
    plt.legend()
    plt.show()


    #==================
    # old_pt = pd.read_csv('output/local_total_contact_time_test1.csv')
    # plt.plot(old_pt['time_interval'].iloc[:24], old_pt['total_contact_time'].iloc[:24], label='old')
    # for unit_move in unit_move_list:
    #     new_pt = pd.read_csv('output/local_total_contact_time_spreading'+ str(unit_move) + '.csv')
    #     plt.plot(new_pt['time_interval'].iloc[:24], new_pt['total_contact_time'].iloc[:24], label=str(unit_move))
    # plt.legend()
    # plt.show()

def plot_impact(save_fig):
    results = pd.read_csv('../data/impact_of_spreading.csv')
    font_size = 16
    plt.figure(figsize=(10, 6))
    plt.plot(results['Spreading'], results['R0'], marker = 's', markersize = 10, linewidth = 1.5,color = colors[3])
    # plt.plot([-1,102],[1,1], 'k--')
    plt.xlabel('Departure time flexibility (min)', fontsize=font_size)
    plt.ylabel('Equivalent ' + r'$R_0$', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    x_ticks = list(range(0,120,10))
    # new_ticks = [str(x) + '%' for x in x_ticks]
    plt.xticks(x_ticks, fontsize=font_size)
    # plt.legend(fontsize=font_size-1)
    plt.ylim([1.735, 1.765])
    # plt.xlim([-1, 100+1])
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/impact_of_spreading.png', dpi=200)
    else:
        plt.show()
    a=1

if __name__ == '__main__':
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31
    sample_size = 100000
    sample_seed = 0
    unit_move_list = list(range(10,90+10,10))
    # plot_time_compare(unit_move_list)
    # plot_I_compare(unit_move_list)
    plot_impact(save_fig = 1)