import pandas as pd
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import _constants
import random
import matplotlib
import powerlaw
import scipy.optimize as opt


flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
colors = sns.color_palette(flatui)

def construct_PT_encounter_net(data,time_int_s, time_int_e):
    #['P_id', 'bus_id', 'date', 'start_time', 'ride_duration', 'boarding_stop', 'alighting_stop']
    data = data.drop(columns = ['start_time'])
    data = data.rename(columns={'start_time_day': 'start_time', 'end_time_day': 'end_time'})
    data_net = data[['P_id','bus_id','start_time',
                     'end_time']].merge(data[['P_id','bus_id','start_time','end_time']],on = ['bus_id'])
    # WLOG, assume y is the late people
    data_net = data_net.loc[data_net['start_time_y']>=data_net['start_time_x']]
    # filter people without contact
    data_net = data_net.loc[data_net['end_time_x'] > data_net['start_time_y']]
    #
    data_net['time_int_s'] = time_int_s
    data_net['time_int_e'] = time_int_e
    data_net['left_bound_x'] = data_net[['start_time_x', 'time_int_s']].max(axis=1)
    data_net['right_bound_x'] = data_net[['end_time_x', 'time_int_e']].min(axis=1)
    data_net['left_bound_y'] = data_net[['start_time_y', 'time_int_s']].max(axis=1)
    data_net['right_bound_y'] = data_net[['end_time_y', 'time_int_e']].min(axis=1)
    data_net['contact_duration'] = data_net[['right_bound_x','right_bound_y']].min(axis=1) - data_net['left_bound_y']
    data_net = data_net.loc[data_net['contact_duration']>0]
    data_net = data_net.loc[data_net['P_id_x']!=data_net['P_id_y']]
    # data_net['w_ij'] = data_net['contact_duration'] / (time_int_e - time_int_s)
    # a=1
    return data_net[['P_id_x','P_id_y','contact_duration']]

def PT_connection(data,time_interval, time_used):
    data['start_time_day'] = data['start_time']
    data['end_time_day'] = data['start_time_day'] + data['ride_duration']
    count = 0
    PT_net = {}
    time_interval_list = [time_int*time_interval - time_interval for time_int in time_used]
    data_date = data
    for time_int in time_interval_list:
        time_int_id = time_used[count]
        print('current time interval id:', time_int_id)

        # give data for one time interval
        time_int_s = time_int
        time_int_e = time_int + time_interval
        data_time_int = data_date.loc[~((data_date['start_time_day'] > time_int_e)|
                                      (data_date['end_time_day']< time_int_s))]
        if len(data_time_int) > 0:
            data_net = construct_PT_encounter_net(data_time_int, time_int_s, time_int_e)
            data_net = data_net.groupby(['P_id_x', 'P_id_y'])['contact_duration'].sum().reset_index()
            data_net.loc[data_net['contact_duration'] > time_interval,
                         'contact_duration'] = time_interval  # should not happen, just for conservative consideration.

            data_net['time_int_id'] = time_int_id
            PT_net[time_int_id] = data_net

        else:
            PT_net[time_int_id] = pd.DataFrame({'P_id_x':[],'P_id_y':[],'contact_duration':[]})
        count += 1
    print('Finish PT encounter network')
    return PT_net


def construct_contact_net(data, time_int_s, time_int_e, start_time_name,end_time_name, od_name):
    #['P_id', 'bus_id', 'date', 'start_time', 'ride_duration', 'boarding_stop', 'alighting_stop']
    data = data.rename(columns = {start_time_name: 'start_time', end_time_name: 'end_time'})
    data_net = data[['P_id',od_name,'start_time',
                     'end_time']].merge(data[['P_id',od_name,'start_time','end_time']],on = [od_name])
    # WLOG, assume y is the late people
    data_net = data_net.loc[data_net['start_time_y']>=data_net['start_time_x']]
    # filter people without contact
    data_net = data_net.loc[data_net['end_time_x'] > data_net['start_time_y']]
    #
    data_net['time_int_s'] = time_int_s
    data_net['time_int_e'] = time_int_e
    data_net['left_bound_x'] = data_net[['start_time_x', 'time_int_s']].max(axis=1)
    data_net['right_bound_x'] = data_net[['end_time_x', 'time_int_e']].min(axis=1)
    data_net['left_bound_y'] = data_net[['start_time_y', 'time_int_s']].max(axis=1)
    data_net['right_bound_y'] = data_net[['end_time_y', 'time_int_e']].min(axis=1)
    data_net['contact_duration'] = data_net[['right_bound_x','right_bound_y']].min(axis=1) - data_net['left_bound_y']
    data_net['contact_duration'] = data_net['contact_duration']*0.5 # assume only stay half of the time at an O or D
    data_net['contact_duration'] = data_net['contact_duration'].apply(round)
    data_net['contact_duration'] = data_net['contact_duration'].astype('int')
    data_net = data_net.loc[data_net['P_id_x']!=data_net['P_id_y']]
    # data_net['w_ij'] = data_net['contact_duration'] / (time_int_e - time_int_s)
    # a=1
    return data_net[['P_id_x','P_id_y','contact_duration']]

def local_contact(data, time_interval, time_used, theta_l):
    # process OD
    # ==================
    # first get sample passenger based on theta_l # not exactly equal to sample from edges, sample from edges is no feasible
    passenger_list = pd.unique(data['P_id'])
    print('total_num_passengers', len(passenger_list))
    sample_size = int(round(len(passenger_list) * (np.sqrt(theta_l))))
    random.seed(0)
    used_p = random.sample(list(passenger_list), sample_size)
    print('num_sample_used',len(used_p))
    data = data.set_index(['P_id'])
    data = data.loc[used_p,:]
    data = data.reset_index(drop=False)
    #==================
    #
    data['start_time_day'] = data['start_time']
    data['end_time_day'] = data['start_time_day'] + data['ride_duration']
    data = data.drop(columns = ['start_time'])
    tic = time.time()
    data = data.sort_values(['P_id','start_time_day'])
    print('sort time', time.time() - tic)
    tic = time.time()
    data['trip_seq'] = data.groupby(['P_id']).cumcount()
    print('group time', time.time() - tic)
    data['trip_seq_next'] = data['trip_seq'] - 1
    data['trip_seq_last'] = data['trip_seq'] + 1
    data['o_end_time'] = data['start_time_day']
    data['d_start_time'] = data['end_time_day']
    data['o_start_time'] = data['o_end_time'] - 8 * 3600
    data['d_end_time'] = data['d_start_time'] + 8 * 3600
    #==== trip sequence define as consecutive trips within 24 hours
    print('Total trip seq records', len(data))
    data = data.loc[(data['o_end_time'] - data['o_start_time']< 24*3600)]
    data = data.loc[(data['d_end_time'] - data['d_start_time']< 24*3600)]
    print('After 24 h threshold trip seq records', len(data))
    #=======
    count = 0
    local_contact_net = {}
    time_interval_list = [time_int * time_interval - time_interval for time_int in time_used]
    data_date = data
    for time_int in time_interval_list:
        time_int_id = time_used[count]
        print('current time interval id:', time_int_id)
        # give data for one time interval
        time_int_s = time_int
        time_int_e = time_int + time_interval

        ########### o connection
        data_time_int_1 = data_date.loc[~((data_date['o_start_time'] > time_int_e)|
                                      (data_date['o_end_time']< time_int_s))]
        data_time_int_1 = data_time_int_1.rename(columns = {'o_start_time':'station_start_time','o_end_time':'station_end_time',
                                                            'boarding_stop':'station_id'})
        data_time_int_2 = data_date.loc[~((data_date['d_start_time'] > time_int_e) |
                                        (data_date['d_end_time'] < time_int_s))]
        data_time_int_2 = data_time_int_2.rename(
            columns={'d_start_time': 'station_start_time', 'd_end_time': 'station_end_time',
                     'alighting_stop':'station_id'})
        data_time_int = pd.concat([data_time_int_1,data_time_int_2],sort = False)
        data_time_int = data_time_int.drop(columns = ['o_start_time','o_end_time','d_start_time','d_end_time'])
        if len(data_time_int) > 0:
            data_net = construct_contact_net(data_time_int, time_int_s, time_int_e,
                                               start_time_name = 'station_start_time',end_time_name ='station_end_time',
                                               od_name ='station_id')
        else:
            data_net = pd.DataFrame({'P_id_x':[],'P_id_y':[],'contact_duration':[]})
        ########## merge to together
        data_net = data_net.groupby(['P_id_x','P_id_y'])['contact_duration'].sum().reset_index()
        data_net.loc[data_net['contact_duration']>time_interval,'contact_duration'] = time_interval # should not happen, just for conservative consideration.
        data_net['time_int_id'] = time_int_id
        local_contact_net[time_int_id] = data_net
        count += 1
    return local_contact_net


def degree_plot_main(data_used, net_name, x_lim,y_lim, save_name, time_used, lable_list, save_fig):
    fig, ax = plt.subplots(figsize=(7, 7))
    marker_list = ['o','s','^']
    font_size = 16
    count = -1
    for time_int in  time_used:
        count+=1
        net = data_used[net_name][time_int]
        all_pax = list(set(pd.unique(net['P_id_x'])).union(set(pd.unique(net['P_id_y']))))
        data_degree = pd.DataFrame({'P_id':all_pax})
        net['count'] = 1
        data_degree_left = data_degree.merge(net[['P_id_x','count']], left_on= ['P_id'], right_on = ['P_id_x'], how = 'left')
        data_degree_left = data_degree_left.rename(columns = {'count':'count_left'}).fillna(0)
        data_degree_left = data_degree_left.groupby(['P_id'])['count_left'].sum().reset_index(drop=False)

        data_degree_right = data_degree.merge(net[['P_id_y', 'count']], left_on=['P_id'], right_on=['P_id_y'], how='left')
        data_degree_right = data_degree_right.rename(columns={'count': 'count_right'}).fillna(0)
        data_degree_right = data_degree_right.groupby(['P_id'])['count_right'].sum().reset_index(drop=False)
        data_degree_all = data_degree_left.merge(data_degree_right, on =['P_id'])
        data_degree_all['degree'] = data_degree_all['count_left'] + data_degree_all['count_right']
        data_degree_group = data_degree_all.groupby(['degree'])['P_id'].count().reset_index(drop=False)
        data_degree_group['p_k'] = data_degree_group['P_id'] / data_degree_group['P_id'].sum()
        #===========================================

        matplotlib.rcParams['font.size'] = font_size - 2
        # the main axes is subplot(111) by default
        # plt.plot(data_degree_group['degree'], data_degree_group['p_k'], marker=marker_list[count], markersize=5,
        #          color=colors[count])
        plt.scatter(data_degree_group['degree'], data_degree_group['p_k'], marker=marker_list[count], s=30,
                 edgecolors=colors[count], facecolors = 'none',linewidth=1, label = lable_list[count])

    plt.xlabel(r'$k$' + ': Degree', fontsize=font_size)
    plt.ylabel(r'$P(k) $ (log-scale)', fontsize=font_size)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.major.formatter._useMathText = True
    plt.yscale('log')
    # plt.xscale('log')
    # x_ticks = list(range(0, trip_freq_lim + 10, 10))
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.legend(fontsize=font_size)
    # if len(x_lim)>0:
    #     plt.xlim(x_lim)
    if len(y_lim) > 0:
        plt.ylim(y_lim)
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/' + save_name + '.eps', dpi=200)
    else:
        plt.show()


def degree_plot_main_cum(data_used, net_name, x_lim,y_lim, save_name, time_used, lable_list, save_fig):
    fig, ax = plt.subplots(figsize=(7, 7))
    marker_list = ['o','s','^']
    font_size = 16
    count = -1
    for time_int in  time_used:
        count+=1
        net = data_used[net_name][time_int]
        all_pax = list(set(pd.unique(net['P_id_x'])).union(set(pd.unique(net['P_id_y']))))
        data_degree = pd.DataFrame({'P_id':all_pax})
        net['count'] = 1
        data_degree_left = data_degree.merge(net[['P_id_x','count']], left_on= ['P_id'], right_on = ['P_id_x'], how = 'left')
        data_degree_left = data_degree_left.rename(columns = {'count':'count_left'}).fillna(0)
        data_degree_left = data_degree_left.groupby(['P_id'])['count_left'].sum().reset_index(drop=False)

        data_degree_right = data_degree.merge(net[['P_id_y', 'count']], left_on=['P_id'], right_on=['P_id_y'], how='left')
        data_degree_right = data_degree_right.rename(columns={'count': 'count_right'}).fillna(0)
        data_degree_right = data_degree_right.groupby(['P_id'])['count_right'].sum().reset_index(drop=False)
        data_degree_all = data_degree_left.merge(data_degree_right, on =['P_id'])
        data_degree_all['degree'] = data_degree_all['count_left'] + data_degree_all['count_right']
        data_degree_group = data_degree_all.groupby(['degree'])['P_id'].count().reset_index(drop=False)

        data_degree_group['F_k'] = data_degree_group.loc[::-1, 'P_id'].cumsum()[::-1]
        data_degree_group['F_k'] /= sum(data_degree_group['P_id'])
        #===========================================

        matplotlib.rcParams['font.size'] = font_size - 2
        # the main axes is subplot(111) by default
        # plt.plot(data_degree_group['degree'], data_degree_group['p_k'], marker=marker_list[count], markersize=5,
        #          color=colors[count])
        plt.scatter(data_degree_group['degree'], data_degree_group['F_k'], marker=marker_list[count], s=30,
                 edgecolors=colors[count], facecolors = 'none',linewidth=1, label = lable_list[count])

    plt.xlabel(r'$k$' + ': Degree', fontsize=font_size)
    plt.ylabel(r'$F(k) $ (log-scale)', fontsize=font_size)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.major.formatter._useMathText = True
    plt.yscale('log')
    plt.xscale('log')
    # x_ticks = list(range(0, trip_freq_lim + 10, 10))
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.legend(fontsize=font_size)
    # if len(x_lim)>0:
    #     plt.xlim(x_lim)
    if len(y_lim) > 0:
        plt.ylim(y_lim)
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/' + save_name + '.png', dpi=200)
    else:
        plt.show()


def degree_plot_with_cut_thresh(data_used, net_name, x_lim,y_lim, save_name, time_used, lable_list, degree_limit,save_fig):
    fig, ax = plt.subplots(figsize=(7, 7))
    marker_list = ['o','s','^']
    font_size = 16
    count = -1
    #
    def func(k, k_sat, gamma, k_cut):
        return (k + k_sat) ** (-gamma) * np.exp(- k / k_cut)


    unit_bin = 10
    x_lim[0] = unit_bin
    log_unit = 2
    for time_int in  time_used:
        count+=1
        net = data_used[net_name][time_int]
        all_pax = list(set(pd.unique(net['P_id_x'])).union(set(pd.unique(net['P_id_y']))))
        data_degree = pd.DataFrame({'P_id':all_pax})
        net['count'] = 1
        data_degree_left = data_degree.merge(net[['P_id_x','count']], left_on= ['P_id'], right_on = ['P_id_x'], how = 'left')
        data_degree_left = data_degree_left.rename(columns = {'count':'count_left'}).fillna(0)
        data_degree_left = data_degree_left.groupby(['P_id'])['count_left'].sum().reset_index(drop=False)

        data_degree_right = data_degree.merge(net[['P_id_y', 'count']], left_on=['P_id'], right_on=['P_id_y'], how='left')
        data_degree_right = data_degree_right.rename(columns={'count': 'count_right'}).fillna(0)
        data_degree_right = data_degree_right.groupby(['P_id'])['count_right'].sum().reset_index(drop=False)
        data_degree_all = data_degree_left.merge(data_degree_right, on =['P_id'])
        data_degree_all['degree'] = data_degree_all['count_left'] + data_degree_all['count_right']
        data_degree_all_before = data_degree_all.copy()
        data_degree_all =  data_degree_all.loc[ data_degree_all['degree'] < degree_limit]
        data = np.array(data_degree_all['degree'])

        # largest = x_lim[1]
        # d = 0
        # bins_edge = [1]
        # while 1:
        #     d +=1
        #     bins_edge.append(log_unit**d)
        #     if log_unit**d > largest:
        #         break
        # # print(data)
        # bins,edge = np.histogram(data, bins=bins_edge)


        data_degree_all['degree']  = data_degree_all['degree'] // unit_bin * unit_bin + 1
        data_degree_group = data_degree_all.groupby(['degree'])['P_id'].count().reset_index(drop=False)

        data_degree_group['p_k'] = data_degree_group['P_id'] / data_degree_group['P_id'].sum()

        data_degree_all_before['degree'] = data_degree_all_before['degree'] // unit_bin * unit_bin + 1

        data_degree_group_before = data_degree_all_before.groupby(['degree'])['P_id'].count().reset_index(drop=False)

        data_degree_group_before['p_k'] = data_degree_group_before['P_id'] / data_degree_group_before['P_id'].sum()

        # P_k = bins / data_degree_group['P_id'].sum()

        # data = data_degree_group['p_k']
        # fit = powerlaw.Fit(data, discrete=True)
        # print(fit.truncated_power_law.parameter1)
        # print(fit.truncated_power_law.parameter2)
        # FigCCDFmax = fit.plot_ccdf(color='b', label=r"Empirical, no $x_{max}$")
        # fit.power_law.plot_ccdf(color='b', linestyle='--', ax=FigCCDFmax, label=r"Fit, no $x_{max}$")
        # fit = powerlaw.Fit(data, discrete=True, xmax=1000)
        # fit.plot_ccdf(color='r', label=r"Empirical, $x_{max}=1000$")
        # fit.power_law.plot_ccdf(color='r', linestyle='--', ax=FigCCDFmax, label=r"Fit, $x_{max}=1000$")
        # FigCCDFmax.set_ylabel(u"p(X≥x)")
        # FigCCDFmax.set_xlabel(r"Word Frequency")
        # handles, labels = FigCCDFmax.get_legend_handles_labels()
        # leg = FigCCDFmax.legend(handles, labels, loc=3)
        # leg.draw_frame(False)
        #
        # plt.show()
        #
        # data_degree_group['generated_p_k'] = func(data_degree_group['degree'], popt[0],popt[1],popt[2])


        # data_degree_group['p_k_tilde'] = data_degree_group['p_k'] * np.exp(data_degree_group['degree']/k_cut)
        # data_degree_group['degree_tilde'] = data_degree_group['degree'] + k_sat



        data_degree_group['F_k'] = data_degree_group.loc[::-1, 'p_k'].cumsum()[::-1]
        #
        popt, pcov = opt.curve_fit(func, data_degree_group['degree'], data_degree_group['F_k'],p0 = [100,0.5,300])
        k_cut = popt[2]
        k_sat = popt[0]
        data_degree_group_before['generated_p_k'] = func(data_degree_group_before['degree'], popt[0],popt[1],popt[2])
        data_degree_group_before['generated_F_k'] = data_degree_group_before.loc[::-1, 'generated_p_k'].cumsum()[::-1]




        #===========================================

        matplotlib.rcParams['font.size'] = font_size - 2
        # the main axes is subplot(111) by default
        # plt.plot(data_degree_group['degree'], data_degree_group['p_k'], marker=marker_list[count], markersize=5,
        #          color=colors[count])
        plt.scatter(data_degree_group['degree'], data_degree_group['F_k'], marker=marker_list[count], s=30,
                 edgecolors=colors[count], facecolors = 'none',linewidth=1, label = lable_list[count])

        # plt.scatter(bins, P_k, marker=marker_list[count], s=30,
        #          edgecolors=colors[count], facecolors = 'none',linewidth=1, label = lable_list[count])

        # x = np.array(data_degree_group['degree'].copy())


        #
        x = data_degree_group_before.loc[data_degree_group_before['degree'] >= x_lim[0],'degree']
        y = data_degree_group_before.loc[data_degree_group_before['degree'] >= x_lim[0], 'generated_p_k']
        plt.plot(x, y, '--', color = colors[count], alpha = 0.5, linewidth = 1.5)

    plt.xlabel(r'$k$' + ': Degree (log-scale)', fontsize=font_size)
    plt.ylabel(r'$F(k) $ (log-scale)', fontsize=font_size)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.major.formatter._useMathText = True
    plt.yscale('log')
    plt.xscale('log')
    # x_ticks = list(range(0, trip_freq_lim + 10, 10))
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.legend(fontsize=font_size)
    if len(x_lim)>0:
        plt.xlim(x_lim)
    if len(y_lim) > 0:
        plt.ylim(y_lim)
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/' + save_name + '.eps', dpi=200)
    else:
        plt.show()

def plot_degree_distribution(data_used, time_used, lable_list, save_fig):
    ############PT
    ##Weekday pt
    degree_limit = 1e3
    save_name = 'degree_distribution_weekday_pt'
    x_lim = [10, degree_limit + 1]
    y_lim = [1e-6, 1 * 10 ** (-1)]
    net_name = 'weekday_pt'
    degree_plot_main(data_used, net_name, x_lim, y_lim, save_name, time_used, lable_list, save_fig)
    ###Weekend pt
    save_name = 'degree_distribution_weekend_pt'
    net_name = 'weekend_pt'
    degree_plot_main(data_used, net_name, x_lim, y_lim, save_name, time_used, lable_list, save_fig)

    # ############Local
    # degree_limit = 1e3
    # save_name = 'degree_distribution_weekday_local'
    # x_lim = [10, degree_limit + 1]
    # y_lim = [1e-4, 1 * 10 ** (-1)]
    # net_name = 'weekday_local'
    # degree_plot_main(data_used, net_name, x_lim, y_lim, save_name,time_used, lable_list, save_fig)
    # ###Weekend pt
    # save_name = 'degree_distribution_weekend_local'
    # net_name = 'weekend_local'
    # degree_plot_main(data_used, net_name, x_lim, y_lim, save_name,time_used, lable_list, save_fig)

    # ############Local cumulative
    degree_limit = 700
    save_name = 'CCDF_degree_distribution_weekday_local'
    x_lim = [10, 1e3]
    y_lim = [1e-2, 1e0+0.1]
    # k_sat = 1e2
    # k_cut = 1e3
    net_name = 'weekday_local'
    degree_plot_with_cut_thresh(data_used, net_name, x_lim, y_lim, save_name,time_used, lable_list,degree_limit, save_fig)
    ###Weekend pt
    save_name = 'CCDF_degree_distribution_weekend_local'
    net_name = 'weekend_local'
    degree_plot_with_cut_thresh(data_used, net_name, x_lim, y_lim, save_name,time_used, lable_list,degree_limit, save_fig)


def contact_time_plot_main(data_used, net_name, x_lim,y_lim, save_name, time_used, lable_list, save_fig):
    fig, ax = plt.subplots(figsize=(7, 7))
    marker_list = ['d','h','v']
    font_size = 16
    count = -1
    contact_duration_unique = 60 * 1  # 1 min
    for time_int in  time_used:
        count+=1
        net = data_used[net_name][time_int]

        net = net.loc[net['contact_duration'] > 0]  # record error
        net['edge_id'] = list(range(len(net)))
        net['duration_interval'] = net['contact_duration'] // contact_duration_unique * contact_duration_unique / 60
        net['duration_interval'] += 1 # ceil round
        net.loc[net['duration_interval']>60,'duration_interval'] = 60
        #
        data_P_CD = net.groupby(['duration_interval'])['edge_id'].count().reset_index(drop=False)
        data_P_CD['p_cd'] = data_P_CD['edge_id'] / data_P_CD['edge_id'].sum()

        #===========================================

        matplotlib.rcParams['font.size'] = font_size - 2
        # the main axes is subplot(111) by default
        # plt.plot(data_degree_group['degree'], data_degree_group['p_k'], marker=marker_list[count], markersize=5,
        #          color=colors[count])
        plt.scatter(data_P_CD['duration_interval'], data_P_CD['p_cd'], marker=marker_list[count], s=30,
                 edgecolors=colors[count], facecolors = 'none',linewidth=1, label = lable_list[count])

    plt.xlabel(r'$CD$' + ': Contact duration (min)', fontsize=font_size)
    plt.ylabel(r'$P(CD) $ (log-scale)', fontsize=font_size)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.major.formatter._useMathText = True
    plt.yscale('log')
    # plt.xscale('log')
    # x_ticks = list(range(0, trip_freq_lim + 10, 10))
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.legend(fontsize=font_size)
    if len(x_lim)>0:
        plt.xlim(x_lim)
    if len(y_lim) > 0:
        plt.ylim(y_lim)
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/' + save_name + '.png', dpi=200)
    else:
        plt.show()


def plot_contact_time_distribution(data_used, time_used, lable_list, save_fig):
    ############PT
    #Weekday pt
    degree_limit = 63
    save_name = 'contact_time_distribution_weekday_pt'
    x_lim = [0, degree_limit + 1]
    y_lim = [1e-5, 1 * 10 ** (-0.7)]
    net_name = 'weekday_pt'
    contact_time_plot_main(data_used, net_name, x_lim, y_lim, save_name, time_used, lable_list, save_fig)
    ###Weekend pt
    save_name = 'contact_time_distribution_weekend_pt'
    net_name = 'weekend_pt'
    contact_time_plot_main(data_used, net_name, x_lim, y_lim, save_name, time_used, lable_list, save_fig)

    ############Local
    degree_limit = 63
    save_name = 'contact_time_distribution_weekday_local'
    x_lim = [0, degree_limit + 1]
    y_lim = [1e-4, 1 * 10 ** (-0)]
    net_name = 'weekday_local'
    contact_time_plot_main(data_used, net_name, x_lim, y_lim, save_name,time_used, lable_list, save_fig)
    ###Weekend pt
    save_name = 'contact_time_distribution_weekend_local'
    net_name = 'weekend_local'
    contact_time_plot_main(data_used, net_name, x_lim, y_lim, save_name,time_used, lable_list, save_fig)


def generate_data(data, date_used, time_interval, time_used, theta_l, TEST):
    data_all = {}
    #weekday
    data_used = data.loc[data['date'] == date_used['weekday']]
    PT_net = PT_connection(data_used, time_interval, time_used)
    data_all['weekday_pt'] = PT_net
    local_contact_net = local_contact(data_used, time_interval, time_used, theta_l)
    data_all['weekday_local'] = local_contact_net

    #weekend
    data_used = data.loc[data['date'] == date_used['weekend']]
    PT_net = PT_connection(data_used, time_interval, time_used)
    data_all['weekend_pt'] = PT_net
    local_contact_net = local_contact(data_used, time_interval, time_used, theta_l)
    data_all['weekend_local'] = local_contact_net
    if TEST:
        with open('../data/data_for_degree_and_contact_time_distribution_plotting_TEST' + '.pickle', 'wb') as handle:
            pickle.dump(data_all, handle)
    else:
        with open('../data/data_for_degree_and_contact_time_distribution_plotting' + '.pickle', 'wb') as handle:
            pickle.dump(data_all, handle)


if __name__ == '__main__':
    time_interval = 1*3600 # 1 hour
    time_used = [9, 13, 19]  # 8-9  12-13 18-19
    lable_list = ['8:00-9:00','12:00-13:00', '18:00-19:00']
    GENERATE = False
    TEST = False
    if GENERATE:
        tic = time.time()
        if TEST:
            sample_size = 100000
            sample_seed = 0
            start_date = 4
            end_date = 31
            with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(
                    sample_size) + '_seed_' + str(sample_seed) + '.pickle', 'rb') as handle:
                data = pickle.load(handle)
        else:
            data = pd.read_csv('../data/data_Aug_compressed.csv')
        print('total trips', len(data))
        print('load file time',time.time() - tic)
        date_used = {}
        date_used['weekday'] = 21  # aug 21
        date_used['weekend'] = 24  # aug 24
        theta_l = _constants.theta_l # cannot be too big, otherwise unable to generate local net
        generate_data(data, date_used, time_interval, time_used, theta_l, TEST)
        print('finish generate data')
    else:
        tic = time.time()
        if TEST:
            with open('../data/data_for_degree_and_contact_time_distribution_plotting_TEST' + '.pickle', 'rb') as handle:
                data_used = pickle.load(handle)
        else:
            with open('../data/data_for_degree_and_contact_time_distribution_plotting' + '.pickle', 'rb') as handle:
                data_used = pickle.load(handle)
        print('load data time', time.time() - tic)
        #=================
        plot_degree_distribution(data_used, time_used, lable_list, save_fig = 1)
        #=================
        plot_contact_time_distribution(data_used, time_used, lable_list, save_fig=1)