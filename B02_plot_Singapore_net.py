import pandas as pd
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from decimal import Decimal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
from sklearn.linear_model import LinearRegression
import math


def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

# colors = sns.color_palette("Paired")
colors = sns.color_palette("muted")

def plot_demand_distribution(data, time_interval, save_fig):
    ####process data
    data['day_of_week'] = (data['date'] - 4) % 7  # because Aug 4 is monday # 0:monday
    data['time_interval'] = data['start_time'] // time_interval
    data_demand = data.groupby(['day_of_week','time_interval'])['P_id'].count().reset_index(drop=False)
    num_week = 3
    data_demand['demand'] = data_demand['P_id'] / num_week
    ############
    week_list = range(0,7)
    time_interval_list = range(0,24)
    x_ticks = []
    new_ticks = []
    time_id = 0
    data_demand['time_id'] = 0
    data_demand = data_demand.drop(columns = ['P_id'])
    row_new_list = []
    x_word_label = []
    for week in week_list:
        for time in time_interval_list:
            time_id += 1
            if len(data_demand.loc[(data_demand['day_of_week'] == week) & (data_demand['time_interval'] == time)]) == 0:
                row_new = pd.DataFrame({'day_of_week':[week], 'time_interval':[time], 'demand':[0], 'time_id':[time_id]})
                row_new_list.append(row_new)
            else:
                data_demand.loc[(data_demand['day_of_week'] == week) & (data_demand['time_interval'] == time),'time_id'] = time_id
            if time in [0,6,12,18]:
                x_ticks.append(time_id)
                new_ticks.append(time + 1)
                if time+1==13:
                    x_word_label.append(time_id)

    data_demand = pd.concat([data_demand] + row_new_list)
    data_demand = data_demand.sort_values(['time_id'])
    max_week_day_id = data_demand.loc[(data_demand['day_of_week'] == 5) & (data_demand['time_interval'] == 0),'time_id'].values[0]
    x_vline = list(data_demand.loc[(data_demand['time_interval'] == 0),'time_id'].values - 0.5)
    ###########Plot
    font_size = 16
    color_id = 0
    matplotlib.rcParams['font.size'] = font_size - 2
    fig,ax = plt.subplots(figsize=(20, 4))
    plt.plot(data_demand.loc[data_demand['time_id'] <= max_week_day_id,'time_id'],
             data_demand.loc[data_demand['time_id'] <= max_week_day_id,'demand'], marker = 'o', markersize = 6,
             color = colors[0],linewidth = 2)
    plt.plot(data_demand.loc[data_demand['time_id'] >= max_week_day_id, 'time_id'],
             data_demand.loc[data_demand['time_id'] >= max_week_day_id, 'demand'], marker='o', markersize=6,
             color=colors[1], linewidth = 2)
    count = 0
    for x in x_vline:
        count += 1
        if count == 1:
            continue
        plt.axvline(x, linestyle = '--', linewidth = 2,color = 'k')
    # add text
    word_y = max(data_demand['demand']) * 1.2
    count = 0
    week_name_list = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    for word_x in x_word_label:
        total_trip = data_demand.loc[data_demand['day_of_week']==count,'demand'].sum()
        total_trip_sci = "{:.1E}".format(Decimal(str(total_trip)))
        num_str = total_trip_sci.split('E')[0] + r'$\times 10^{}$'.format(total_trip_sci.split('E+')[1])
        plt.text(word_x-8, word_y, week_name_list[count] + ' (' +  num_str + ')' ,fontsize = font_size)
        count+=1
    plt.xlabel('Time', fontsize=font_size)
    plt.ylabel('Hourly ridership', fontsize=font_size)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    #plt.rc('font', size=font_size)
    ax.yaxis.major.formatter._useMathText = True
    plt.yticks(fontsize=font_size)
    plt.xticks(x_ticks, new_ticks, fontsize=font_size)
    plt.xlim([0.5,max(data_demand['time_id'])+0.5])
    plt.ylim([ - max(data_demand['demand']) * 0.05, max(data_demand['demand']) * 1.4])
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/demand_distribution.png', dpi = 200)
    else:
        plt.show()
    #plt.close()


def plot_trip_duration(data, save_fig):
    trip_duration_unit = 60 * 1 # 1 min
    data = data.loc[data['ride_duration']>0] # record error

    data['duration_interval'] = data['ride_duration'] // trip_duration_unit * trip_duration_unit / 60
    data['duration_interval'] += 1

    print('Average trip duration', data['duration_interval'].mean())
    print('Standard dev trip duration', data['duration_interval'].std())
    #
    data_P_TD = data.groupby(['duration_interval'])['P_id'].count().reset_index(drop=False)
    data_P_TD['p_td'] = data_P_TD['P_id'] / data_P_TD['P_id'].sum()



    # create some data to use for the plot
    font_size = 16
    matplotlib.rcParams['font.size'] = font_size-2
    fig,ax = plt.subplots(figsize=(7, 7))
    # the main axes is subplot(111) by default
    plt.plot(data_P_TD['duration_interval'], data_P_TD['p_td'], marker = 'o', markersize = 5,
             color = colors[2],linewidth = 2)
    plt.xlabel('TD: Trip duration (min)', fontsize=font_size)
    plt.ylabel(r'$P(TD)$', fontsize=font_size)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.major.formatter._useMathText = True
    x_ticks = list(range(0,120+20,20))
    plt.yticks(fontsize=font_size)
    plt.xticks(x_ticks, fontsize=font_size)

    plt.xlim([0,130])
    # plt.ylim([])
    plt.tight_layout()
    # this is an inset axes over the main axes

    inset_axes(ax, width=3, height=3, loc=3,bbox_to_anchor=(0.4,0.4,.3,.3), bbox_transform=ax.transAxes)
    plt.plot(data_P_TD['duration_interval'],data_P_TD['p_td'], marker = 'o', markersize = 5,
             color = colors[3],linewidth = 2)
    plt.yscale('log')

    # plt.title('Probability')
    plt.xlabel('TD: Trip duration (min)', fontsize=font_size)
    plt.ylabel(r'$P(TD)$ (log-scale)', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(x_ticks, fontsize=font_size)
    #============Fit and add parameters
    X = data_P_TD.loc[(data_P_TD['duration_interval']>=10) & (data_P_TD['duration_interval']<=120),'duration_interval'].values.reshape(-1,1)
    y = np.log(data_P_TD.loc[(data_P_TD['duration_interval']>=10) & (data_P_TD['duration_interval']<=120),'p_td'].values).reshape(-1,1)
    reg = LinearRegression().fit(X, y)
    _lambda = reg.coef_[0][0]
    word_x = 100 - 37
    word_y = 0.003
    plt.text(word_x, word_y, r'$\lambda_{td} = $' + str(round(-1/_lambda,2))+ ' min')
    x_pred = np.arange(10,130,1).reshape(-1,1)
    y_pred = reg.predict(x_pred) + 1 # -> move it a little bit upward
    plt.plot(x_pred.ravel(),np.exp(y_pred.ravel()), 'k--', linewidth = 1.5) #
    plt.xlim([0, 130])
    plt.ylim([ math.pow(10, -6), math.pow(10, -0.7)])
    #======================

    if save_fig == 1:
        plt.savefig('img/trip_duration_distribution.png', dpi = 200)
    else:
        plt.show()


def plot_trip_frequency(data, save_fig):
    data = data.loc[data['ride_duration']>0] # record error

    tic = time.time()
    data['week_id'] = (data['date'] - 4) // 7
    data['week_id'] += 1
    # data.loc[(data['date'] >= 4) & (data['date'] <= 10),'week_id'] = 1
    # data.loc[(data['date'] >= 11) & (data['date'] <= 17), 'week_id'] = 2
    # data.loc[(data['date'] >= 18) & (data['date'] <= 24), 'week_id'] = 3
    # data.loc[(data['date'] >= 25) & (data['date'] <= 31), 'week_id'] = 4
    print('Allocate week id time', time.time() - tic)
    data['num_trip'] = 1
    data_P_f = data.groupby(['P_id', 'week_id'])['num_trip'].count().reset_index(drop=False)
    data_P_f = data_P_f.groupby(['P_id'])['num_trip'].mean().reset_index(drop=False)

    num_trip_unit = 1
    print('Average trip per people', data_P_f['num_trip'].mean())
    print('Standard dev trip per people', data_P_f['num_trip'].std())
    data_P_f['num_trip_interval'] = data_P_f['num_trip'] // num_trip_unit * num_trip_unit
    data_P_f = data_P_f.groupby(['num_trip_interval'])['P_id'].count().reset_index(drop=False)
    data_P_f['p_f'] = data_P_f['P_id'] / data_P_f['P_id'].sum()



    # create some data to use for the plot
    trip_freq_lim = 60

    font_size = 16
    matplotlib.rcParams['font.size'] = font_size-2
    fig,ax = plt.subplots(figsize=(7, 7))
    # the main axes is subplot(111) by default
    plt.plot(data_P_f['num_trip_interval'], data_P_f['p_f'], marker = 'o', markersize = 5,
             color = colors[2],linewidth = 2)
    plt.xlabel(r'$f$' + ': Number of trips per week', fontsize=font_size)
    plt.ylabel(r'$P(f)$', fontsize=font_size)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.major.formatter._useMathText = True
    x_ticks = list(range(0,trip_freq_lim + 10,10))
    plt.yticks(fontsize=font_size)
    plt.xticks(x_ticks, fontsize=font_size)



    plt.xlim([0,trip_freq_lim + 1])
    # plt.ylim([])
    plt.tight_layout()
    # this is an inset axes over the main axes

    inset_axes(ax, width=3, height=3, loc=3,bbox_to_anchor=(0.4,0.4,.3,.3), bbox_transform=ax.transAxes)
    plt.plot(data_P_f['num_trip_interval'],data_P_f['p_f'], marker = 'o', markersize = 5,
             color = colors[3],linewidth = 2)
    plt.yscale('log')

    # plt.title('Probability')
    plt.xlabel(r'$f$' + ': Number of trips per week', fontsize=font_size)
    plt.ylabel(r'$P(f)$' + ' (log-scale)', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(x_ticks, fontsize=font_size)
    #============Fit and add parameters
    X = data_P_f.loc[(data_P_f['num_trip_interval']>=1) & (data_P_f['num_trip_interval']<=trip_freq_lim),'num_trip_interval'].values.reshape(-1,1)
    y = np.log(data_P_f.loc[(data_P_f['num_trip_interval']>=1) & (data_P_f['num_trip_interval']<=trip_freq_lim),'p_f'].values).reshape(-1,1)
    reg = LinearRegression().fit(X, y)
    _lambda = reg.coef_[0][0]
    word_x = trip_freq_lim - 35
    word_y = 0.02
    plt.text(word_x, word_y, r'$\lambda_{tf} = $' + str(round(-1/_lambda,2))+ ' week' + r'$^{-1}$')
    x_pred = np.arange(1,trip_freq_lim + 20,1).reshape(-1,1)
    y_pred = reg.predict(x_pred) + 0.8 # -> move it a little bit upward
    plt.plot(x_pred.ravel(),np.exp(y_pred.ravel()), 'k--', linewidth = 1.5) #
    plt.xlim([0, trip_freq_lim + 1])
    plt.ylim([math.pow(10,-5), math.pow(10,-0.2)])
    #======================

    if save_fig == 1:
        plt.savefig('img/trip_frequency_distribution.png', dpi = 200)
    else:
        plt.show()

if __name__ == '__main__':
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31

    TEST = False
    if TEST:
        sample_size = 100000
        sample_seed = 0
        with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(sample_seed) +'.pickle', 'rb') as handle:
            data = pickle.load(handle)
    else:
        data = pd.read_csv('../data/data_Aug_compressed.csv')


    data = data.loc[(data['date'] >= start_date) & (data['date'] <= end_date)]
    #====================
    # plot_demand_distribution(data, time_interval, save_fig=0)
    #====================
    plot_trip_duration(data, save_fig=1)
    #====================
    plot_trip_frequency(data, save_fig=1)