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
colors = sns.color_palette("muted")


def output_simple_data():
    data_used = []
    data_list = ['D:/Data/Singapore AFC/2014_AUG_01_15.csv'] #, 'D:/Data/Singapore AFC/2014_AUG_16_31.csv'
    for file in data_list:
        chunksize = 1e7
        iter_csv = pd.read_csv(file, iterator=True, chunksize=chunksize)
        used_col = ['CRD_NUM','TRAVEL_MODE','Bus_Trip_Num','BOARDING_STOP_STN',
                    'ALIGHTING_STOP_STN','RIDE_Start_Date','RIDE_Start_Time','Ride_Time']
        count = 0
        for chunk in iter_csv:
            # if count > 2:
            #     break
            count+=1
            print('current process', count, 'total', count * chunksize)
            chunk_new = chunk.loc[:,used_col]
            chunk_new = chunk_new.loc[chunk_new['TRAVEL_MODE'] == 'RTS'] # only train
            data_used.append(chunk_new)
            print('--------')
    data_used = pd.concat(data_used)
    return data_used

def conpressed_prcess(data):
    Card_id = pd.unique(data['CRD_NUM'])
    P_id_index = pd.DataFrame({'CRD_NUM': Card_id, 'P_id':range(1, len(Card_id) + 1)})
    data = data.merge(P_id_index, on = ['CRD_NUM'])
    data = data.drop(columns = ['CRD_NUM'])

    data['date_dt'] = pd.to_datetime(data['RIDE_Start_Date'], format='%Y-%m-%d')
    data['time_dt'] = pd.to_datetime(data['RIDE_Start_Time'], format='%H:%M:%S')
    data['start_time'] = data['time_dt'].dt.hour * 3600 + data['time_dt'].dt.minute * 60 + data['time_dt'].dt.second
    data['date'] = data['date_dt'].dt.day


    # check type
    # print(type(data['RIDE_Start_Date'].iloc[0]))
    # print(type(data['BOARDING_STOP_STN'].iloc[0]))

    data['day_week'] = data['date_dt'].dt.weekday
    data['ride_duration'] = data['Ride_Time']*60
    data['ride_duration'] = data['ride_duration'].apply(round)

    # generate new bus stop id
    bus_stop_id1 = set(pd.unique(data['BOARDING_STOP_STN']))
    bus_stop_id2 = set(pd.unique(data['ALIGHTING_STOP_STN']))
    bus_stop_id = list(bus_stop_id1.intersection(bus_stop_id2))

    bus_stop_id_index = pd.DataFrame({'Old_stop_id': bus_stop_id, 'bus_stop':range(1, len(bus_stop_id) + 1)})
    data = data.merge(bus_stop_id_index, left_on = ['BOARDING_STOP_STN'],right_on = ['Old_stop_id'])
    data = data.rename(columns = {'bus_stop':'boarding_stop'})
    data = data.merge(bus_stop_id_index, left_on=['ALIGHTING_STOP_STN'], right_on=['Old_stop_id'])
    data = data.rename(columns={'bus_stop': 'alighting_stop'})
    used_col = ['P_id', 'bus_id','date','start_time','ride_duration','boarding_stop','alighting_stop']

    data = data.loc[:, used_col]
    data.to_csv('../data/data_Aug_compressed_MTR.csv',index=False)
    # SAVE lookup table
    P_id_index.to_csv('../data/P_id_lookup_MTR.csv',index=False)
    bus_stop_id_index.to_csv('../data/MTR_station_id_lookup.csv', index=False)




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
        plt.savefig('img/demand_distribution_MTR.eps', dpi = 200)
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

    plt.xlim([0,120])
    # plt.ylim([])
    plt.tight_layout()
    # this is an inset axes over the main axes

    inset_axes(ax, width=2.6, height=2.6, loc=3,bbox_to_anchor=(0.47,0.47,.3,.3), bbox_transform=ax.transAxes)
    plt.plot(data_P_TD['duration_interval'],data_P_TD['p_td'], marker = 'o', markersize = 5,
             color = colors[3],linewidth = 2)
    plt.yscale('log')

    # plt.title('Probability')
    plt.xlabel('TD: Trip duration (min)', fontsize=font_size)
    plt.ylabel(r'$P(TD)$ (log-scale)', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(x_ticks, fontsize=font_size)
    #============Fit and add parameters
    X = data_P_TD.loc[(data_P_TD['duration_interval']>=40) & (data_P_TD['duration_interval']<=120),'duration_interval'].values.reshape(-1,1)
    y = np.log(data_P_TD.loc[(data_P_TD['duration_interval']>=40) & (data_P_TD['duration_interval']<=120),'p_td'].values).reshape(-1,1)
    reg = LinearRegression().fit(X, y)
    _lambda = reg.coef_[0][0]
    word_x = 100 - 53
    word_y = 0.05
    plt.text(word_x, word_y, r'$\lambda_{td} = $' + str(round(-1/_lambda,2))+ ' min')
    x_pred = np.arange(10,120,1).reshape(-1,1)
    y_pred = reg.predict(x_pred) + 1 # -> move it a little bit upward
    plt.plot(x_pred.ravel(),np.exp(y_pred.ravel()), 'k--', linewidth = 1.5) #
    plt.xlim([0, 120])
    plt.ylim([ math.pow(10, -6), math.pow(10, -0.7)])
    #======================

    if save_fig == 1:
        plt.savefig('img/trip_duration_distribution_MTR.eps', dpi = 200)
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
    trip_freq_lim = 40

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

    inset_axes(ax, width=3, height=3, loc=3,bbox_to_anchor=(0.42,0.42,.3,.3), bbox_transform=ax.transAxes)
    plt.plot(data_P_f['num_trip_interval'],data_P_f['p_f'], marker = 'o', markersize = 5,
             color = colors[3],linewidth = 2)
    plt.yscale('log')

    # plt.title('Probability')
    plt.xlabel(r'$f$' + ': Number of trips per week', fontsize=font_size)
    plt.ylabel(r'$P(f)$' + ' (log-scale)', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(x_ticks, fontsize=font_size)
    #============Fit and add parameters
    X = data_P_f.loc[(data_P_f['num_trip_interval']>=8) & (data_P_f['num_trip_interval']<=trip_freq_lim),'num_trip_interval'].values.reshape(-1,1)
    y = np.log(data_P_f.loc[(data_P_f['num_trip_interval']>=8) & (data_P_f['num_trip_interval']<=trip_freq_lim),'p_f'].values).reshape(-1,1)
    reg = LinearRegression().fit(X, y)
    _lambda = reg.coef_[0][0]
    word_x = trip_freq_lim - 25
    word_y = 0.02
    plt.text(word_x, word_y, r'$\lambda_{tf} = $' + str(round(-1/_lambda,2))+ ' week' + r'$^{-1}$')
    x_pred = np.arange(1,trip_freq_lim + 20,1).reshape(-1,1)
    y_pred = reg.predict(x_pred) + 1.2 # -> move it a little bit upward
    plt.plot(x_pred.ravel(),np.exp(y_pred.ravel()), 'k--', linewidth = 1.5) #
    plt.xlim([0, trip_freq_lim + 1])
    plt.ylim([math.pow(10,-7), math.pow(10,-0.2)])
    #======================

    if save_fig == 1:
        plt.savefig('img/trip_frequency_distribution_MTR.eps', dpi = 200)
    else:
        plt.show()


if __name__ == '__main__':
    ################
    # data_used = output_simple_data()
    # conpressed_prcess(data_used)
    ##################
    data = pd.read_csv('../data/data_Aug_compressed_MTR.csv')
    time_interval = 1*3600 # 1 hour
    start_date = 4
    end_date = 31
    plot_demand_distribution(data, time_interval, save_fig = 1)
    plot_trip_duration(data, save_fig = 1)
    plot_trip_frequency(data, save_fig = 1)

    ###############
    # generate_contract_duration():
