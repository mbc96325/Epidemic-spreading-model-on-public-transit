import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


colors = sns.color_palette("muted")

def calculate_death_cured_rate(data,city):
    data = data.loc[data['cityName']==city]
    data['updateTime'] =  pd.to_datetime(data['updateTime'], format = '%Y-%m-%d %H:%M:%S')
    data['month'] = data['updateTime'].dt.month
    data['day'] = data['updateTime'].dt.day
    data_daily = data.groupby(['month','day'])[['city_confirmedCount','city_curedCount','city_deadCount']].\
        agg(lambda x:x.value_counts().index[0]).reset_index(drop=False)
    data_daily['key'] = list(range(len(data_daily)))
    data_daily['key'] += 1
    data_daily['key_next_day'] = data_daily['key'] - 1
    data_daily = data_daily.merge(data_daily[['city_confirmedCount','city_curedCount','city_deadCount','key_next_day']], left_on = ['key'], right_on = ['key_next_day'])
    data_daily['daily_add_confirmed'] = data_daily['city_confirmedCount_y'] - data_daily['city_confirmedCount_x']
    data_daily['daily_add_cured'] = data_daily['city_curedCount_y'] - data_daily['city_curedCount_x']
    data_daily['daily_add_dead'] = data_daily['city_deadCount_y'] - data_daily['city_deadCount_x']


    data_daily['daily_cured_prob'] = data_daily['daily_add_cured']/data_daily['city_confirmedCount_x']
    data_daily['daily_death_prob'] = data_daily['daily_add_dead'] / data_daily['city_confirmedCount_x']

    # plt.plot(data_daily['key'], data_daily['daily_cured_prob'],'k-^')
    # plt.show()
    # plt.close()
    # plt.plot(data_daily['key'], data_daily['daily_death_prob'],'k-^')
    # plt.show()
    # plt.close()
    # plt.plot(data_daily['key'], data_daily['daily_add_dead'],'k-^')
    # plt.show()
    # plt.close()
    return data_daily

def generate_new_x_ticks(data_daily, jump_date):
    max_key = data_daily['key'].max()
    x_ticks = []
    new_ticks = []
    for used_key in range(0, max_key+1, jump_date):
        x_ticks.append(data_daily['key'].iloc[used_key])
        month = data_daily['month'].iloc[used_key]
        day = data_daily['day'].iloc[used_key]
        new_ticks.append(str(month) + '/' + str(day))
    a=1
    return x_ticks, new_ticks


def plot_death_recover_confirmed(data_daily, save_fig):
    data_daily = data_daily.rename(columns={'city_confirmedCount_x':'confirmed_all','city_curedCount_x':'cured_all','city_deadCount_x':'dead_all'})
    font_size = 16
    fig,ax = plt.subplots(figsize=(9, 6))
    plt.plot(data_daily['key'], data_daily['confirmed_all']/1000, marker = 'o', markersize = 5,
             color = colors[0],linewidth = 2,label='Confirmed')
    plt.plot(data_daily['key'], data_daily['cured_all']/1000, marker = '^', markersize = 5,
             color = colors[1],linewidth = 2,label='Cured')
    plt.plot(data_daily['key'], data_daily['dead_all']/1000, marker = 's', markersize = 5,
             color = colors[2],linewidth = 2, label='Dead')
    plt.legend(fontsize = font_size)
    plt.ylim([0, 50])
    plt.xlabel('Time (month/date)', fontsize=font_size)
    plt.ylabel('Number of people (' + r'$\times 10^3$)', fontsize=font_size)
    x_ticks, new_ticks = generate_new_x_ticks(data_daily, jump_date = 3)
    plt.yticks(fontsize=font_size)
    plt.xticks(x_ticks, new_ticks, fontsize=font_size)
    #+=================================
    inset_axes(ax, width=3, height=1.5, loc=3, bbox_to_anchor=(0.1,0.35,.2,.2), bbox_transform=ax.transAxes)
    plt.plot(data_daily['key'], data_daily['cured_all']/1000, marker = '^', markersize = 5,
             color = colors[1],linewidth = 2,label='Cured')
    plt.plot(data_daily['key'], data_daily['dead_all']/1000, marker = 's', markersize = 5,
             color = colors[2],linewidth = 2, label='Dead')
    plt.ylim([0, 6])
    # plt.title('Probability')
    plt.xlabel('Time (month/date)', fontsize=font_size-3)
    plt.ylabel('Cured and dead (' + r'$\times 10^3$)', fontsize=font_size-3)
    plt.yticks(fontsize=font_size-3)
    x_ticks, new_ticks = generate_new_x_ticks(data_daily, jump_date=6)
    plt.xticks(x_ticks,new_ticks, fontsize=font_size-3)
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/Wuhan_situation.png', dpi = 200)
    else:
        plt.show()
    a=1

def plot_cure_death_rate(data_daily, save_fig):
    # data_daily = data_daily.rename(columns={'city_confirmedCount_x':'confirmed_all','city_curedCount_x':'cured_all','city_deadCount_x':'dead_all'})
    font_size = 16
    colors_new = sns.color_palette("Set2")
    fig,ax = plt.subplots(figsize=(10, 6))
    plt.plot(data_daily['key'], data_daily['daily_cured_prob'], marker = 'd', markersize = 8,
             color = colors_new[0],linewidth = 2,label='Cure rate')
    plt.plot(data_daily['key'], data_daily['daily_death_prob'], marker = '>', markersize = 8,
             color = colors_new[1],linewidth = 2,label='Death rate')
    plt.legend(fontsize = font_size)
    # plt.ylim([0, 50])
    plt.xlabel('Time (month/date)', fontsize=font_size)
    plt.ylabel('Daily cure and death probability', fontsize=font_size)
    x_ticks, new_ticks = generate_new_x_ticks(data_daily, jump_date = 3)
    plt.yticks(fontsize=font_size)
    plt.xticks(x_ticks, new_ticks, fontsize=font_size)
    #+=================================
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/cure_death_rate.png', dpi = 200)
    else:
        plt.show()
    a=1

# def singapore_data():
#     daily_increase = []

if __name__ == '__main__':
    data = pd.read_csv('../data/DXYArea2.csv')
    data_daily = calculate_death_cured_rate(data,city='武汉')
    # plot_death_recover_confirmed(data_daily, save_fig=1)
    plot_cure_death_rate(data_daily, save_fig=1)
    a=1