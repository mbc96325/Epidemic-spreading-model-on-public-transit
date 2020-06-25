import pandas as pd
import numpy as np
import time
import pickle

time_interval = 1 * 3600  # 1 hour
start_date = 4
end_date = 31
TEST = False
if TEST:
    sample_size = 100000
    sample_seed = 0
    with open('../data/data_Aug_' + str(start_date) + '_' + str(end_date) + '_' + str(sample_size) + '_seed_' + str(
            sample_seed) + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
else:
    data = pd.read_csv('../data/data_Aug_compressed.csv')

bus_info_dict = {}

for date in range(start_date, end_date+1):
    data_date = data.loc[data['date'] == date]
    bus_info = data_date.groupby(['boarding_stop', 'bus_id'])['start_time'].mean().reset_index()
    bus_info['start_time'] = np.round(bus_info['start_time'])
    bus_info['start_time'] = bus_info['start_time'].astype('int')
    bus_info_dict[date] = bus_info

with open('../data/data_bus_arrival_info' + str(start_date) + '_' + str(end_date) + '.pickle', 'wb') as handle:
    pickle.dump(bus_info_dict, handle)

