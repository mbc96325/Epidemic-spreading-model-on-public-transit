import pandas as pd

def output_simple_data():
    data_used = []
    data_list = ['../data/2014_AUG_01_15.csv', '../data/2014_AUG_16_31.csv']
    for file in data_list:
        chunksize = 1e7
        iter_csv = pd.read_csv(file, iterator=True, chunksize=chunksize)
        used_col = ['CRD_NUM','Srvc_Number','BUS_REG_NUM','Direction','Bus_Trip_Num','BOARDING_STOP_STN',
                    'ALIGHTING_STOP_STN','RIDE_Start_Date','RIDE_Start_Time','Ride_Time']
        count = 0
        for chunk in iter_csv:
            # if count > 2:
            #     break
            count+=1
            print('current process', count, 'total', count * chunksize)
            chunk_new = chunk.loc[:,used_col]
            chunk_new = chunk_new.dropna() # only bus records
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

    bus_id_col = ['Srvc_Number', 'BUS_REG_NUM', 'Bus_Trip_Num', 'Direction']
    bus_id = data.groupby(bus_id_col)['P_id'].first()
    bus_id = bus_id.reset_index(drop=False)
    bus_id['bus_id'] = range(1, len(bus_id) + 1)
    bus_id = bus_id.drop(columns = ['P_id'])
    data = data.merge(bus_id[['bus_id'] + bus_id_col], on = bus_id_col)


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
    data.to_csv('../data/data_Aug_compressed.csv',index=False)
    # SAVE lookup table
    P_id_index.to_csv('../data/P_id_lookup.csv',index=False)
    bus_stop_id_index.to_csv('../data/bus_stop_id_lookup.csv', index=False)
    bus_id.to_csv('../data/bus_id_lookup.csv', index=False)

if __name__ == '__main__':
    data_used = output_simple_data()
    conpressed_prcess(data_used)