import pandas as pd
import pyarrow.parquet as pq

flag10 = False
flag30 = True

if flag10 == True:
    base_path = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\10G_data_new\part-%05d.parquet'

    #batch_size = 100_000
    dfs = []

    for i in range(8):
        file_path = base_path % i
        df = pd.read_parquet(file_path)
        # 重复值分析
        print(f"part-0000{i}.parquet\n[重复值统计]", df.duplicated(subset='user_name', keep='first').sum())
        df.drop_duplicates(subset='user_name', keep='first',inplace=True)
        print('[删除重复值后数据量统计]', len(df)) 
    #df_10G = pd.concat(dfs, axis=0, sort=False, join='outer')
    #print('10G数据整合完毕，总行数:', len(df_10G))

if flag30 == True:
    base_path = r"D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data_new\part-%05d.parquet"
    dfs = []
    for i in range(16):
        file_path = base_path % i
        df = pd.read_parquet(file_path)
        # 重复值分析
        print(f"part-0000{i}.parquet\n[重复值统计]", df.duplicated(subset='user_name', keep='first').sum())
        df.drop_duplicates(subset='user_name', keep='first', inplace=True)
        print('[删除重复值后数据量统计]', len(df)) 
        #dfs.append(df)
    #df_30G = pd.concat(dfs, axis=0, sort=False, join='outer')
    #print('30G数据整合完毕，总行数:', len(df_30G)) 
