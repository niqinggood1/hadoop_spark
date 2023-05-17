#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pdvm_spark 
@File    ：calc_high_frequence_factors.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2023/1/4 15:32 
'''
import pandas as pd
def process_weibi( pamarm ):
    #filepath = './T0data/%s.csv'%stk
    dir,filename = pamarm.split('_')
    stk          = filename[:6]
    date         = dir[-8:]
    filepath     = dir+'/' + filename

    df          = pd.read_csv( filepath  )
    df.columns  =  [ i.replace('SSE.','').replace('SZSE.','').replace('%s.'%stk,'') for i  in df.columns ]
    df['time']  =  df[  'datetime' ].apply(lambda x: x[11:19])
    df          =  df[  (df['time']>='09:25:00') & (df['time']<='14:59:59')  ]
    # print( df.head(4) )
    # 委比=(委买手数-委卖手数)÷(委买手数+委卖手数)×100%
    df['a_all'] = df['ask_volume1'] + df['ask_volume2'] + df['ask_volume3'] + df['ask_volume4'] +df['ask_volume5']  #ask_volume1
    df['b_all'] = df['bid_volume1'] + df['bid_volume2'] + df['bid_volume3'] + df['bid_volume4'] +df['bid_volume5']
    df['weibi'] = ( df['b_all']- df['a_all'])*100/( df['b_all']+df['a_all'] )

    weibi_median = df['weibi'].median()
    weibi_mean   = df['weibi'].mean( )
    weibi_max    = df['weibi'].max( )
    weibi_min    = df['weibi'].min( )
    weibi_std    = df['weibi'].std(  )

    ret_list     = [ weibi_median,  weibi_mean,  weibi_max, weibi_min, weibi_std   ]
    ret_list     = map(  lambda x: str(x) ,ret_list )
    ret_str      = '_'.join(  ret_list )
    # print('ret_str:', ret_str )
    return  date+'_'+stk+ '_'  + ret_str

import os
def get_oneday_factors( dir='E:/4月/' ,date='' ):
    data_dir =  dir + date
    stks_file = os.listdir(data_dir);  print( 'stks_file:',stks_file )
    data_stk_file = [data_dir + '_' + i for i in stks_file]#[:20]
    print('data_stk_file:',data_stk_file )
    ret_list = [ ]
    for k in data_stk_file:
        print( k )
        ret = process_weibi( k )
        ret_list.append(  ret )

    weibi_std_df = pd.DataFrame( ret_list, columns=['src'])
    weibi_std_df['src_l'] = weibi_std_df['src'].apply(lambda x: x.split('_'))
    print(weibi_std_df)
    weibi_std_df['date'] = weibi_std_df['src_l'].apply(lambda x: x[0])
    weibi_std_df['stks'] = weibi_std_df['src_l'].apply(lambda x: x[1])

    weibi_std_df['weibi_median']= weibi_std_df['src_l'].apply(lambda x: x[2])
    weibi_std_df['weibi_mean']  = weibi_std_df['src_l'].apply(lambda x: x[3])
    weibi_std_df['weibi_max']   = weibi_std_df['src_l'].apply(lambda x: x[4])
    weibi_std_df['weibi_min']   = weibi_std_df['src_l'].apply(lambda x: x[5])
    weibi_std_df['weibi_std']   = weibi_std_df['src_l'].apply(lambda x: x[6])
    del weibi_std_df['src'], weibi_std_df['src_l']
    return  weibi_std_df

if __name__ == '__main__':
    df = pd.DataFrame()
    dir='E:/6月/'
    date_lists = os.listdir(dir)
    print( 'dir:',date_lists )
    for d in date_lists:
        tmp = get_oneday_factors(  dir=dir ,date=d  )
        tmp.to_excel('E:/hf_reslut_data/hf_%s.xlsx'%d  )
        # df = df.append(tmp)
        # df.to_csv('E:/hf_reslut_data/hf_all.xlsx')
    exit()
    
  
  