#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pdvm_spark 
@File    ：calc_stk_factors.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2022/12/27 18:39 
'''

def process_weibi( pamarm ):
    import pandas as pd
    #filepath = './T0data/%s.csv'%stk
    dir,filename = pamarm.split('_')
    stk          = filename[:6]
    date         = dir[-8:]
    filepath     = dir+'/' + filename

    df = pd.read_csv( filepath  )
    df.columns =  [ i.replace('SZSE.%s.'%stk,'') for i  in df.columns ]

    # 委比=(委买手数-委卖手数)÷(委买手数+委卖手数)×100%
    df['a_all'] = df['ask_volume1'] + df['ask_volume2'] + df['ask_volume3'] + df['ask_volume4'] +df['ask_volume5']  #ask_volume1
    df['b_all'] = df['bid_volume1'] + df['bid_volume2'] + df['bid_volume3'] + df['bid_volume4'] +df['bid_volume5']
    df['weibi'] = ( df['b_all']-df['a_all'] )/( df['b_all']+df['a_all'] )
    weibi_std   = df['weibi'].std()
    return  date+'_'+stk+'_'+ str(weibi_std)


def calc_factors_with_spark(  params):
    print('in calc_factors  ')
    import os
    from pyspark.sql import SparkSession
    # from pyspark.sql import SQLContext
    from pyspark import SparkContext, SparkConf
    from pyspark.sql.types import StringType
    from pyspark.sql.functions import lit

    sparkConf = SparkConf()
    sparkConf.setAppName("sheet8_recronstruct")
    spark    = SparkSession.builder.config(conf=sparkConf).getOrCreate()
    sc       = spark.sparkContext

    data_dir      = 'E:/20220425'
    stks_file     = os.listdir(data_dir)
    data_stk_file = [ data_dir +'_'+i   for i in  stks_file ][:30]
    print('stks file:', stks_file)
    print('data_stk_file:',data_stk_file)

    cnt= 6
    record_rdd = sc.parallelize(data_stk_file, cnt)
    weibi_stds = record_rdd.map(lambda x:  process_weibi(x) ).collect()
    print( 'weibi_stds:',weibi_stds )

    import  pandas as pd
    weibi_std_df           = pd.DataFrame( weibi_stds, columns=['src'] )
    weibi_std_df['src_l']  = weibi_std_df['src'].apply(lambda x: x.split('_') )
    weibi_std_df['date']   = weibi_std_df['src_l'].apply(lambda x : x[0] )
    weibi_std_df['stks']   = weibi_std_df['src_l'].apply(lambda x : x[1] )
    weibi_std_df['weibi']  = weibi_std_df['src_l'].apply(lambda x : x[2] )
    del weibi_std_df['src'],weibi_std_df['src_l']
    print( weibi_std_df )
    return weibi_std_df


import params.params as params
if __name__ == '__main__':
    calc_factors_with_spark(  params=params)
    exit()

  
  