#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pvdm-data 
@File    ：sheet8_reconstruct.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2022/11/11 10:16 
'''
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

import importlib
importlib.reload(sys)
import copy
import params.params as params
def proc_db_data(db='pdvm_prd_prepare',params=None):
    import sqlalchemy as sa
    import pandas as pd
    print('in proc_db_data db:',db)
    #mysql_sa = sa.create_engine("mysql+pymysql://root:Hello2020@119.23.216.1:3306/%s?charset=utf8mb4"%(,,db), max_overflow=5, encoding='utf8', pool_recycle=3600)
    mysql_sa = sa.create_engine("mysql+pymysql://%s:%s@%s/%s?charset=utf8mb4" % (params.mysql_user,params.mysql_passwd, params.mysql_ip_port, db), max_overflow=5, encoding='utf8', pool_recycle=3600)
    action  = pd.read_sql('select * from action', mysql_sa)
    cust    = pd.read_sql('select * from customer', mysql_sa)
    order   = pd.read_sql('select * from `order`', mysql_sa)
    product = pd.read_sql('select * from product', mysql_sa)
    record  = pd.read_sql('select * from record', mysql_sa)
    store   = pd.read_sql('select * from store', mysql_sa)
    tag     = pd.read_sql('select * from tag', mysql_sa)
    teacher = pd.read_sql('select * from teacher', mysql_sa)

    for df in [order,cust,product,record,teacher,store,tag,action]:
        for idx in df.columns:
            if 'id' in idx:
                df[idx] = df[idx].astype(str)
    order.columns   = ['order-' + i for i in order.columns]
    cust.columns    = ['customer-' + i for i in cust.columns]
    product.columns = ['product-' + i for i in product.columns]
    teacher.columns = ['teacher-' + i for i in teacher.columns]
    store.columns   = ['store-' + i for i in store.columns]
    record.columns  = ['record-' + i for i in record.columns]

    merge = pd.merge( cust,order, left_on='customer-customer_id', right_on='order-customer_id', how='left')
    merge = pd.merge(merge, product, left_on='order-product_id', right_on='product-product_id', how='left')
    merge = pd.merge(merge, store, left_on='order-store_id', right_on='store-store_id', how='left')
    merge = pd.merge(merge, teacher, left_on='order-seller_id', right_on='teacher-teacher_id', how='left')
    merge = pd.merge(merge, record, left_on=['order-order_id', 'order-seller_id', 'order-store_id'],
                      right_on=['record-order_id', 'record-teacher_id', 'record-store_id'], how='left')

    merge = pd.merge(merge, store, left_on=['order-store_id'], right_on=['store-store_id'], how='left')
    tag_pivot = tag[['customer_id', 'name', 'weight']].pivot_table(index='customer_id', columns='name', values='weight').reset_index()
    tag_pivot.columns = ['tag-' + i for i in tag_pivot.columns]

    action['onehot'] = 1
    action_pivot = action[['customer_id', 'name', 'onehot']].pivot_table(index='customer_id', columns='name', values='onehot').reset_index()
    action_pivot.columns = ['action-' + i for i in action_pivot.columns]

    merge = pd.merge(merge, tag_pivot, left_on=['customer-customer_id'], right_on=['tag-customer_id'], how='left')
    merge = pd.merge(merge, action_pivot, left_on=['customer-customer_id'], right_on=['action-customer_id'], how='left')
    #merge.to_csv('%s_big_table.csv'%db, index=False, encoding='utf_8_sig')
    #merge.to_sql('%s_big_table.csv' % db, mysql,if_exists='w', index=False, encoding='utf_8_sig',sep='|')
    return merge

def spark_proc_db_data( db,params ):
    print('in spark_proc_db_data db:',db)
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StringType
    from pyspark.sql.functions import lit

    sparkConf = SparkConf()
    sparkConf.setAppName("sheet8_recronstruct")
    if params.deployMode=="client":
        sparkConf.setMaster(params.k8s)#("k8s://https://172.21.63.22:6443")
        sparkConf.set("spark.submit.deployMode", "client")  # client
        sparkConf.set("spark.kubernetes.authenticate.driver.serviceAccountName", "spark")
        sparkConf.set("spark.executor.instances", 2)
        sparkConf.set("spark.kubernetes.container.image", "niqinggood/spark300py:1")
        sparkConf.set("spark.kubernetes.pyspark.pythonVersion", 2)
        sparkConf.set("spark.driver.memory", "4g")
        sparkConf.set("spark.executor.memory", "4g")
        sparkConf.set("spark.driver.host", "172.21.63.22")
        sparkConf.set("spark.sql.autoBroadcastJoinThresholds", "-1")
        sparkConf.set("spark.sql.debug.maxToStringFields", "500")
        # sparkConf.set("spark.driver.port", "10000")
        # sparkConf.set("spark.driver.host", "172.21.63.21")


    spark = SparkSession.builder.config( conf=sparkConf ).getOrCreate()
    sc    = spark.sparkContext
    
    reader = spark.read.format('jdbc').option('url','jdbc:mysql://%s/%s?zeroDateTimeBehavior=convertToNull'%(params.mysql_ip_port,db))\
        .option('user',params.mysql_user).\
        option('password',params.mysql_passwd).\
        option("driver", "com.mysql.jdbc.Driver")
    action = reader.option('dbtable', '%s.action'%db).load()
    cust   = reader.option('dbtable', '%s.customer'%db).load()
    order  = reader.option('dbtable', '%s.`order`'%db).load()
    cust   = reader.option('dbtable', '%s.customer'%db).load()
    order  = reader.option('dbtable', '%s.`order`' % db).load()
    product= reader.option('dbtable', '%s.product'%db).load()
    record = reader.option('dbtable', '%s.record'%db).load()
    store  = reader.option('dbtable', '%s.store'%db).load()
    tag    = reader.option('dbtable', '%s.tag'%db).load()
    teacher= reader.option('dbtable', '%s.teacher'%db).load()

    for i in cust.schema.names:     cust = cust.withColumnRenamed( i, 'customer__'+i       ) # if 'id' in i:  cust=cust.withColumn(i,cust[i].cast(StringType()) )
    for i in order.schema.names:    order= order.withColumnRenamed( i, 'order__' + i       )
    for i in product.schema.names:  product = product.withColumnRenamed(i, 'product__' + i )
    for i in record.schema.names:   record = record.withColumnRenamed(i, 'record__' + i    )
    for i in store.schema.names:    store = store.withColumnRenamed(i, 'store__' + i       )
    for i in teacher.schema.names:  teacher = teacher.withColumnRenamed(i, 'teacher__' + i )

    action       = action.withColumn('onehot', lit(1) )
    action_pivot = action.select(['customer_id', 'name', 'onehot','create_time']).groupBy('customer_id', 'create_time').pivot('name').sum('onehot')
    for i in action_pivot.schema.names: action_pivot = action_pivot.withColumnRenamed( i, 'action__' + i )

    tag_pivot = tag.select(['customer_id', 'name', 'weight','create_time']).groupBy('customer_id', 'create_time').pivot('name').sum('weight')
    for i in tag_pivot.schema.names:      tag_pivot = tag_pivot.withColumnRenamed(i, 'tag__' + i)


    cust  = cust.repartition( 2,'customer__id' )
    merge = cust.join( order,  cust.customer__customer_id == order.order__customer_id, how='left'); #merge = cust.join(order,[cust.customer__customer_id==order.order__customer_id]  , how='inner')
    merge = merge.join(product,[ merge.order__product_id == product.product__product_id ], how='left')
    merge = merge.join(store, [ merge.order__store_id == store.store__store_id ], how='left')
    merge = merge.join(teacher, [merge.order__seller_id== teacher.teacher__teacher_id ], how='left')
    merge = merge.join(record, [ merge.order__order_id == record.record__order_id, \
                                 merge.order__seller_id==record.record__teacher_id, \
                                 merge.order__store_id==record.record__store_id   ], how='left' )
    merge = merge.join(tag_pivot, [ merge.customer__customer_id == tag_pivot.tag__customer_id], how='left')
    merge = merge.join(action_pivot, [ merge.customer__seller_id == action_pivot.action__customer_id], how='left')
    print( merge.show() )
    merge = merge.toPandas()
    # merge.to_csv('%s_big_table_spark.csv'%db, index=False, encoding='utf_8_sig',sep='|')
    #merge.write.parquet(path='%s_big_table.csv' % db, mode='overwrite')
    return merge


def spark_process(params):
    import time
    start = time.time()
    spark_proc_db_data(db='pdvm_prd_demo',params=params)
    spark_proc_db_data(db='pdvm_prd_prepare',params=params)
    spark_proc_db_data(db='pdvm_prd_bc',params=params)
    spark_proc_db_data(db='pdvm_prd_cs',params=params)
    spark_proc_db_data(db='pdvm_prd_shaue',params=params)
    spark_proc_db_data(db='pdvm_prd_shuheng',params=params)
    spark_proc_db_data(db='pdvm_prd_sucai',params=params)
    end = time.time()
    print('spark times:', (end - start))
    return


def trad_process(params):
    import time
    start = time.time()
    proc_db_data(db='pdvm_prd_demo',params=params)
    proc_db_data(db='pdvm_prd_prepare',params=params)
    proc_db_data(db='pdvm_prd_bc',params=params)
    proc_db_data(db='pdvm_prd_cs',params=params)
    proc_db_data(db='pdvm_prd_esc_xcx',params=params)
    proc_db_data(db='pdvm_prd_shaue',params=params)
    proc_db_data(db='pdvm_prd_shuheng',params=params)
    proc_db_data(db='pdvm_prd_sucai',params=params)
    end = time.time()
    print('trad times:', (end - start))
    return

import time
def stress_test(times=5):
    # start = time.time()
    # proc_db_data(db='pdvm_prd_bc', times=times)
    # end = time.time()
    # print('times1:', (end - start))
    start = time.time()
    spark_proc_db_data(db='pdvm_prd_bc',times=times)
    end = time.time()
    print('times2:', (end - start) )
    return


if __name__ == '__main__':
    spark_proc_db_data( db='pdvm_prd_demo', params=params)
    #trad_process(params)
    #spark_process(params)
    exit()

