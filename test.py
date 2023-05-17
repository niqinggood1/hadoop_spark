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
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import *
import time

def spark_proc_db_data(db):
    print('db:',db)
    sparkConf = SparkConf()
    sparkConf.setAppName("testPi")
    
    # sparkConf.set("spark.driver.port", "10000")
    # sparkConf.set("spark.driver.host", "172.21.63.21")
    spark = SparkSession.builder.config( conf=sparkConf ).getOrCreate()
    sc    = spark.sparkContext

    action= spark.read.format('jdbc').option('url','jdbc:mysql://119.23.216.1:3306').option('user','root').option('password','Hello2020').option("driver", "com.mysql.jdbc.Driver").option('dbtable', '%s.action'%db).load()
    cust  = spark.read.format('jdbc').option('url','jdbc:mysql://119.23.216.1:3306').option('user','root').option('password','Hello2020').option("driver", "com.mysql.jdbc.Driver").option('dbtable', '%s.customer'%db).load()
    order = spark.read.format('jdbc').option('url','jdbc:mysql://119.23.216.1:3306/%s?zeroDateTimeBehavior=convertToNull'%db).option('user', 'root').option('password', 'Hello2020').option("driver", "com.mysql.jdbc.Driver").option('dbtable', '%s.`order`' % db).load()
    #/yourMySqlDatabase?zeroDateTimeBehavior=convertToNull
    product= spark.read.format('jdbc').option('url','jdbc:mysql://119.23.216.1:3306').option('user','root').option('password','Hello2020').option("driver", "com.mysql.jdbc.Driver").option('dbtable', '%s.product'%db).load()
    record = spark.read.format('jdbc').option('url','jdbc:mysql://119.23.216.1:3306').option('user','root').option('password','Hello2020').option("driver", "com.mysql.jdbc.Driver").option('dbtable', '%s.record'%db).load()
    store  = spark.read.format('jdbc').option('url','jdbc:mysql://119.23.216.1:3306').option('user','root').option('password','Hello2020').option("driver", "com.mysql.jdbc.Driver").option('dbtable', '%s.store'%db).load()
    tag    = spark.read.format('jdbc').option('url','jdbc:mysql://119.23.216.1:3306').option('user','root').option('password','Hello2020').option("driver", "com.mysql.jdbc.Driver").option('dbtable', '%s.tag'%db).load()
    teacher= spark.read.format('jdbc').option('url','jdbc:mysql://119.23.216.1:3306').option('user','root').option('password','Hello2020').option("driver", "com.mysql.jdbc.Driver").option('dbtable', '%s.teacher'%db).load()
    times=20
    if times>1:
          tmp1 = cust
          tmp2 = order
          for i in range(times):
                tmp1=tmp1.union(cust)
                tmp2=tmp2.union(order)
          cust = tmp1
          order = tmp2

    for i in cust.schema.names:     cust = cust.withColumnRenamed( i, 'customer__'+i) # if 'id' in i:  cust=cust.withColumn(i,cust[i].cast(StringType()) )
    for i in order.schema.names:    order= order.withColumnRenamed( i, 'order__' + i)
    for i in product.schema.names:  product = product.withColumnRenamed(i, 'product__' + i)
    for i in record.schema.names:   record = record.withColumnRenamed(i, 'record__' + i)
    for i in store.schema.names:    store = store.withColumnRenamed(i, 'store__' + i)
    for i in teacher.schema.names:  teacher = teacher.withColumnRenamed(i, 'teacher__' + i)

    action       = action.withColumn('onehot', lit(1) )
    action_pivot = action.select(['customer_id', 'name', 'onehot','create_time']).groupBy('customer_id', 'create_time').pivot('name').sum('onehot')
    for i in action_pivot.schema.names: action_pivot = action_pivot.withColumnRenamed( i, 'action__' + i )

    tag_pivot = tag.select(['customer_id', 'name', 'weight','create_time']).groupBy('customer_id', 'create_time').pivot('name').sum('weight')
    for i in tag_pivot.schema.names:      tag_pivot = tag_pivot.withColumnRenamed(i, 'tag__' + i)

    # cust  = cust.na.fill(value='None').distinct().dropna( how='all'  )
    # order = order.na.fill(value='None').distinct().dropna( how='all' )

    #cust  = cust.repartition( 2,'customer__id' )
    merge = cust.join( order,  cust.customer__customer_id == order.order__customer_id, how='left'); #merge = cust.join(order,[cust.customer__customer_id==order.order__customer_id]  , how='inner')
    merge = merge.join(product,[ merge.order__product_id == product.product__product_id ], how='left')
    merge = merge.join(store, [ merge.order__store_id == store.store__store_id ], how='left')
    merge = merge.join(teacher, [merge.order__seller_id== teacher.teacher__teacher_id ], how='left')
    merge = merge.join(record, [ merge.order__order_id == record.record__order_id, \
                                 merge.order__seller_id==record.record__teacher_id, \
                                 merge.order__store_id==record.record__store_id   ], how='left')

    merge = merge.join(tag_pivot, [ merge.customer__customer_id == tag_pivot.tag__customer_id], how='left')
    merge = merge.join(action_pivot, [ merge.customer__seller_id == action_pivot.action__customer_id], how='left')
    #merge = merge.toPandas()#.collect()
    #merge.to_csv('%s_big_table.csv'%db, index=False, encoding='utf_8_sig',sep='|')
    print(merge.head(10))
    spark.stop()
    return


def spark_process():
    start = time.time()
    spark_proc_db_data(db='pdvm_prd_bc')
    end = time.time()
    print('times:', (end - start))

    return



if __name__ == '__main__':

    spark_process()
    exit()
