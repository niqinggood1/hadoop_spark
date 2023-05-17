#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pdvm_spark 
@File    ：test_tf.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2023/1/6 13:53 
'''
from saddle import dcn
if __name__ == '__main__':
    import tensorflow as tf
    input = tf.cast([[1,2],[2,3],[3,4]], tf.float32)
    a = input[:,0]
    b = input[:,1]
    print('a:',a )
    ret = tf.multiply( a,b )
    print( 'ret:',ret )

    ret = tf.reduce_sum( input,axis=1 )
    print('ret2:', ret)
    exit()
    
  
  