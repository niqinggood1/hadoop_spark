#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import requests
from func import *
from sklearn.metrics.cluster import normalized_mutual_info_score


def is_nan(data):
    if np.isnan(data):
        return 0
    return data


def cal(df1, df2):
    m_score = is_nan(normalized_mutual_info_score(df1, df2))
    # p_score = pearsonr(nor_df1, nor_df2)
    p1 = df1.corr(df2, method="spearman")
    p2 = df1.corr(df2, method="pearson")
    k_score = df1.corr(df2, method="kendall")
    return m_score, p1, p2, k_score


def corr(fea1, fea2, cname):
    # col = ['customer+customer_id', 'action+action_id', 'action+name', 'action+type',
    #        'customer+gender', 'customer+age', 'customer+highest_education', 'customer+home_province', 'customer+home_city_level',
    #        'customer+hometown_city_level', 'customer+income', 'customer+member_level', 'customer+source1',
    #        'tag+name', 'tag+tag_id',
    #        'order+amount', 'order+pay_method', 'order+status', 'order+source', 'order+string112', 'order+string113', 'order+string114', 'order+string115']
    # col = ['customer+customer_id', 'action+action_id', 'action+name', 'action+type',
    #        'customer+gender', 'customer+age', 'customer+highest_education', 'customer+home_province',
    #        'customer+home_city_level',
    #        'customer+hometown_city_level', 'customer+income', 'customer+member_level',
    #        'tag+name', 'tag+tag_id',
    #        'order+order_id', 'order+amount', 'order+pay_method', 'order+status', 'order+source']

    num_col = ['customer-age', 'order-amount', 'product-price']
    # if fea1 not in col or fea2 not in col:
    #     return None

    # 预处理
    data = write_csv(cname)
    if data is None:
        return None
    col = data.columns
    # if fea1 not in col or fea2 not in col:
    #     return None
    con_col = list(set(col)-set(num_col))
    data[con_col] = data[con_col].astype(str)
    nor_df1, nor_df2 = normal_encode(fea1, fea2, data)
    one_df1, one_df2 = one_hot_encode(fea1, fea2, data, col)

    # with open("data/out.csv", mode='r') as data_file:
    #     df = pd.read_csv(data_file)
    # 计算相关系数
    # 类别编码
    if nor_df1 is None or nor_df2 is None:
        nor_str = fea1 + '|' + fea2 + ':0,0,0,0'
    else:
        m_score, p1, p2, k_score = cal(nor_df1, nor_df2)
        nor_str = fea1 + '|' + fea2 + ':' + str(m_score) + ',' + str(p1) + ',' + str(p2) + ',' + str(k_score)
    # m_score1 = is_nan(normalized_mutual_info_score(nor_df1, nor_df2))
    # #p_score1 = pearsonr(nor_df1, nor_df2)
    # p1 = nor_df1.corr(nor_df2, method="spearman")
    # p2 = nor_df1.corr(nor_df2, method="pearson")
    # #p1, p2 = is_nan(p_score1[0]), is_nan(p_score1[1])
    # k_score1 = nor_df1.corr(nor_df2, method="kendall")
    # nor_str = fea1+'|'+fea2+':'+str(m_score1)+','+str(p1)+','+str(p2)+','+str(k_score1)
    #fdict[fea1+'|'+fea2] = [m_score, p_score, k_score]
    # one-hot编码
    one_str = [nor_str]
    if one_df1 is None or one_df2 is None:
        return nor_str
    for n1 in one_df1.columns:
        for n2 in one_df2.columns:
            # m_score = is_nan(normalized_mutual_info_score(one_df1[n1], one_df2[n2]))
            # #p_score = pearsonr(one_df1[n1], one_df2[n2])
            # #p1, p2 = is_nan(p_score[0]), is_nan(p_score[1])
            # p1 = one_df1[n1].corr(one_df2[n2], method="pearson")
            # p2 = one_df1[n1].corr(one_df2[n2], method="spearman")
            # k_score = is_nan(one_df1[n1].corr(one_df2[n2], method="kendall"))
            m_score, p1, p2, k_score = cal(one_df1[n1], one_df2[n2])
            tmp_str = n1+'|'+n2+':'+str(m_score)+','+str(p1)+','+str(p2)+','+str(k_score)
            #print(tmp_str)
            #fdict[n1+'|'+n2] = [m_score, p_score, k_score]
            one_str.append(tmp_str)
    #print(one_str)
    #print('='.join(one_str))
    ans = '='.join(one_str)
    # print(ans)
    return ans
    # return '='.join(one_str)


def corr_main(fea1, fea2, cname, url, report_id, cycle_id, ventity_name, independent):
    data = corr(fea1, fea2, cname)
    json_d = json.dumps({'data': data, 'report_id':report_id, 'cycle_id':cycle_id, 'ventity_name':ventity_name, 'independent': independent})
    addr = url
    content_type = 'application/json'
    headers = {'content-type': content_type}
    response = requests.post(addr, data=json_d, headers=headers)
    # print(response.text)


if __name__ == '__main__':
    # 获取参数
    fea1 = sys.argv[1]
    fea2 = sys.argv[2]
    cname = sys.argv[3]
    # print(main('tag-type', 'order-amount'))
    print(corr(fea1, fea2, cname))
