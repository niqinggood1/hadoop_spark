# coding=UTF-8
#from matplotlib.font_manager import _rebuild
#_rebuild()

import sys
import os
import json
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth',100)

# 接收参数（阈值和公司名称）
# if len(sys.argv) > 1:
#     cname = sys.argv[1]
# else:
#     print('Please input company name1')

def clu(cname):
    th = 0.98*(1-0.98)
    #读入数据和标准化
    lr_dir = os.getcwd()
    file_path = '/var/www/dev/pdvm-php/cluster_data/' + cname +'.csv'
    # file_path = '/Users/huangnanxi/PycharmProjects/pdvm-php/cluster_data/' + cname + '.csv'
    data = pd.read_csv(file_path)
    data.drop(columns=['customer-customer_id'], inplace=True)
    # data = pd.read_csv('./pdvm.csv')
    columns = data.columns
    ss = MinMaxScaler()
    data_s = ss.fit_transform(data)
    # new_df = pd.DataFrame(data_s, columns=columns)
    # new_df.to_csv('out.csv')


    try:
        sel = VarianceThreshold(threshold=th)
        new_data = sel.fit_transform(data_s)
        sup = sel.get_support()
        cols = []
        for i in range(len(sup)):
            if sup[i]:
                cols.append(columns[i])
        df = pd.DataFrame(new_data, columns=cols)
        # df.to_csv('new.csv', index=None)

        #自动K值
        score_list = list()
        silhouette_int = -1
        for n_clusters in range(3, 6):
            model_kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels_tmp = model_kmeans.fit_predict(df)
            silhouette_tmp = metrics.silhouette_score(df, cluster_labels_tmp)
            if silhouette_tmp > silhouette_int:
                best_k = n_clusters
                silhouette_int = silhouette_tmp
                best_kmeans = model_kmeans
                cluster_labels_k = cluster_labels_tmp
            score_list.append([n_clusters, silhouette_tmp])
        ans = str(cluster_labels_k) +'+'+str(cols)
        return ans
    except:
        return None

    #限制特征个数
    # selectModel = SelectKBest(chi2, k=8)
    # selectModel.fit_transform(X,cluster_labels_k)
    # X_select = selectModel.get_support(True)
    # data = data.iloc[:,X_select]



    #计算每个类别均值
    # cluster_labels = pd.DataFrame(cluster_labels_k, columns=['clusters'])
    # merge_data = pd.concat((df, cluster_labels), axis=1)
    # cluster_features = []
    # for line in range(best_k):
    #     label_data = merge_data[merge_data['clusters'] == line]
    #     part_desc = label_data.describe().round(3)
    #     merge_line = part_desc.iloc[1,:]
    #     cluster_features.append(merge_line)
    # cluster_pd = pd.DataFrame(cluster_features).T

    #画图
    # num_sets = cluster_pd.T.astype(np.float64)
    # num_sets_max_min = ss.fit_transform(num_sets)
    # fig = plt.figure(figsize=(7,7))
    # ax = fig.add_subplot(111, polar=True)
    # labels = np.array(merge_line.index[:-1])
    # color_list = ['r','g','b','c','y','m','k','lightgreen','purple']
    # angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    # for i in range(len(num_sets)):
    #     data_tmp = num_sets_max_min[i, :-1]
    #     data = data_tmp
    #     ax.plot(angles, data, 'o-', c=color_list[i], label=i,linewidth=2.5)
    #     ax.set_thetagrids(angles*180/np.pi, labels, fontproperties='SimHei',fontsize=14)
    #     ax.set_title(u'聚类分析',fontproperties='SimHei',fontsize=18)
    #     ax.set_rlim(-0.2, 1.2)
    #     plt.legend()
    # plt.show()


def clu_main(cname, url, report_id, cycle_id):
    data = clu(cname)
    json_d = json.dumps({'data': data, 'report_id':report_id, 'cycle_id':cycle_id})
    addr = url
    content_type = 'application/json'
    headers = {'content-type': content_type}
    response = requests.post(addr, data=json_d, headers=headers)


if __name__ == '__main__':
    cname = sys.argv[1]
    print(clu(cname))
