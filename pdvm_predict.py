#!/usr/bin/python
# coding=utf-8
import json
import random
import sys
import pandas as pd


def predict_real(r_json):
    data =r_json.get('data')
    model_id = r_json.get('model_id')
    model_name = r_json.get('prefix')
    print(model_id, model_name)
    print(type(data))
    cols = data.pop(0)
    df = pd.DataFrame(data, columns=cols)
    print(df.shape)
    print(df.columns)
    n = df.shape[0]
    pred = [random.random() for _ in range(n)]
    df['pred'] = pred
    return df

def abtp_predict_real(r_json):
    # dtype = r_json['usage']
    model_id = r_json['model_id']
    sample_id = r_json['sample_id']
    n = 5
    sample_ids = [sample_id for _ in range(n)]
    qus = [i+1 for i in range(n)]
    # sample_ids = np.array(sample_ids).reshape(3,1)
    # labels = np.zeros([n, 1], dtype=int)
    labels = [0 for _ in range(n)]
    df = pd.DataFrame({'sample_id': sample_ids, 'question_id':qus, 'label':labels})
    return df

if __name__ == '__main__':
    model_id = sys.argv[1]
    cid = sys.argv[2]
    model_name = sys.argv[3]
    print(predict_real(model_id, cid, model_name))

