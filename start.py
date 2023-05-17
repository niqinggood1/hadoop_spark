#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, escape, url_for, request, Response, jsonify
from run_classifier_predict import text_predict
from pdvm_predict import abtp_predict_real, predict_real
from flasgger import Swagger, swag_from
from kol_process import load_and_predict

from correlation_old import corr_main
from cluster_old     import clu_main

app = Flask(__name__)
Swagger(app)
executor = ThreadPoolExecutor(1)

@app.route('/')
def hello_world():
    return 'Hello, World!'


###########################   Run batch

from sheet8_reconstruct import spark_proc_db_data
@app.route('/correlation', methods=['POST'])

if_use_spark=0
def correlation():

    ##根据参数，处理数据
    r = request.get_data()
    r_json = json.loads(r)
    fea1 = r_json['fea1']
    fea2 = r_json['fea2']
    if if_use_spark:
        cname           = r_json['cname']
        url             = r_json['url']
        report_id       = r_json['report_id']
        cycle_id        = r_json['cycle_id']
        ventity_name    = r_json['ventity_name']
        independent     = r_json['independent']
        executor.submit( corr_main, fea1, fea2, cname, url, report_id, cycle_id, ventity_name, independent )
        #通过参数 找聚类生成好的数据文件，基于数据文件触发聚类
    else:
        import params.params as params
        df = spark_proc_db_data(db='pdvm_prd_demo', params=params)

        from correlation import main_correlation
        main_correlation(  df, fea1, fea2  )
    return "Correlation Task started."

@app.route('/cluster', methods=['POST'])
def cluster():
    r           = request.get_data()
    r_json      = json.loads(r)
    cname       = r_json['cname']
    url         = r_json['url']
    report_id   = r_json['report_id']
    cycle_id    = r_json['cycle_id']
    # 根据参数，处理数据
    if if_use_spark==False:
        executor.submit( clu_main, cname, url, report_id, cycle_id  )
    else:
        from cluster import culuster_main
        import params.params as params
        df = spark_proc_db_data(db='pdvm_prd_demo', params=params )
        culuster_main( df )

    return "Cluster Task started."

###########################    realtime on line
@app.route('/text_predict', methods=['POST'])
def text_predict_():
    """
        This is the language model predict API
        Call this api passing a text and get back its label
        ---
        tags:
          - Language API
        parameters:
          - name: model_id
            type: string
            # required: true
            description: The model id
          - name: question_id
            type: integer
            description: The question id
          - name: text
            type: string
            description: The text to be predict
        responses:
          500:
            description: Error
          200:
            description: Success
            schema:
              id: algo
              properties:
                label:
                  type: integer
        """
    # df = pd.read_csv(request.files['file'])
    r = request.get_data()
    r_json = json.loads(r)
    print(r_json, type(r_json))
    # qus_id = r_json['question_id']
    # text = r_json['text']
    res = text_predict(r_json)
    print(res)
    response = {'label': res}
    response_pickled = json.dumps(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/abtp_predict', methods=['POST'])
def abtp_predict():
    """
        This is the speech model predict API
        Call this api passing a sample id and get back its label
        ---
        tags:
          - Speech API
        parameters:
          - name: model_id
            type: string
            # required: true
            description: The model id
          - name: sample_id
            type: integer
            description: The sample id
        responses:
          500:
            description: Error
          200:
            description: Success
            schema:
              id: speech
              properties:
                sample_id:
                  type: integer
                question_id:
                  type: integer
                label:
                  type: integer

            """
    # df = pd.read_csv(request.files['file'])
    r = request.get_data()
    r_json = json.loads(r)
    # cname = r_json['cname']
    res = abtp_predict_real(r_json)
    # res.to_csv(filename)
    # files = {'file': open(filename, 'rb')}
    # jdata = res.to_json(orient='records', force_ascii=False)
    # return jsonify(json.loads(jdata))
    return Response(res.to_json(orient="records"), mimetype='application/json')

@app.route('/predict', methods=['POST'])
def cstt_predict():  #lr_infer,训练处理数据用spark
    r = request.get_data()
    print(r, type(r))
    r_json = json.loads(r)
    print(r_json,type(r_json))
    # r = request.get_data()
    # r_json = json.loads(r)
    # model_id = r_json['model_id']
    # cid = r_json['cid']
    # cname = r_json['cname']
    res = predict_real(r_json)
    return Response(res.to_json(orient="records"), mimetype='application/json')

@app.route('/kol_predict', methods=['POST'])
def kol_predict():
    r = request.get_data()
    r_json = json.loads(r)
    # cname = r_json['cname']
    df = pd.DataFrame([r_json])
    df.rename(columns={'like': 'like_count', 'favorite': 'collect_count', 'comment': 'comment_count',
                       'share': 'share_count', 'total': 'total_count'}, inplace=True)
    df['read_count'] = 0
    view, imp = load_and_predict('231', df, 'kol')
    # res.to_csv(filename)
    # files = {'file': open(filename, 'rb')}
    # jdata = res.to_json(orient='records', force_ascii=False)
    # return jsonify(json.loads(jdata))
    # return Response(res.to_json(orient="records"), mimetype='application/json')
    response = {'view': int(view), 'impression': imp}
    response_pickled = json.dumps(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/kol_predict_pdvm', methods=['POST'])
def kol_predict_pdvm():
    r = request.get_data()
    r_json = json.loads(r)
    print(r_json)
    model_id = str(r_json['model_id'])
    data = r_json['data']
    method = r_json['method']
    is_use = r_json['is_use']
    df = pd.DataFrame(data)
    # try:
    res = load_and_predict(model_id, df, method, is_use)
    return Response(res.to_json(orient="records"), mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, processes=True, debug=True)
    # app.run(processes=True)
