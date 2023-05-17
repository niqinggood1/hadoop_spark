"""
An example demonstrating Logistic Regression Summary.
Run with:
  bin/spark-submit examples/src/main/python/ml/logistic_regression_summary_example.py
"""
import tempfile
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplitModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, MapType, DoubleType
from pyspark.sql.functions import col
from pyspark.ml.feature import MinMaxScaler, VarianceThresholdSelector, VectorAssembler, StringIndexer, OneHotEncoder
from config import *
from sheet8_reconstruct import spark_proc_db_data
import params.params as params


def get_file_path(model_id, cname):
    model_path = CSTT_MODEL_PATH + cname + model_id + '_lr_model'
    pipe_path = CSTT_MODEL_PATH + cname + model_id + '_pipe_modle'
    print(model_path)
    return model_path, pipe_path


def cstt_model_main(df, model_id, cname):
    # Load training data
    # df = spark.read.csv('tmp_cs.csv', header=True, inferSchema=True)
    df = df.na.drop()
    no_list = ['customer__customer_id', 'customer__id', 'order__amount', 'customer__encrypt_phone', 'customer__7more_sid']
    # date_today = datetime.datetime.now()
    # today = date_today.strftime("%Y-%m-%d")
    print('RT models begin!')
    model_path, pipe_path = get_file_path(model_id, cname)
    # 去掉order id类特征、product相关特征
    drop_list = ['customer__birthday', 'customer__name', 'order__pay_time', 'order__order_id', 'product__price',
                 'product__name', 'product__publish_time', 'order__user_name', 'order__order_time', 'customer__age',
                 'customer__create_time', 'customer__customer_id', 'customer__id', 'order__amount',
                 'customer__encrypt_phone', 'customer__7more_sid']
    df = df.drop(','.join(drop_list))
    cate_cols = []
    num_cols = []
    stages = []
    for c, d in df.dtypes:
        if c == 'label':
            continue
        elif d == 'string':
            cate_cols.append(c)
            df = df.na.fill('unkown', subset=c)
            indexer = StringIndexer(inputCol=c, outputCol=c + "Index")
            onehotencoder = OneHotEncoder(inputCol=indexer.getOutputCol(), handleInvalid='keep', dropLast=True,
                                          outputCol=c + "classVec")
            stages.append(indexer)
            stages.append(onehotencoder)
        elif d in ['int', 'bigint', 'double']:
            num_cols.append(c)
            df = df.na.fill(0, subset=c)
    cols = [c + "classVec" for c in cate_cols] + num_cols
    vecAssembler = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid='skip')
    mmScaler = MinMaxScaler(inputCol=vecAssembler.getOutputCol(), outputCol="scaled")
    stages.append(vecAssembler)
    stages.append(mmScaler)
    pipeline = Pipeline(stages=stages)
    pipe_model = pipeline.fit(df)
    pipe_model.write().overwrite().save(pipe_path)
    df = pipe_model.transform(df)
    # split data into train and validation
    df_split = df.randomSplit([0.8, 0.2], seed=26)
    df_train, df_val = df_split[0], df_split[1]
    # 模型训练和验证
    ### 方式一 LR模型
    # lr = LogisticRegression(featuresCol='scaled', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    # model = lr.fit(df_train)
    # coeff = model.coefficients.toArray()
    #
    # # inteecept = model.intercept
    # trainingSummary = model.summary
    # objectiveHistory = trainingSummary.objectiveHistory
    # print("objectiveHistory:")
    # for objective in objectiveHistory:
    #     print(objective)
    # # Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    # trainingSummary.roc.show()
    # print("train_auc: " + str(trainingSummary.areaUnderROC))
    # predictions = model.transform(df_val)
    # evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    # print('validation_auc：', evaluator.evaluate(predictions))

    ### 方式二 使用TrainValidationSplit选择模型和调参
    lr = LogisticRegression(featuresCol='scaled')
    evaluator = BinaryClassificationEvaluator()
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .addGrid(lr.fitIntercept, [False, True]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .addGrid(lr.maxIter, [5, 10, 15]) \
        .build()
    tvs = TrainValidationSplit(estimator=lr,
                               estimatorParamMaps=paramGrid,
                               evaluator=evaluator,
                               trainRatio=0.8,
                               parallelism=1,
                               seed=42
                               )
    tvsModel = tvs.fit(df_train)
    predictions = tvsModel.transform(df_val)
    print('validation_auc：', evaluator.evaluate(predictions))
    model = tvsModel.bestModel
    model.write().overwrite().save(model_path)
    print('train_auc: ', model.summary.areaUnderROC)
    # best model
    coeff = model.coefficients.toArray()
    coeff = [float(x) for x in coeff]
    fea_index_dict = df.schema.fields[-2].metadata['ml_attr']['attrs']
    fea_index = fea_index_dict['numeric'] + fea_index_dict['binary']
    index_fea = {}
    for i in fea_index:
        index_fea.setdefault(i['idx'], (i['name'], coeff[i['idx']]))
    df_weight = spark.createDataFrame(index_fea.values(), ['feature', 'weight'], verifySchema=False)
    prop = {'user': 'root',
            'password': 'Hello2020',
            'driver': 'com.mysql.cj.jdbc.Driver'
            }
    url = 'jdbc:mysql://8.129.223.139:3306/cel_stock'
    df_weight.write.jdbc(url=url, table='break_back_to_trample_old', mode='append', properties=prop)
    print('Done')


if __name__ == "__main__":
    # spark = SparkSession \
    #     .builder \
    #     .appName("LogisticRegressionSummary") \
    #     .config("spark.executor.extraJavaOptions=-XX:ReservedCodeCacheSize", "600m") \
    #     .getOrCreate()
    spark, sc, df = spark_proc_db_data(db='pdvm_prd_demo', params=params)
    cstt_model_main(df, '3', 'demo')
    spark.stop()