from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession
from config import *

def get_file_path(model_id, cname):
    model_path = CSTT_MODEL_PATH + cname + model_id + '_lr_model'
    pipe_path = CSTT_MODEL_PATH + cname + model_id + '_pipe_modle'
    print(model_path)
    return model_path, pipe_path


def cstt_model_infer(df, model_id, cname):
    df = spark.read.csv('tmp_cs.csv', header=True, inferSchema=True)
    df = df.na.drop()
    no_list = ['customer-customer_id', 'customer-id', 'order-amount', 'customer-encrypt_phone', 'customer-7more_sid']
    print('Predict begin!')
    model_path, pipe_path = get_file_path(model_id, cname)
    # 去掉order id类特征、product相关特征
    drop_list = ['customer-birthday', 'customer-name', 'order-pay_time', 'order-order_id', 'product-price',
                 'product-name', 'product-publish_time', 'order-user_name', 'order-order_time', 'customer-age',
                 'customer-create_time', 'customer-customer_id', 'customer-id', 'order-amount',
                 'customer-encrypt_phone', 'customer-7more_sid', 'label']
    df = df.drop(','.join(drop_list))
    df_split = df.randomSplit([0.9, 0.1], seed=26)
    df_train, df_test = df_split[0], df_split[1]
    evaluator = BinaryClassificationEvaluator()
    pipe_model = PipelineModel.load(pipe_path)
    model = LogisticRegressionModel.load(model_path)
    df_test = pipe_model.transform(df_test)
    prediction = model.transform(df_test)
    print(prediction.select('prediction', 'probability').head(5))
    # print(evaluator.evaluate(prediction))
    return prediction.select('customer-id', 'prediction').toJSON()





if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("LogisticRegressionSummary") \
        .config("spark.executor.extraJavaOptions=-XX:ReservedCodeCacheSize", "600m") \
        .getOrCreate()
    res = cstt_model_infer(None, '1', 'cs')
    print(res.first())
    spark.stop()