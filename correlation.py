from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler, VarianceThresholdSelector, VectorAssembler, StringIndexer, OneHotEncoder, IndexToString

def main_correlation(df, feature1= 'customer-gender',feature2='order-PH1' ):
    spark = SparkSession \
        .builder \
        .appName("CorrelationExample") \
        .getOrCreate()
    # feature1, feature2 = 'customer-gender', 'order-PH1'
    # df = spark.read.csv('test_correlation_31.csv', header=True, inferSchema=True)
    df = df.select(feature1, feature2)
    cate_cols = []
    num_cols  = []
    for col, d in df.dtypes:
        if d == 'string':
            cate_cols.append(col)
            df = df.na.fill('unkown', subset=col)
        else:
            num_cols.append(col)
            df = df.na.fill(0, subset=col)
    show = df.select(feature2).collect()
    stages = []
    for c in cate_cols:
        indexer = StringIndexer(inputCol=c, outputCol=c + "Index")
        onehotencoder = OneHotEncoder(inputCol=indexer.getOutputCol(), handleInvalid='keep', dropLast=True, outputCol=c + "classVec")
        stages.append(indexer)
        stages.append(onehotencoder)
    cols = [c + "classVec" for c in cate_cols] + num_cols
    vecAssembler = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid='skip')
    mmScaler = MinMaxScaler(inputCol=vecAssembler.getOutputCol(), outputCol="scaled")
    stages.append(vecAssembler)
    stages.append(mmScaler)
    pipeline = Pipeline(stages=stages)
    df_tr = pipeline.fit(df).transform(df)
    fea_index_dict = df_tr.schema.fields[-2].metadata['ml_attr']['attrs']
    index_fea = {}
    for k, v in fea_index_dict.items():
        for i in v:
            index_fea.setdefault(i['idx'], i['name'])
    head = df_tr.head()
    # print(df_tr.collect())
    r1 = Correlation.corr(df_tr, "scaled").collect()[0][0].toArray()
    # print("Pearson correlation matrix:\n" + str(r1))
    # print("Pearson correlation matrix:\n" + str(r1[0].toArray().tolist()[0]))
    # r2 = Correlation.corr(df_tr, "scaled", "spearman").head()
    res = []
    for i in range(len(index_fea)):
        if feature1 not in index_fea[i]:
            break
        for j in range(len(index_fea) - 1, i, -1):
            if feature2 not in index_fea[j]:
                break
            res.append(index_fea[i] + '|' + index_fea[j] + ':' + str(r1[i][j]))
    print(res)
    spark.stop()


if __name__ == "__main__":


