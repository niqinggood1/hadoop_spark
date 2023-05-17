"""
An example demonstrating k-means clustering.
Run with:
  bin/spark-submit examples/src/main/python/ml/kmeans_example.py
This example requires NumPy (http://www.numpy.org/).
"""
# $example on$
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import MinMaxScaler, VarianceThresholdSelector, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.sql import SparkSession

def culuster_main(df,):
    spark = SparkSession \
        .builder \
        .appName("KMeansExample") \
        .config('spark.sql.debug.maxToStringFields', 2000) \
        .config(
        "spark.driver.extraJavaOptions",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED",
    ) \
        .getOrCreate()
    # Loads data.
    # df = spark.read.csv('tmp_cs.csv', header=True, inferSchema=True)
    cate_cols = []
    num_cols = []
    for col, d in df.dtypes:
        if d == 'string':
            cate_cols.append(col)
            df = df.na.fill('unkown', subset=col)
        else:
            num_cols.append(col)
            df = df.na.fill(0, subset=col)
    stages = []
    for c in cate_cols:
        indexer = StringIndexer(inputCol=c, outputCol=c + "Index")
        onehotencoder = OneHotEncoder(inputCol=indexer.getOutputCol(), handleInvalid='keep', outputCol=c + "classVec")
        stages.append(indexer)
        stages.append(onehotencoder)
    cols = [c + "classVec" for c in cate_cols] + num_cols
    vecAssembler = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid='skip')
    mmScaler = MinMaxScaler(inputCol=vecAssembler.getOutputCol(), outputCol="scaled")
    stages.append(vecAssembler)
    stages.append(mmScaler)
    pipeline = Pipeline(stages=stages)
    df_tr = pipeline.fit(df).transform(df).select('scaled')
    print(df_tr.collect())
    # th = 0.98 * (1 - 0.98)
    # select_col = []
    # for c in df.columns:
    #     # df = df.withColumn(c, df[c].cast('int'))
    #     df = df.withColumnRenamed(c, c.replace('.', ''))
    #     c = c.replace('.', '')
    #     vecAssembler = VectorAssembler(inputCols=[c], outputCol="features", handleInvalid='skip')
    #     try:
    #         dfc = df.select(c)
    #     except Exception as e:
    #         print(e)
    #         continue
    #     dfc = vecAssembler.transform(dfc)
    #     selector = VarianceThresholdSelector(featuresCol="features", varianceThreshold=th, outputCol="selectedFeatures")
    #     model = selector.fit(dfc)
    #     new_df = model.transform(dfc).select('selectedFeatures')
    #     if len(new_df.head(0)) == 0:
    #         # drop_col.append(c)
    #         print('del:', c)
    #         continue
    #     mmScaler = MinMaxScaler(inputCol="features", outputCol="scaled")
    #     model = mmScaler.fit(dfc)
    #     df[c] = model.transform(dfc)['scaled']
    #     select_col.append(c)
    #     print(df.columns)
    # df = pd.read_csv('CSTT_2021-01-27_2.csv')
    # pyspark_df = spark.createDataFrame(df)
    # df = spark.createDataFrame(
    #     [(1.0, 2.0, None), (3.0, float("nan"), 4.0), (5.0, 6.0, 7.0)], ["a", "b", "c"])
    # df.drop('customer-customer_id')
    # Trains a k-means model.
    score_list = list()
    silhouette_init = -1
    for k in range(3, 8):
        kmeans = KMeans(featuresCol='scaled').setK(k).setSeed(1)
        model = kmeans.fit(df_tr)

        # Make predictions
        predictions = model.transform(df_tr)

        # Evaluate clustering by computing Silhouette score
        evaluator = ClusteringEvaluator(featuresCol='scaled')

        silhouette_tmp = evaluator.evaluate(predictions)
        print("Silhouette with squared euclidean distance = " + str(silhouette_tmp))

        if silhouette_tmp > silhouette_init:
            best_k = k
            silhouette_int = silhouette_tmp
            best_kmeans = model
            res = predictions.select('prediction').collect()
        score_list.append([k, silhouette_tmp])
    ans = str(res) + '+' + str(df.columns)

    # Shows the result.
    print(score_list)
    # centers = model.clusterCenters()
    # print("Cluster Centers: ")
    # for center in centers:
    #     print(center)
    # $example off$

    spark.stop()

if __name__ == "__main__":
