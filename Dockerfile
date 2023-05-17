FROM 172.18.166.168/spark/spark-py:v3.1.1.1
ADD test.py /opt/spark/examples/src/main/python/
ADD mysql-connector-java-8.0.30.jar /opt/spark/jars/