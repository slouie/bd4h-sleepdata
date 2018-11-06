from pyspark.sql import SparkSession


def start_spark():
    spark_builder = SparkSession.builder.master('local[*]').appName('bd4h-sleepdata')
    config = {
        'spark.executor.memory'         : '8G',
        'spark.driver.memory'           : '2G',
        'spark.serializer'              : 'org.apache.spark.serializer.KyroSerializer',
        'spark.kyroserializer.buffer'   : '24'
    }
    for k, v in config.items():
        spark_builder.config(k, v)
    spark = spark_builder.getOrCreate()
    return spark
