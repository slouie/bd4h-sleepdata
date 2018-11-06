class FeatureConstruction(object):

    @staticmethod
    def construct(rdd):
        return rdd


def load_rdd_from_edf(spark_session, edf_paths):
    sc = spark_session.sparkContext
    for psg_path, hypno_path in edf_paths:
        pass
    return sc.parallelize([1,2,3,4,5])