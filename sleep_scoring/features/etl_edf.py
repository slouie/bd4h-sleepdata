class FeatureConstruction(object):

    @staticmethod
    def construct(rdd):
        return rdd


def load_rdd_from_edf(spark_session, edf_paths):
    for psg_path, hypno_path in edf_paths:
        pass
    return spark_session.parallelize([1,2,3,4,5])