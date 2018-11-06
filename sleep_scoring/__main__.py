from features.edf_loader import PhysiobankEDFLoader
from features.etl_edf import FeatureConstruction, load_rdd_from_edf
from helper import spark_helper


if __name__ == "__main__":
    loader = PhysiobankEDFLoader()
    records = loader.load_sc_records(save=True)

    spark_session = spark_helper.start_spark()
    rdd = load_rdd_from_edf(spark_session, records)
    features = FeatureConstruction.construct(rdd)
    print(len(features))
    
    