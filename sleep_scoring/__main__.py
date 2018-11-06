from features.edf_loader import PhysiobankEDFLoader
from features.etl_edf import FeatureConstruction, load_rdd_from_edf
from helper import spark_helper


if __name__ == "__main__":
    spark_session = spark_helper.start_spark()
    sc = spark_session.sparkContext

    loader = PhysiobankEDFLoader()
    records = loader.load_sc_records(save=True)
    loader.print_record(records[0][0])
    # loader.print_record(records[0][1])

    rdd = load_rdd_from_edf(sc, [records[0]])
    features = FeatureConstruction.construct(sc, rdd)
    print(len(features[0]))
    
    
    