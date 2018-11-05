from features.edf_loader import PhysiobankEDFLoader

if __name__ == "__main__":
    loader = PhysiobankEDFLoader()
    records = loader.load_sc_records(save=True)
    loader.print_record(records[0][0])
    loader.print_record(records[0][1])