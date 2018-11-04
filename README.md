# bd4h-sleepdata
Single channel EEG sleep stage scoring using neural networks

https://physionet.org/physiobank/database/sleep-edfx/

https://sleepdata.org/datasets

Proposal: https://docs.google.com/document/d/1lojRzTHTZVBpYQdOLJnpGNFtkVIb-YAWktPaLCkSsmM/edit?usp=sharing

(?) Physionet --> Load EDFs --> PySpark ETL --> Feature files --> PyTorch
```
conda env create -f environment.yml
source activate bd4hproj
python sleep_scoring
```
