# bd4h-sleepdata
Single channel EEG sleep stage scoring using neural networks

https://physionet.org/physiobank/database/sleep-edfx/

https://sleepdata.org/datasets

Proposal: https://docs.google.com/document/d/1lojRzTHTZVBpYQdOLJnpGNFtkVIb-YAWktPaLCkSsmM/edit?usp=sharing

(?) Physionet --> Load EDFs --> PySpark ETL --> Feature files --> PyTorch
```
conda env create -f environment.yml
export SPARK_HOME=/usr/local/conda3/envs/bd4hproj/lib/python3.6/site-packages/pyspark
source activate bd4hproj
python sleep_scoring
python -m unittest discover
```

## Draft due 11/11
### Project execution
Once your project is approved, you should quickly work on
getting results and iterate with your sponsors on the progress.
Iteration  is  the  key.  The  first  iteration  should  be  fast  and
positive otherwise you are at risk losing momentum from the
sponsors/project owners (e.g., your boss, clinical experts, your
partners from another organization). This successful execution
will  lead  to  long-term  sustainability  of  your  team  and  will
greatly improve your reputation in the organization, so please
focus on that.

1)  Gather data that will be used in your project if you haven’t
already.
2)  Design the study (e.g., define cohort, target and features;
carefully   split   data   into   training,   validation   to   avoid
overfitting)
3)  Clean and process the data.
4)  Develop and implement the modeling pipeline.
5)  Evaluate  the  model  candidates  on  the  performance  met-
rics.
6)  Interpret the results from your model (e.g., show predic-
tive  features,  compare  to  literature  in  terms  of  finding,
present as cool visualization).

### Deliverables
- Up to 5-page write-up + 1 page of references
- Guide
  - Make sure your write-up cover all aspects described in
project execution.
  - Conduct literature search and cite at least 8 papers or
more that are relevant to the project.

All  the  steps  in  project  execution  should  be  done  by  the
paper  draft  due  date  and  iterate  at  least  another  time  by  the
final due day.
