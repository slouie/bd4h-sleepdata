### Requirements
- Python 3.6
- PyTorch >= 0.4.1
- PySpark 2.3.1

### Running
# First, set up your Python environment
conda env create -f environment.yml
source activate bd4hproj

# To test an existing best model,
python sleep_scoring --best_model [modelName]

# To train a model,
python sleep_scoring --train [modelName]

`modelName` can be one of:
- SimpleCNN
- SimpleRNN
- RCNN

*Note: training a model will write a new model file under output/bestmodels/, potentially overwriting an existing bestmodel*

# To output plots use `--plot` flag.

# Example uses
- python sleep_scoring --best_model SimpleCNN --plot
- python sleep_scoring --train SimpleCNN --plot