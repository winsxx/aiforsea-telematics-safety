# AI For SEA - Safety
This project is for ["AI For SEA" Challenge](https://www.aiforsea.com). 

## Setup Data
The data files should be structured in this format.
```
./data
    ./features
        [filename1].csv
        [filename2].csv
    ./labels
        [filename2].csv
```
To setup the data provided by Grab, run `./setup-1-dataset`. You could cancel and re-run if it is stuck. The download will continue.

## Setup Environment
TLDR, you could use `./setup-2-environment` to setup, but you still need to activate the environment yourself. We need to make an isolated environment and install the requirements.  For manual setup:
1. Create virtual environment with **python 3** either with `virtualenv` or `conda`
    * `virtualenv -p python3 venv`, then `source venv/bin/activate`
    * or, `conda create -n venv python=3.6`, then `source activate venv`
2. Run `pip install -r requirements.txt`

## Notebook
To view the notebook without re-running the script, open `2019-06-09+Telematics+Safety+Analysis.html` file in project root.

To re-run the notebook:
1. Setup data in `./data` directory.
2. Setup environment.
5. Open notebook with `jupyter notebook`, or other tools that support `.ipynb`
6. Re-run all the cells.

## Train, Predict and Evaluate Model
1. Setup data in `./data` directory.
2. Setup environment.
3. Make sure to put your test data in `./data-test` directory with structure mentioned above.
4. Run `./model-predict -d "./data-test" -m "./model/safety_model_cnn_rf_stack.mdl" -o "./output/test_prediction.csv"` to predict the test data. Run `./model-predict -h` for more information.
5. To evaluate, use `./model-evaluate [prediction_csv_file_path] [test_label_csv_file_path]`. Example: `./model-evaluate output/test_prediction.csv data-test-label/sample-labels.csv`
6. (Optional) If you want to re-train the model, run `./model-train cnn-rf-stack -v 0.3 -s 6000 -d "./data" -m "./model/safety_model_cnn_rf_stack.mdl"` to train model with 70% 
data and validate with 30% data. Add `-s 6000` parameter to only use 6000 bookings for training and validation. 