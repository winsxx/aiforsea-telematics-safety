# AI For SEA - Safety
This project is for ["AI For SEA" Challenge](https://www.aiforsea.com). 

## Setup data
The data files should be structured in this format.
```
./data
    ./features
        [filename1].csv
        [filename2].csv
    ./labels
        [filename2].csv
```
To setup the data provided by Grab, run `./setup-1-dataset`

## Notebook
1. Open project directory
2. Setup data in `./data` directory. Run `./setup-1-dataset` for default dataset. 
3. Create virtual environment with **python 3** either with `virtualenv` or `conda`
    * `virtualenv -p pyton3 venv`, then `source venv/bin/activate`
    * or, `conda create -n venv python=3.6`, then `source activate venv`
4. Run `pip install -r requirements.txt`
5. Open notebook with **Jupyter Notebook**, **Jupyter Lab**, or other tools that support `.ipynb`
6. Re-run all the cells

## Train and Evaluate Model



