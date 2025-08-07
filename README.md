# CM3
Source code for the proposed ($CM^3$) is located at `src/models/cm3.py`.

## Dependencies
The script has been tested running under Python 3.7.11, with the following packages installed (along with their dependencies):
- `torch: 1.11.0+cu113`
- `pandas: 1.3.5`
- `pyyaml: 6.0`
- `numpy: 1.21.5`
- `scipy: 1.7.3`
- `sentence-transformers: 2.2.0`

## Datasets 
- The model is based on MMRec. Please refer to the datasets used in the MMRec framework on GitHub.

## How to run
1. Put your downloaded data (e.g. `baby`) under `data` dir.
2. Enter `src` folder and run with  
`python main.py -m CM3 -d baby`  
You may specify other parameters in CMD or config with `configs/model/*.yaml` and `configs/dataset/*.yaml`.

