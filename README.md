

# HOTS using diffusive memristor relaxation curves

This repository provides Python code for Hierarchy of Time Surfaces (HOTS) processing of N-MNIST events using measured diffusive-memristor relaxation curves. Python scripts are modified based on paper "HOTS: A Hierarchy of Event-Based Time-Surfaces for Pattern Recognition, Xavier Lagorce, Garrick Orchard, Francesco Galluppi, Bertram E. Shi, IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 39, NO. 7, JULY 2017", please check the original script for your research.

## 1. Download the files

Download the repository using **Code → Download ZIP**, then extract it. Keep these files together:

- `Hots_main.py`: main program.
- `Hotslib.py`: HOTS processing and classification functions.
- `layer1.xlsx` and `layer2.xlsx`: measured relaxation curves for the first and second HOTS layers.

## 2. Install dependencies

Install Python 3 and run:

```bash
python -m pip install numpy scipy pandas openpyxl scikit-learn joblib numba
```

## 3. Prepare the N-MNIST dataset

You can download the N-MNIST dataset from https://www.garrickorchard.com/datasets/n-mnist
Download N-MNIST and prepare the training and test data as MATLAB files readable by `scipy.io.loadmat`:

- `train_set.mat`, containing a variable named `train_set`.
- `test_set.mat`, containing a variable named `test_set`.

Place them in an `N-MNIST` folder:

```text
HOTS-using-diffusive-memristor/
├── Hots_main.py
├── Hotslib.py
├── layer1.xlsx
├── layer2.xlsx
└── N-MNIST/
    ├── train_set.mat
    └── test_set.mat
```


## 4. Run the program

Open a terminal in the folder containing `Hots_main.py` and run:

```bash
python Hots_main.py
```



## Scope of this release

This script runs a software HOTS pipeline using experimental relaxation curves. It does not control the physical hardware or include the two-layer MLP training and evaluation used in the manuscript. 
