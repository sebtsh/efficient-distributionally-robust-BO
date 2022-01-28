# Efficient Distributionally Robust Bayesian Optimization with Worst-case Sensitivity
The code for the paper "Efficient Distributionally Robust Bayesian Optimization with Worst-case Sensitivity".

## Requirements
1. Linux machine (experiments were run on Ubuntu 18.04.5 LTS and Ubuntu 20.04.3 LTS)
2. Python 3.7

## Setup
In the main directory, run the following command to install the required libraries.
```shell
pip install -r requirements.txt
```

## Running experiments
The experiment scripts may be found in the `experiments` folder, and may be run with the following commands in the main directory.
Random functions from GP prior:
```shell
python experiments/rand_func_bigexp.py with default
```
Plant maximum leaf area:
```shell
python experiments/plant_bigexp.py with default
```
Wind power dataset:
```shell
python experiments/wind_bigexp.py with default
```
COVID-19 test allocation:
```shell
python experiments/covid_bigexp.py with default
```
Computation time:
```shell
python experiments/timing.py with default
```
```shell
python experiments/pareto.py with default
```
