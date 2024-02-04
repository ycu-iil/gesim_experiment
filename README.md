# gesim_experiment

## Package requirements

- Python: 3.11
- GESim:
- RDKit: 2023.9.1
- Polars: 0.19.15
- pandas: 2.2.0
- matplotlib: 3.8.2
- seaborn: 0.13.2
- jupyter: 1.0.0
- networkx: 3.2.1

## How to setup

```bash
git clone git@github.com:ycu-iil/gesim_experiment.git
cd gesim_experiment
conda create -n gesim_expt python=3.11
# Switch Python virtual environment to gesim_expt
pip install poetry==1.7.1
poetry install --no-root

git clone git@github.com:LazyShion/graph_entropy.git
pip install --upgrade .
```

## How to run virtual screening experiment

```bash
cd gesim_experiment
git -C data/ clone git@github.com:rdkit/benchmarking_platform.git
python prepare_dataset.py
python calculate_similarities.py
python calculate_scores.py 
```
